/*
 * Compiles the public header as C and drives the whole API against the linked
 * library: build, lower, read the tables, run a trace, and take the failures a
 * front-end has to report.
 *
 * This is the only test that proves the header and the library agree — in
 * particular that the sizes it declares for the objects are big enough and its
 * enumerators line up, neither of which anything on the Zig side can check.
 */

#include "sva.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

/* Every builder can fail, and only for want of memory. */
#define BUILD(call) CHECK((call) == SVA_OK)

enum { ATOM_REQ = 0, ATOM_ACK = 1, ATOM_RESET = 2 };

/* A one-cycle sequence over an atom. */
static int atom_cycle(sva_ir *ir, uint32_t atom, sva_node *out) {
    sva_node boolean;
    BUILD(sva_bool_atom(ir, atom, &boolean));
    BUILD(sva_seq_boolean(ir, boolean, out));
    return 0;
}

/* `##[lo:hi] right` from the start of an attempt: a one-cycle `true` to the
   left, which is the only way to spell it now that no edge is optional. */
static int leading_delay(sva_ir *ir, uint32_t lo, uint32_t hi, sva_node right,
                        sva_node *out) {
    sva_node always, head;
    BUILD(sva_bool_const(ir, true, &always));
    BUILD(sva_seq_boolean(ir, always, &head));
    BUILD(sva_seq_delay(ir, head, lo, hi, right, out));
    return 0;
}

/* An assertion that never resets, which is a constant false rather than an
   absent edge. */
static int add_directive(sva_ir *ir, sva_node property, sva_node reset,
                        uint32_t *out) {
    sva_directive directive;
    memset(&directive, 0, sizeof directive);
    directive.id = 42;
    directive.kind = SVA_ASSERT;
    directive.clock = SVA_CLOCK_NONE;
    directive.property = property;
    if (reset == UINT32_MAX) {
        BUILD(sva_bool_const(ir, false, &directive.reset));
    } else {
        directive.reset = reset;
    }
    BUILD(sva_ir_add(ir, &directive, out));
    return 0;
}

/* Map each root back to the atom it wants. A root is a boolean expression in
   general, so a backend walks it; the reset here is a constant, not an atom. */
static int resolve_roots(const sva_ir *ir, const sva_monitor *monitor,
                        uint32_t *root_of_atom, uint32_t atoms) {
    for (uint32_t atom = 0; atom < atoms; atom++) root_of_atom[atom] = UINT32_MAX;
    const sva_node *roots = sva_monitor_roots(monitor);
    for (uint32_t root = 0; root < sva_monitor_root_count(monitor); root++) {
        sva_bool_view view;
        sva_bool_read(ir, roots[root], &view);
        if (view.op != SVA_BOOL_ATOM) continue;
        CHECK(view.lhs < atoms);
        root_of_atom[view.lhs] = root;
    }
    for (uint32_t atom = 0; atom < atoms; atom++) CHECK(root_of_atom[atom] != UINT32_MAX);
    return 0;
}

/* Build `req |=> ack ##[1:2] ack`, lower it, read it the way a code generator
   would, and run it. */
static int implication(void) {
    sva_ir ir;
    sva_ir_init(&ir);

    sva_node req, ack, tail, body, consequent, property;
    if (atom_cycle(&ir, ATOM_REQ, &req) != 0) return 1;
    if (atom_cycle(&ir, ATOM_ACK, &ack) != 0) return 1;
    if (atom_cycle(&ir, ATOM_ACK, &tail) != 0) return 1;
    BUILD(sva_seq_delay(&ir, ack, 1, 2, tail, &body));
    BUILD(sva_prop_sequence(&ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(&ir, SVA_NONOVERLAPPING, req, consequent, &property));

    uint32_t index;
    if (add_directive(&ir, property, UINT32_MAX, &index) != 0) return 1;
    CHECK(sva_ir_atom_count(&ir) == 2);
    CHECK(sva_ir_node_count(&ir) > 0);

    sva_monitor monitor;
    CHECK(sva_lower(&monitor, &ir, index, NULL, NULL) == SVA_OK);

    CHECK(sva_monitor_id(&monitor) == 42);
    CHECK(sva_monitor_kind(&monitor) == SVA_ASSERT);
    CHECK(sva_monitor_clock(&monitor) == SVA_CLOCK_NONE);
    CHECK(sva_monitor_implication(&monitor) == SVA_NONOVERLAPPING);
    CHECK(sva_monitor_strength(&monitor) == SVA_WEAK);
    /* `ack ##[1:2] ack` spans three cycles at most, so three attempts can be in
       flight and a backend allocates three state vectors. */
    CHECK(sva_monitor_depth(&monitor) == 3);
    /* Two atoms plus the constant false that says the assertion never resets. */
    CHECK(sva_monitor_root_count(&monitor) == 3);
    CHECK(sva_monitor_reset(&monitor) < sva_monitor_root_count(&monitor));
    CHECK(sva_monitor_trigger(&monitor) != NULL);

    /* Every guard is a cube over the roots, and guard 0 is empty. */
    uint32_t guards = sva_monitor_guard_count(&monitor);
    CHECK(guards >= 2);
    uint32_t count = 7;
    CHECK(sva_monitor_guard(&monitor, 0, &count) == NULL);
    CHECK(count == 0);
    for (uint32_t guard = 2; guard < guards; guard++) {
        const sva_literal *literals = sva_monitor_guard(&monitor, guard, &count);
        CHECK(count > 0 && literals != NULL);
        for (uint32_t i = 0; i < count; i++) {
            CHECK(SVA_LITERAL_ROOT(literals[i]) < sva_monitor_root_count(&monitor));
            (void)SVA_LITERAL_NEGATED(literals[i]);
        }
    }

    /* Every state names a guard in range, and no edge names a state that is not
       there. */
    const sva_automaton *automata[2];
    automata[0] = sva_monitor_obligation(&monitor);
    automata[1] = sva_monitor_trigger(&monitor);
    for (int which = 0; which < 2; which++) {
        const sva_automaton *automaton = automata[which];
        uint32_t states = sva_automaton_state_count(automaton);
        uint32_t words = sva_automaton_words(automaton);
        CHECK(states > 0);
        CHECK(words == (states + 63) / 64);
        CHECK(sva_automaton_first(automaton) != NULL);
        CHECK(sva_automaton_last(automaton) != NULL);

        const uint32_t *state_guards = sva_automaton_guards(automaton);
        for (uint32_t state = 0; state < states; state++) {
            CHECK(state_guards[state] < guards);
            CHECK(sva_automaton_guard_states(automaton, state_guards[state]) != NULL);
            const sva_word *row = sva_automaton_follow(automaton, state);
            for (uint32_t word = 0; word < words; word++) {
                uint32_t top = states - word * 64;
                if (top < 64) CHECK((row[word] >> top) == 0);
            }
        }
    }

    /* Drive the reference runtime. */
    uint32_t root_of_atom[2];
    if (resolve_roots(&ir, &monitor, root_of_atom, 2) != 0) return 1;

    sva_run run;
    CHECK(sva_run_init(&run, &monitor) == SVA_OK);

    struct { bool req, ack; uint32_t failures, matches; } trace[] = {
        /* req raises an obligation for the next cycle. */
        { true,  false, 0, 0 },
        /* ack, but `ack ##[1:2] ack` needs a second one. */
        { false, true,  0, 0 },
        /* the second ack discharges it. */
        { false, true,  0, 1 },
        /* req again, and then nothing: the obligation fails at once. */
        { true,  false, 0, 0 },
        { false, false, 1, 0 },
    };
    for (size_t cycle = 0; cycle < sizeof trace / sizeof trace[0]; cycle++) {
        sva_word bits = 0;
        if (trace[cycle].req) bits |= (sva_word)1 << root_of_atom[ATOM_REQ];
        if (trace[cycle].ack) bits |= (sva_word)1 << root_of_atom[ATOM_ACK];
        sva_report report = sva_run_step(&run, &bits);
        CHECK(report.failures == trace[cycle].failures);
        CHECK(report.matches == trace[cycle].matches);
        CHECK(!report.reset);
    }
    /* Weak, so nothing left in flight fails. */
    CHECK(sva_run_finish(&run).failures == 0);

    printf("ok: %u roots, %u guards, depth %u; sizeof ir/monitor/run = %zu/%zu/%zu\n",
           sva_monitor_root_count(&monitor), guards, sva_monitor_depth(&monitor),
           sizeof(sva_ir), sizeof(sva_monitor), sizeof(sva_run));

    sva_run_deinit(&run);
    sva_monitor_deinit(&monitor);
    sva_ir_deinit(&ir);
    return 0;
}

/* A root can be any boolean expression, so a backend walks it rather than
   assuming an atom. */
static int walk(void) {
    sva_ir ir;
    sva_ir_init(&ir);

    sva_node a, b, negated, expression;
    BUILD(sva_bool_atom(&ir, 3, &a));
    BUILD(sva_bool_atom(&ir, 4, &b));
    BUILD(sva_bool_not(&ir, b, &negated));
    BUILD(sva_bool_or(&ir, a, negated, &expression));

    sva_bool_view view;
    sva_bool_read(&ir, expression, &view);
    CHECK(view.op == SVA_BOOL_OR);
    sva_node left = view.lhs, right = view.rhs;

    sva_bool_read(&ir, left, &view);
    CHECK(view.op == SVA_BOOL_ATOM && view.lhs == 3);
    sva_bool_read(&ir, right, &view);
    CHECK(view.op == SVA_BOOL_NOT);
    sva_bool_read(&ir, view.lhs, &view);
    CHECK(view.op == SVA_BOOL_ATOM && view.lhs == 4);

    sva_node constant, conjunction;
    BUILD(sva_bool_const(&ir, true, &constant));
    sva_bool_read(&ir, constant, &view);
    CHECK(view.op == SVA_BOOL_CONST && view.lhs == 1);
    BUILD(sva_bool_and(&ir, a, b, &conjunction));
    sva_bool_read(&ir, conjunction, &view);
    CHECK(view.op == SVA_BOOL_AND);

    sva_ir_deinit(&ir);
    return 0;
}

/* A bare sequence property has no trigger and starts an obligation every cycle.
   Plus a reset, which is a root of its own. */
static int shapes(void) {
    {
        sva_ir ir;
        sva_ir_init(&ir);

        sva_node a, b, pattern, property;
        if (atom_cycle(&ir, 0, &a) != 0) return 1;
        if (atom_cycle(&ir, 0, &b) != 0) return 1;
        BUILD(sva_seq_delay(&ir, a, 1, 1, b, &pattern));
        BUILD(sva_prop_sequence(&ir, SVA_STRONG, pattern, &property));

        uint32_t index;
        if (add_directive(&ir, property, UINT32_MAX, &index) != 0) return 1;

        sva_monitor monitor;
        CHECK(sva_lower(&monitor, &ir, index, NULL, NULL) == SVA_OK);
        CHECK(sva_monitor_trigger(&monitor) == NULL);
        CHECK(sva_monitor_strength(&monitor) == SVA_STRONG);
        CHECK(sva_monitor_depth(&monitor) == 2);
        /* One atom plus the constant false reset. */
        CHECK(sva_monitor_root_count(&monitor) == 2);

        uint32_t root_of_atom[1];
        if (resolve_roots(&ir, &monitor, root_of_atom, 1) != 0) return 1;

        sva_run run;
        CHECK(sva_run_init(&run, &monitor) == SVA_OK);
        /* `a ##1 a` over a held `a`: an attempt completes every cycle from the
           second on, and the newest is still pending at the end. Being strong,
           it fails. The atom's bit has to be looked up: the reset is a root too,
           so bit positions are not ours to assume. */
        sva_word held = (sva_word)1 << root_of_atom[0];
        CHECK(sva_run_step(&run, &held).matches == 0);
        CHECK(sva_run_step(&run, &held).matches == 1);
        CHECK(sva_run_step(&run, &held).matches == 1);
        CHECK(sva_run_finish(&run).failures == 1);

        sva_run_deinit(&run);
        sva_monitor_deinit(&monitor);
        sva_ir_deinit(&ir);
    }
    {
        sva_ir ir;
        sva_ir_init(&ir);

        sva_node req, ack, consequent, property, reset;
        if (atom_cycle(&ir, ATOM_REQ, &req) != 0) return 1;
        if (atom_cycle(&ir, ATOM_ACK, &ack) != 0) return 1;
        BUILD(sva_prop_sequence(&ir, SVA_WEAK, ack, &consequent));
        BUILD(sva_prop_implies(&ir, SVA_NONOVERLAPPING, req, consequent, &property));
        BUILD(sva_bool_atom(&ir, ATOM_RESET, &reset));

        uint32_t index;
        if (add_directive(&ir, property, reset, &index) != 0) return 1;

        sva_monitor monitor;
        CHECK(sva_lower(&monitor, &ir, index, NULL, NULL) == SVA_OK);
        CHECK(sva_monitor_reset(&monitor) < sva_monitor_root_count(&monitor));
        /* Two atoms plus the reset. */
        CHECK(sva_monitor_root_count(&monitor) == 3);

        uint32_t root_of_atom[3];
        if (resolve_roots(&ir, &monitor, root_of_atom, 3) != 0) return 1;
        /* The reset root names the boolean the directive was given. */
        CHECK(root_of_atom[ATOM_RESET] == sva_monitor_reset(&monitor));

        sva_run run;
        CHECK(sva_run_init(&run, &monitor) == SVA_OK);

        /* req, then a reset instead of the ack that would have been required. */
        sva_word bits = (sva_word)1 << root_of_atom[ATOM_REQ];
        sva_report report = sva_run_step(&run, &bits);
        CHECK(report.failures == 0 && !report.reset);

        bits = (sva_word)1 << root_of_atom[ATOM_RESET];
        report = sva_run_step(&run, &bits);
        CHECK(report.reset && report.failures == 0);

        /* And the abandoned obligation does not come back. */
        bits = 0;
        report = sva_run_step(&run, &bits);
        CHECK(report.failures == 0 && !report.reset);
        CHECK(sva_run_finish(&run).failures == 0);

        sva_run_deinit(&run);
        sva_monitor_deinit(&monitor);
        sva_ir_deinit(&ir);
    }
    return 0;
}

/* The failures a front-end has to report: legal input libsva cannot lower yet,
   named by node. A malformed tree is not among them — that aborts, because it is
   a bug in the lowering rather than in the source. */
static int errors(void) {
    sva_ir ir;
    sva_ir_init(&ir);

    /* req |=> ##[1:$] ack has no bounded obligation. */
    sva_node ack, body, req, weak, unbounded;
    if (atom_cycle(&ir, ATOM_ACK, &ack) != 0) return 1;
    if (leading_delay(&ir, 1, SVA_UNBOUNDED, ack, &body) != 0) return 1;
    if (atom_cycle(&ir, ATOM_REQ, &req) != 0) return 1;
    BUILD(sva_prop_sequence(&ir, SVA_WEAK, body, &weak));
    BUILD(sva_prop_implies(&ir, SVA_NONOVERLAPPING, req, weak, &unbounded));

    uint32_t first;
    if (add_directive(&ir, unbounded, UINT32_MAX, &first) != 0) return 1;

    /* And a wide delay range: many filler states, but a longest match under the
       depth cap, which is checked first and would otherwise mask the state cap. */
    sva_node tail, wide, bare;
    if (atom_cycle(&ir, ATOM_ACK, &tail) != 0) return 1;
    BUILD(sva_seq_delay(&ir, req, 1, 100, tail, &wide));
    BUILD(sva_prop_sequence(&ir, SVA_WEAK, wide, &bare));

    uint32_t second;
    if (add_directive(&ir, bare, UINT32_MAX, &second) != 0) return 1;

    sva_monitor monitor;
    sva_diagnostic reason;
    CHECK(sva_lower(&monitor, &ir, first, NULL, &reason) ==
          SVA_ERROR_UNBOUNDED_OBLIGATION);
    CHECK(reason.node == body);

    sva_options options;
    memset(&options, 0, sizeof options);
    options.state_limit = 64;
    CHECK(sva_lower(&monitor, &ir, second, &options, &reason) == SVA_ERROR_STATE_LIMIT);
    CHECK(reason.node == wide);
    /* One state per cycle the delay may skip, plus the two matches. */
    CHECK(reason.size == 101);

    sva_ir_deinit(&ir);
    return 0;
}

int main(void) {
    if (implication() != 0) return 1;
    if (walk() != 0) return 1;
    if (shapes() != 0) return 1;
    if (errors() != 0) return 1;
    return 0;
}
