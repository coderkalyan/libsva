/*
 * Compiles the public header as C and drives the whole API against the linked
 * library: build, lower, read the tables, run a trace, and take the failures a
 * front-end has to report.
 *
 * This is the only test that proves the header and the library agree — in
 * particular that SVA_ABI_VERSION matches the version the archive reports, and
 * that the enumerators line up, neither of which anything on the Zig side can
 * check.
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
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    sva_node req, ack, tail, body, consequent, property;
    if (atom_cycle(ir, ATOM_REQ, &req) != 0) return 1;
    if (atom_cycle(ir, ATOM_ACK, &ack) != 0) return 1;
    if (atom_cycle(ir, ATOM_ACK, &tail) != 0) return 1;
    BUILD(sva_seq_delay(ir, ack, 1, 2, tail, &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));

    uint32_t index;
    if (add_directive(ir, property, UINT32_MAX, &index) != 0) return 1;
    CHECK(sva_ir_atom_count(ir) == 2);
    CHECK(sva_ir_node_count(ir) > 0);

    sva_monitor *monitor;
    CHECK(sva_lower(&monitor, ir, index, NULL, NULL) == SVA_OK);

    CHECK(sva_monitor_id(monitor) == 42);
    CHECK(sva_monitor_kind(monitor) == SVA_ASSERT);
    CHECK(sva_monitor_clock(monitor) == SVA_CLOCK_NONE);
    CHECK(sva_monitor_implication(monitor) == SVA_NONOVERLAPPING);
    CHECK(sva_monitor_strength(monitor) == SVA_WEAK);
    /* `ack ##[1:2] ack` spans three cycles at most, so three attempts can be in
       flight and a backend allocates three state vectors. */
    CHECK(sva_monitor_depth(monitor) == 3);
    /* Two atoms plus the constant false that says the assertion never resets. */
    CHECK(sva_monitor_root_count(monitor) == 3);
    CHECK(sva_monitor_reset(monitor) < sva_monitor_root_count(monitor));
    CHECK(sva_monitor_trigger(monitor) != NULL);
    /* `req` matches in one cycle, so only the attempt started in a cycle can have
       its antecedent end there and one bit counts them. Zero window, and the id
       machinery folded away: the common case, asserted so it stays that way. */
    CHECK(sva_monitor_trigger_window(monitor) == 0);

    /* Every guard is a cube over the roots, and guard 0 is empty. */
    uint32_t guards = sva_monitor_guard_count(monitor);
    CHECK(guards >= 2);
    uint32_t count = 7;
    CHECK(sva_monitor_guard(monitor, 0, &count) == NULL);
    CHECK(count == 0);
    for (uint32_t guard = 2; guard < guards; guard++) {
        const sva_literal *literals = sva_monitor_guard(monitor, guard, &count);
        CHECK(count > 0 && literals != NULL);
        for (uint32_t i = 0; i < count; i++) {
            CHECK(SVA_LITERAL_ROOT(literals[i]) < sva_monitor_root_count(monitor));
            (void)SVA_LITERAL_NEGATED(literals[i]);
        }
    }

    /* Every state names a guard in range, and no edge names a state that is not
       there. */
    const sva_automaton *automata[2];
    automata[0] = sva_monitor_obligation(monitor);
    automata[1] = sva_monitor_trigger(monitor);
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
    if (resolve_roots(ir, monitor, root_of_atom, 2) != 0) return 1;

    sva_run *run = sva_run_create(monitor);
    CHECK(run != NULL);

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
        sva_report report = sva_run_step(run, &bits);
        CHECK(report.failures == trace[cycle].failures);
        CHECK(report.matches == trace[cycle].matches);
        CHECK(!report.reset);
    }
    /* Weak, so nothing left in flight fails. */
    CHECK(sva_run_finish(run).failures == 0);

    printf("ok: %u roots, %u guards, depth %u; abi %u\n",
           sva_monitor_root_count(monitor), guards, sva_monitor_depth(monitor),
           sva_abi_version());

    sva_run_destroy(run);
    sva_monitor_destroy(monitor);
    sva_ir_destroy(ir);
    return 0;
}

/* A root can be any boolean expression, so a backend walks it rather than
   assuming an atom. */
static int walk(void) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    sva_node a, b, negated, expression;
    BUILD(sva_bool_atom(ir, 3, &a));
    BUILD(sva_bool_atom(ir, 4, &b));
    BUILD(sva_bool_not(ir, b, &negated));
    BUILD(sva_bool_or(ir, a, negated, &expression));

    sva_bool_view view;
    sva_bool_read(ir, expression, &view);
    CHECK(view.op == SVA_BOOL_OR);
    sva_node left = view.lhs, right = view.rhs;

    sva_bool_read(ir, left, &view);
    CHECK(view.op == SVA_BOOL_ATOM && view.lhs == 3);
    sva_bool_read(ir, right, &view);
    CHECK(view.op == SVA_BOOL_NOT);
    sva_bool_read(ir, view.lhs, &view);
    CHECK(view.op == SVA_BOOL_ATOM && view.lhs == 4);

    sva_node constant, conjunction;
    BUILD(sva_bool_const(ir, true, &constant));
    sva_bool_read(ir, constant, &view);
    CHECK(view.op == SVA_BOOL_CONST && view.lhs == 1);
    BUILD(sva_bool_and(ir, a, b, &conjunction));
    sva_bool_read(ir, conjunction, &view);
    CHECK(view.op == SVA_BOOL_AND);

    sva_ir_destroy(ir);
    return 0;
}

/* A bare sequence property has no trigger and starts an obligation every cycle.
   Plus a reset, which is a root of its own. */
static int shapes(void) {
    {
        sva_ir *ir = sva_ir_create();
        CHECK(ir != NULL);

        sva_node a, b, pattern, property;
        if (atom_cycle(ir, 0, &a) != 0) return 1;
        if (atom_cycle(ir, 0, &b) != 0) return 1;
        BUILD(sva_seq_delay(ir, a, 1, 1, b, &pattern));
        BUILD(sva_prop_sequence(ir, SVA_STRONG, pattern, &property));

        uint32_t index;
        if (add_directive(ir, property, UINT32_MAX, &index) != 0) return 1;

        sva_monitor *monitor;
        CHECK(sva_lower(&monitor, ir, index, NULL, NULL) == SVA_OK);
        CHECK(sva_monitor_trigger(monitor) == NULL);
        CHECK(sva_monitor_strength(monitor) == SVA_STRONG);
        CHECK(sva_monitor_depth(monitor) == 2);
        /* One atom plus the constant false reset. */
        CHECK(sva_monitor_root_count(monitor) == 2);

        uint32_t root_of_atom[1];
        if (resolve_roots(ir, monitor, root_of_atom, 1) != 0) return 1;

        sva_run *run = sva_run_create(monitor);
        CHECK(run != NULL);
        /* `a ##1 a` over a held `a`: an attempt completes every cycle from the
           second on, and the newest is still pending at the end. Being strong,
           it fails. The atom's bit has to be looked up: the reset is a root too,
           so bit positions are not ours to assume. */
        sva_word held = (sva_word)1 << root_of_atom[0];
        CHECK(sva_run_step(run, &held).matches == 0);
        CHECK(sva_run_step(run, &held).matches == 1);
        CHECK(sva_run_step(run, &held).matches == 1);
        CHECK(sva_run_finish(run).failures == 1);

        sva_run_destroy(run);
        sva_monitor_destroy(monitor);
        sva_ir_destroy(ir);
    }
    {
        sva_ir *ir = sva_ir_create();
        CHECK(ir != NULL);

        sva_node req, ack, consequent, property, reset;
        if (atom_cycle(ir, ATOM_REQ, &req) != 0) return 1;
        if (atom_cycle(ir, ATOM_ACK, &ack) != 0) return 1;
        BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
        BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
        BUILD(sva_bool_atom(ir, ATOM_RESET, &reset));

        uint32_t index;
        if (add_directive(ir, property, reset, &index) != 0) return 1;

        sva_monitor *monitor;
        CHECK(sva_lower(&monitor, ir, index, NULL, NULL) == SVA_OK);
        CHECK(sva_monitor_reset(monitor) < sva_monitor_root_count(monitor));
        /* Two atoms plus the reset. */
        CHECK(sva_monitor_root_count(monitor) == 3);

        uint32_t root_of_atom[3];
        if (resolve_roots(ir, monitor, root_of_atom, 3) != 0) return 1;
        /* The reset root names the boolean the directive was given. */
        CHECK(root_of_atom[ATOM_RESET] == sva_monitor_reset(monitor));

        sva_run *run = sva_run_create(monitor);
        CHECK(run != NULL);

        /* req, then a reset instead of the ack that would have been required. */
        sva_word bits = (sva_word)1 << root_of_atom[ATOM_REQ];
        sva_report report = sva_run_step(run, &bits);
        CHECK(report.failures == 0 && !report.reset);

        bits = (sva_word)1 << root_of_atom[ATOM_RESET];
        report = sva_run_step(run, &bits);
        CHECK(report.reset && report.failures == 0);

        /* And the abandoned obligation does not come back. */
        bits = 0;
        report = sva_run_step(run, &bits);
        CHECK(report.failures == 0 && !report.reset);
        CHECK(sva_run_finish(run).failures == 0);

        sva_run_destroy(run);
        sva_monitor_destroy(monitor);
        sva_ir_destroy(ir);
    }
    return 0;
}

/* The failures a front-end has to report: legal input libsva cannot lower yet,
   named by node. A malformed tree is not among them — that aborts, because it is
   a bug in the lowering rather than in the source. */
static int errors(void) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    /* always (req |=> ack): a trigger strictly below an unbounded operator, so
       the implication's bounded obligation wants a ring and the operator above it
       wants counters. Tracking mode is one per monitor, so it is refused.

       `req |=> ##[1:$] ack` used to live here and now lowers, in
       SVA_MODE_COUNTED: a trigger ABOVE an unbounded operator is fine. */
    sva_node ack, req, consequent, implication, mixed;
    if (atom_cycle(ir, ATOM_ACK, &ack) != 0) return 1;
    if (atom_cycle(ir, ATOM_REQ, &req) != 0) return 1;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &implication));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 0, SVA_UNBOUNDED, implication, &mixed));

    uint32_t first;
    if (add_directive(ir, mixed, UINT32_MAX, &first) != 0) return 1;

    /* And a wide delay range: many filler states, but a longest match under the
       depth cap, which is checked first and would otherwise mask the state cap. */
    sva_node tail, wide, bare;
    if (atom_cycle(ir, ATOM_ACK, &tail) != 0) return 1;
    BUILD(sva_seq_delay(ir, req, 1, 100, tail, &wide));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, wide, &bare));

    uint32_t second;
    if (add_directive(ir, bare, UINT32_MAX, &second) != 0) return 1;

    /* An unbounded antecedent: `(req ##[1:$] ack) |-> ack`. An obligation is owed
       per antecedent MATCH, so the cycle a match ends in has to say how many
       attempts ended one there, and this shape has unboundedly many matching at
       once for sva_monitor_trigger_window to tell apart. Refused, naming the
       antecedent, rather than answered with the number of matching cycles - which
       is a wrong number, not a coarse one. */
    sva_node open_end, open_antecedent, open_consequent, open_property;
    if (atom_cycle(ir, ATOM_ACK, &open_end) != 0) return 1;
    BUILD(sva_seq_delay(ir, req, 1, SVA_UNBOUNDED, open_end, &open_antecedent));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, open_end, &open_consequent));
    BUILD(sva_prop_implies(ir, SVA_OVERLAPPING, open_antecedent, open_consequent,
                           &open_property));

    uint32_t third;
    if (add_directive(ir, open_property, UINT32_MAX, &third) != 0) return 1;

    /* Every lowering below fails, so this is never written and there is nothing
       to destroy: a refused lowering hands back no monitor. */
    sva_monitor *monitor;
    sva_diagnostic reason;
    CHECK(sva_lower(&monitor, ir, first, NULL, &reason) == SVA_ERROR_MODE_MIX);
    CHECK(reason.node == mixed);

    sva_options options;
    memset(&options, 0, sizeof options);
    options.state_limit = 64;
    CHECK(sva_lower(&monitor, ir, second, &options, &reason) == SVA_ERROR_STATE_LIMIT);
    CHECK(reason.node == wide);
    /* One state per cycle the delay may skip, plus the two matches. */
    CHECK(reason.size == 101);

    CHECK(sva_lower(&monitor, ir, third, NULL, &reason) ==
          SVA_ERROR_UNBOUNDED_ANTECEDENT);
    CHECK(reason.node == open_antecedent);

    sva_ir_destroy(ir);
    return 0;
}

/* The counted discipline, and every export it added. `req |=> s_eventually ack`
   is the shape that used to be refused: a trigger above an unbounded obligation.
   Then `a until_with b`, which reports a match rather than only a failure, so
   both verdict paths of the counted step are driven from C. */
static int counted(void) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    sva_node ack, req, consequent, shifted, property;
    if (atom_cycle(ir, ATOM_ACK, &ack) != 0) return 1;
    if (atom_cycle(ir, ATOM_REQ, &req) != 0) return 1;
    BUILD(sva_prop_sequence(ir, SVA_STRONG, ack, &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 1, SVA_UNBOUNDED, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, shifted, &property));

    uint32_t index;
    if (add_directive(ir, property, UINT32_MAX, &index) != 0) return 1;

    sva_monitor *monitor;
    CHECK(sva_lower(&monitor, ir, index, NULL, NULL) == SVA_OK);
    CHECK(sva_monitor_mode(monitor) == SVA_MODE_COUNTED);
    /* Zero and meaningless here; a backend must not read it as a bound. */
    CHECK(sva_monitor_depth(monitor) == 0);
    CHECK(sva_monitor_trigger(monitor) != NULL);

    /* `s_eventually` can never fail inside the trace — a later `ack` is always
       possible — so it has no dead state, and the accessor still answers. */
    const sva_automaton *obligation = sva_monitor_obligation(monitor);
    CHECK(sva_automaton_words(obligation) == 1);
    CHECK(sva_automaton_dead(obligation)[0] == 0);
    CHECK(sva_automaton_last(obligation)[0] != 0);

    /* A trigger is a ring-free automaton either way, and its dead mask is empty
       because only a counted obligation has one. */
    CHECK(sva_automaton_dead(sva_monitor_trigger(monitor))[0] == 0);

    sva_run *run = sva_run_create(monitor);
    CHECK(run != NULL);
    /* req with no ack ever: the attempt stays pending, and being strong it fails
       when the trace ends. Nothing resolves inside it. */
    uint32_t req_root = UINT32_MAX;
    const sva_node *root_node = sva_monitor_roots(monitor);
    for (uint32_t r = 0; r < sva_monitor_root_count(monitor); r++) {
        sva_bool_view view;
        sva_bool_read(ir, root_node[r], &view);
        if (view.op == SVA_BOOL_ATOM && view.lhs == ATOM_REQ) req_root = r;
    }
    CHECK(req_root != UINT32_MAX);

    sva_word bits = (sva_word)1 << req_root;
    CHECK(sva_run_step(run, &bits).failures == 0);
    bits = 0;
    CHECK(sva_run_step(run, &bits).failures == 0);
    CHECK(sva_run_finish(run).failures == 1);
    sva_run_destroy(run);
    sva_monitor_destroy(monitor);

    /* The same absence of a bound reached through the *sequence* layer rather
       than through a property operator: `req |=> ##[1:$] ack`. This is the shape
       `errors` used to assert a refusal for. */
    sva_node body, weak, delayed;
    if (leading_delay(ir, 1, SVA_UNBOUNDED, ack, &body) != 0) return 1;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &weak));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, weak, &delayed));

    uint32_t delayed_index;
    if (add_directive(ir, delayed, UINT32_MAX, &delayed_index) != 0) return 1;
    sva_monitor *delayed_monitor;
    CHECK(sva_lower(&delayed_monitor, ir, delayed_index, NULL, NULL) == SVA_OK);
    CHECK(sva_monitor_mode(delayed_monitor) == SVA_MODE_COUNTED);
    CHECK(sva_monitor_depth(delayed_monitor) == 0);
    sva_monitor_destroy(delayed_monitor);

    /* `a until_with b`, over the four kinds so every enumerator is used. Only the
       last is run; building all four is what pins the constructor's signature. */
    const sva_until kinds[4] = { SVA_UNTIL_WEAK, SVA_UNTIL_WEAK_WITH, SVA_UNTIL_STRONG,
                                 SVA_UNTIL_STRONG_WITH };
    sva_node left, right, until = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, req, &left));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &right));
    for (unsigned k = 0; k < 4; k++) {
        BUILD(sva_prop_until(ir, kinds[k], left, right, &until));
    }

    uint32_t until_index;
    if (add_directive(ir, until, UINT32_MAX, &until_index) != 0) return 1;
    sva_monitor *until_monitor;
    CHECK(sva_lower(&until_monitor, ir, until_index, NULL, NULL) == SVA_OK);
    CHECK(sva_monitor_mode(until_monitor) == SVA_MODE_COUNTED);
    /* Strong, because the kind carries the disposition rather than the operands:
       both of these are weak sequence properties. */
    CHECK(sva_monitor_strength(until_monitor) == SVA_STRONG);
    /* An until can both match and die, so both masks are populated. */
    const sva_automaton *until_obligation = sva_monitor_obligation(until_monitor);
    CHECK(sva_automaton_last(until_obligation)[0] != 0);
    CHECK(sva_automaton_dead(until_obligation)[0] != 0);
    sva_monitor_destroy(until_monitor);

    sva_ir_destroy(ir);
    return 0;
}

/* Every export phase 4 added, driven rather than merely declared: the
   constructor, both new junction kinds, both verdict calls, and the two
   refusals that are properties of the operators rather than of any operand. */
static int complement(void) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    sva_node req, ack;
    if (atom_cycle(ir, ATOM_REQ, &req) != 0) return 1;
    if (atom_cycle(ir, ATOM_ACK, &ack) != 0) return 1;

    /* The negation table, including the row that is the whole reason it is in
       the ABI: negating an undecided attempt leaves it undecided. */
    CHECK(sva_verdict_not(SVA_VERDICT_HOLDS) == SVA_VERDICT_FAILS);
    CHECK(sva_verdict_not(SVA_VERDICT_FAILS) == SVA_VERDICT_HOLDS);
    CHECK(sva_verdict_not(SVA_VERDICT_PENDING) == SVA_VERDICT_PENDING);

    /* `not weak(req ##1 ack)`. Weak in, strong out. */
    sva_node body, inner, negated;
    BUILD(sva_seq_delay(ir, req, 1, 1, ack, &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &inner));
    BUILD(sva_prop_not(ir, inner, &negated));

    uint32_t negated_index;
    if (add_directive(ir, negated, UINT32_MAX, &negated_index) != 0) return 1;
    sva_monitor *monitor;
    CHECK(sva_lower(&monitor, ir, negated_index, NULL, NULL) == SVA_OK);
    CHECK(sva_monitor_strength(monitor) == SVA_STRONG);
    CHECK(sva_monitor_pending(monitor) == SVA_VERDICT_FAILS);
    /* A complement of a bounded operand does NOT select the counted mode. It
       borrows the operand's bound, and the operand resolves within its own
       longest match however the complement reads it. */
    CHECK(sva_monitor_mode(monitor) == SVA_MODE_RING);
    CHECK(sva_monitor_depth(monitor) == 2);
    /* The complement's exits are the cycles the operand ran out of futures, so
       it has some; its failures are an empty vector, as for any ring shape, so
       `dead` is untouched. */
    const sva_automaton *obligation = sva_monitor_obligation(monitor);
    CHECK(sva_automaton_last(obligation)[0] != 0);
    CHECK(sva_automaton_dead(obligation)[0] == 0);
    sva_monitor_destroy(monitor);

    /* `strong(req ##1 ack) implies weak(ack)`: opposite strengths, so it
       lowers, and the pair carries the right-hand operand's. */
    sva_node strong_left, weak_right, implication;
    BUILD(sva_prop_sequence(ir, SVA_STRONG, body, &strong_left));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &weak_right));
    BUILD(sva_prop_junction(ir, SVA_IMPLIES, strong_left, weak_right, &implication));

    uint32_t implies_index;
    if (add_directive(ir, implication, UINT32_MAX, &implies_index) != 0) return 1;
    sva_monitor *implies_monitor;
    CHECK(sva_lower(&implies_monitor, ir, implies_index, NULL, NULL) == SVA_OK);
    CHECK(sva_monitor_strength(implies_monitor) == SVA_WEAK);
    CHECK(sva_monitor_pending(implies_monitor) == SVA_VERDICT_HOLDS);
    sva_monitor_destroy(implies_monitor);

    /* The same two operands with EQUAL strengths. `implies` reduces to
       `not p or q`, `not` turns the left strength over, and the disjunction
       wants the pair to agree — so equal in means mixed out. */
    sva_node weak_left, same_strength;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &weak_left));
    BUILD(sva_prop_junction(ir, SVA_IMPLIES, weak_left, weak_right, &same_strength));

    uint32_t same_index;
    if (add_directive(ir, same_strength, UINT32_MAX, &same_index) != 0) return 1;
    sva_diagnostic diagnostic;
    memset(&diagnostic, 0, sizeof diagnostic);
    sva_monitor *refused = NULL;
    CHECK(sva_lower(&refused, ir, same_index, NULL, &diagnostic) ==
          SVA_ERROR_MIXED_STRENGTH);
    /* The operator as written, not the disjunction libsva reduced it onto. */
    CHECK(diagnostic.node == same_strength);

    /* Every iff, for the reason on SVA_ERROR_MIXED_STRENGTH: the two directions
       carry opposite strengths by construction, so the conjunction of them
       cannot be satisfied by any operand pair. This one differs, which is what
       an implication wants, and it is still refused. */
    sva_node equivalence;
    BUILD(sva_prop_junction(ir, SVA_IFF, strong_left, weak_right, &equivalence));
    uint32_t iff_index;
    if (add_directive(ir, equivalence, UINT32_MAX, &iff_index) != 0) return 1;
    memset(&diagnostic, 0, sizeof diagnostic);
    CHECK(sva_lower(&refused, ir, iff_index, NULL, &diagnostic) ==
          SVA_ERROR_MIXED_STRENGTH);
    CHECK(diagnostic.node == equivalence);

    /* A complement of something that owns a trigger. The trigger is a union of
       attempts and has thrown away what complementing needs. */
    sva_node consequent, triggered, impossible;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &triggered));
    BUILD(sva_prop_not(ir, triggered, &impossible));
    uint32_t trigger_index;
    if (add_directive(ir, impossible, UINT32_MAX, &trigger_index) != 0) return 1;
    memset(&diagnostic, 0, sizeof diagnostic);
    CHECK(sva_lower(&refused, ir, trigger_index, NULL, &diagnostic) ==
          SVA_ERROR_UNSUPPORTED_COMPLEMENT);
    CHECK(diagnostic.node == impossible);

    sva_ir_destroy(ir);
    return 0;
}

int main(void) {
    /* The header's half of the ABI contract, against the archive's. Everything
       below reads libsva's tables through pointers whose layout only the
       library knows, so a mismatch here means every one of those reads is
       against a shape this file was not compiled for. Check it before anything
       else runs -- this is the check the deleted size constants used to stand
       in for, and unlike them it sees both sides. */
    CHECK(sva_abi_version() == SVA_ABI_VERSION);

    if (implication() != 0) return 1;
    if (walk() != 0) return 1;
    if (shapes() != 0) return 1;
    if (errors() != 0) return 1;
    if (counted() != 0) return 1;
    if (complement() != 0) return 1;
}
