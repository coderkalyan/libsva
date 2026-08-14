// Compiles the public header as C++ and drives a representative slice of the API.
//
// The C smoke test next door proves the header agrees with the exports. It cannot
// prove the header is *includable* from C++, because C and C++ disagree on enough
// to matter — an `alignas` spelled the C way broke a real consumer here once, and
// the incomplete types that replaced it are the same kind of hazard from the
// other direction. Athena, the reason this library exists, is C++23, so a C++
// translation unit is not an exotic case to support; it is the expected one.
//
// Deliberately strict: this is built with -Wall -Wextra -Wconversion -Werror so
// implicit narrowing at the boundary shows up here rather than in a consumer.

#include "sva.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,           \
                         #condition);                                         \
            failures++;                                                       \
        }                                                                     \
    } while (0)

#define BUILD(call) CHECK((call) == SVA_OK)

sva_node atom_cycle(sva_ir* ir, std::uint32_t atom) {
    sva_node boolean = 0;
    sva_node seq = 0;
    BUILD(sva_bool_atom(ir, atom, &boolean));
    BUILD(sva_seq_boolean(ir, boolean, &seq));
    return seq;
}

// The shape athena's first assertion lowers to: `req |=> ack`, no disable.
void implication() {
    sva_ir* ir = sva_ir_create();
    CHECK(ir != nullptr);

    const sva_node req = atom_cycle(ir, 0);
    const sva_node ack = atom_cycle(ir, 1);
    sva_node consequent = 0;
    sva_node property = 0;
    sva_node never = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    BUILD(sva_bool_const(ir, false, &never));

    // Designated initializers over an extern "C" struct, which is how a C++
    // caller would actually write this.
    sva_directive directive{};
    directive.id = 7;
    directive.kind = SVA_ASSERT;
    directive.clock = SVA_CLOCK_NONE;
    directive.reset = never;
    directive.property = property;

    std::uint32_t index = 0;
    BUILD(sva_ir_add(ir, &directive, &index));

    sva_monitor* monitor;
    sva_diagnostic reason{};
    BUILD(sva_lower(&monitor, ir, index, nullptr, &reason));

    CHECK(sva_monitor_id(monitor) == 7);
    CHECK(sva_monitor_kind(monitor) == SVA_ASSERT);
    CHECK(sva_monitor_strength(monitor) == SVA_WEAK);
    CHECK(sva_monitor_implication(monitor) == SVA_NONOVERLAPPING);
    CHECK(sva_monitor_depth(monitor) == 1);
    CHECK(sva_monitor_trigger(monitor) != nullptr);
    // Two atoms plus the constant-false disable condition.
    CHECK(sva_monitor_root_count(monitor) == 3);

    // Copy the tables out the way a compiling backend does, into owning C++
    // containers. This is what exercises the conversions.
    const sva_automaton* obligation = sva_monitor_obligation(monitor);
    const std::uint32_t states = sva_automaton_state_count(obligation);
    const std::uint32_t words = sva_automaton_words(obligation);
    CHECK(states == 1);
    CHECK(words == 1);

    std::vector<std::uint32_t> guard_of(sva_automaton_guards(obligation),
                                        sva_automaton_guards(obligation) + states);
    std::vector<sva_word> follow(static_cast<std::size_t>(states) * words);
    for (std::uint32_t state = 0; state < states; state++) {
        std::memcpy(follow.data() + static_cast<std::size_t>(state) * words,
                    sva_automaton_follow(obligation, state),
                    static_cast<std::size_t>(words) * sizeof(sva_word));
    }
    CHECK(guard_of.size() == 1);
    CHECK(follow.size() == 1);
    CHECK(follow[0] == 0); // one state, and it has no successor

    const std::uint32_t guards = sva_monitor_guard_count(monitor);
    CHECK(guards >= 2);
    std::uint32_t count = 7;
    CHECK(sva_monitor_guard(monitor, 0, &count) == nullptr);
    CHECK(count == 0);

    // Reading a root back is how a backend discovers what to evaluate.
    const sva_node* roots = sva_monitor_roots(monitor);
    bool saw_atom = false;
    bool saw_const = false;
    for (std::uint32_t root = 0; root < sva_monitor_root_count(monitor); root++) {
        sva_bool_view view{};
        sva_bool_read(ir, roots[root], &view);
        if (view.op == SVA_BOOL_ATOM) {
            saw_atom = true;
        }
        if (view.op == SVA_BOOL_CONST) {
            saw_const = true;
            CHECK(view.lhs == 0);
        }
    }
    CHECK(saw_atom);
    CHECK(saw_const);

    // And the reference runtime still runs from C++.
    sva_run* run = sva_run_create(monitor);
    CHECK(run != nullptr);
    sva_word bits = 0;
    const sva_report report = sva_run_step(run, &bits);
    CHECK(report.failures == 0);
    CHECK(report.matches == 0);
    CHECK(!report.reset);
    sva_run_destroy(run);

    sva_monitor_destroy(monitor);
    sva_ir_destroy(ir);
}

// A lowering failure is a status plus a node, not an abort.
void reportable_failure() {
    sva_ir* ir = sva_ir_create();
    CHECK(ir != nullptr);

    // req |=> ##[1:$] ack has no bounded obligation.
    sva_node always = 0;
    sva_node head = 0;
    sva_node body = 0;
    sva_node weak = 0;
    sva_node property = 0;
    sva_node never = 0;
    const sva_node ack = atom_cycle(ir, 1);
    BUILD(sva_bool_const(ir, true, &always));
    BUILD(sva_seq_boolean(ir, always, &head));
    BUILD(sva_seq_delay(ir, head, 1, SVA_UNBOUNDED, ack, &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &weak));
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, weak, &property));
    BUILD(sva_bool_const(ir, false, &never));

    sva_directive directive{};
    directive.kind = SVA_ASSERT;
    directive.clock = SVA_CLOCK_NONE;
    directive.reset = never;
    directive.property = property;
    std::uint32_t index = 0;
    BUILD(sva_ir_add(ir, &directive, &index));

    // `req |=> ##[1:$] ack` lowers now, in SVA_MODE_COUNTED, so what is refused
    // here is the shape that wants both disciplines at once: a trigger strictly
    // below an unbounded operator.
    sva_node consequent = 0, implication = 0, mixed = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &implication));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 0, SVA_UNBOUNDED, implication, &mixed));

    sva_directive mixed_directive{};
    mixed_directive.kind = SVA_ASSERT;
    mixed_directive.clock = SVA_CLOCK_NONE;
    mixed_directive.reset = never;
    mixed_directive.property = mixed;
    std::uint32_t mixed_index = 0;
    BUILD(sva_ir_add(ir, &mixed_directive, &mixed_index));

    // The lowering below fails, so this is never written and owns nothing.
    sva_monitor* monitor;
    sva_diagnostic reason{};
    CHECK(sva_lower(&monitor, ir, mixed_index, nullptr, &reason) == SVA_ERROR_MODE_MIX);
    CHECK(reason.node == mixed);

    // And the shape that used to be refused, now lowering: counted, no depth.
    CHECK(sva_lower(&monitor, ir, index, nullptr, &reason) == SVA_OK);
    CHECK(sva_monitor_mode(monitor) == SVA_MODE_COUNTED);
    CHECK(sva_monitor_depth(monitor) == 0);
    CHECK(sva_automaton_dead(sva_monitor_obligation(monitor)) != nullptr);
    sva_monitor_destroy(monitor);

    sva_ir_destroy(ir);
}

} // namespace

int main() {
    // The new export, reached from C++: the version contract has to be callable
    // from the language the expected consumer is written in.
    CHECK(sva_abi_version() == SVA_ABI_VERSION);
    implication();
    reportable_failure();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok: the header compiles as C++ and the API drives from it\n");
    return 0;
}
