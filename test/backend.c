/*
 * Proves the C API is sufficient to build a *compiling* backend, and that the
 * obvious way to compile the protocol is correct.
 *
 * `Monitor.State` iterates the set bits of a state vector at run time. A code
 * generator cannot: it has no loop over an unknown set. It must instead read
 * every count and every mask out of the monitor once, at compile time, and emit
 * a fold over statically-known states that selects each state's constant follow
 * row by whether that state is live. `emitted_step` below is written that way -
 * one machine integer per state vector, no memory reads, no branches in the hot
 * path - and is differential-tested against `sva_run_step` over random traces.
 *
 * That makes this a test of the contract rather than of the implementation: if
 * the tables an `sva_monitor` exposes were not enough to reconstruct the
 * protocol, or the transposition were unsound, this disagrees.
 */

#include "sva.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);   \
            failures++;                                                       \
        }                                                                     \
    } while (0)

#define BUILD(call) CHECK((call) == SVA_OK)

/* One machine word of states keeps the harness honest without a word loop; the
 * transposition itself is per-word and does not care. */
#define MAX_STATES 64
#define MAX_GUARDS 32
#define MAX_DEPTH  64
#define MAX_LITS   8
#define MAX_ABORTS 4

/* ---- What a code generator bakes in as constants ------------------------ */

typedef struct {
    uint32_t states;
    sva_word first, last, dead;
    sva_word follow[MAX_STATES];
    sva_word guard_states[MAX_GUARDS];
} Tables;

typedef struct {
    uint32_t    count;
    uint32_t    nlits[MAX_GUARDS];
    sva_literal lit[MAX_GUARDS][MAX_LITS];
} Cubes;

/* One abort scoped to part of the obligation. `mask` is the states it guards;
 * `resolves` says whether firing settles the whole attempt or only clears those
 * states. `covers_trigger` says it also covers the antecedent, so a match ending
 * this cycle is resolved by it rather than seeding. `sampled` is metadata a real
 * backend would use to pick which valuation of the root to compute, and is
 * unused here because this test supplies root bits directly. */
typedef struct {
    uint32_t root;
    sva_word mask;
    bool     accept, resolves, covers_trigger, sampled;
} Abort;

typedef struct {
    Cubes    cubes;
    Tables   obligation, trigger;
    bool     has_trigger, overlap;
    /* SVA_VERDICT_HOLDS or SVA_VERDICT_FAILS: what the end of a finite trace
     * makes of an attempt still pending. Read as a verdict rather than as a
     * strength because that is the form that cannot be got backwards, and
     * because sva_prop_not turns it over — a complement of a weak property is
     * strong, and a backend that hard-coded the polarity would be wrong on
     * every negated property without being wrong on any other. */
    sva_verdict at_end;
    bool     counted;
    uint32_t depth, reset_root, roots;
    /* sva_monitor_trigger_window: how many antecedent attempts the trigger tells
     * apart. Zero folds the whole id fold below away, which is what it is for. */
    uint32_t window;
    uint32_t aborts;
    Abort    abort[MAX_ABORTS];
} Compiled;

/* The per-instance mutable state a backend allocates. `attempts` is not
 * redundant with `pending != 0`: an attempt whose state vector is empty is a
 * *failure*, and deriving occupancy from emptiness makes it indistinguishable
 * from an absent attempt, silently dropping every failure. See the test at the
 * bottom of this file. What it holds is how many antecedent attempts the slot
 * stands for — several matches ending in one cycle impose one obligation and owe
 * one verdict each — so zero is free and three is three failures.
 *
 * `counts` is the other discipline and replaces both: in SVA_MODE_COUNTED the
 * obligation is determinised, so an attempt is one state rather than a set and
 * attempts sharing a state share a counter. Exactly one of the two is live.
 *
 * `trigger_ids` is the antecedent's half of the same question: which attempts
 * have a thread in each trigger state, a bit each, so that the matches ending in
 * one cycle can be counted rather than merely noticed. Live only when
 * sva_monitor_trigger_window is non-zero. */
typedef struct {
    sva_word pending[MAX_DEPTH];
    uint64_t attempts[MAX_DEPTH];
    uint64_t counts[MAX_STATES];
    sva_word trigger;
    uint64_t trigger_ids[MAX_STATES];
    uint32_t id_now;
    uint64_t fired;
} Emitted;

static void read_tables(const sva_monitor *m, const sva_automaton *a, Tables *t) {
    t->states = sva_automaton_state_count(a);
    CHECK(t->states <= MAX_STATES);
    CHECK(sva_automaton_words(a) == 1);
    t->first = sva_automaton_first(a)[0];
    t->last  = sva_automaton_last(a)[0];
    t->dead  = sva_automaton_dead(a)[0];
    for (uint32_t s = 0; s < t->states; s++) t->follow[s] = sva_automaton_follow(a, s)[0];
    for (uint32_t g = 0; g < sva_monitor_guard_count(m); g++)
        t->guard_states[g] = sva_automaton_guard_states(a, g)[0];
}

static void compile(const sva_monitor *m, Compiled *c) {
    memset(c, 0, sizeof *c);
    c->counted     = sva_monitor_mode(m) == SVA_MODE_COUNTED;
    c->depth       = sva_monitor_depth(m);
    c->at_end      = sva_monitor_pending(m);
    /* The two accessors are one number in two spellings, and disagreeing is
     * not a thing either of them is allowed to do. */
    CHECK(c->at_end == (sva_monitor_strength(m) == SVA_STRONG ? SVA_VERDICT_FAILS
                                                              : SVA_VERDICT_HOLDS));
    /* And the lattice is an involution, pending included. */
    CHECK(sva_verdict_not(sva_verdict_not(c->at_end)) == c->at_end);
    CHECK(sva_verdict_not(SVA_VERDICT_PENDING) == SVA_VERDICT_PENDING);
    c->overlap     = sva_monitor_implication(m) == SVA_OVERLAPPING;
    c->reset_root  = sva_monitor_reset(m);
    c->roots       = sva_monitor_root_count(m);
    c->has_trigger = sva_monitor_trigger(m) != NULL;
    c->window      = sva_monitor_trigger_window(m);
    CHECK(c->depth <= MAX_DEPTH);
    /* One id per bit of the set, and no window at all without a trigger. */
    CHECK(c->window <= 64);
    CHECK(c->has_trigger || c->window == 0);

    c->cubes.count = sva_monitor_guard_count(m);
    CHECK(c->cubes.count <= MAX_GUARDS);
    for (uint32_t g = 0; g < c->cubes.count; g++) {
        uint32_t           n = 0;
        const sva_literal *l = sva_monitor_guard(m, g, &n);
        CHECK(n <= MAX_LITS);
        c->cubes.nlits[g] = n;
        for (uint32_t i = 0; i < n; i++) c->cubes.lit[g][i] = l[i];
    }
    read_tables(m, sva_monitor_obligation(m), &c->obligation);
    if (c->has_trigger) read_tables(m, sva_monitor_trigger(m), &c->trigger);

    c->aborts = sva_monitor_abort_count(m);
    CHECK(c->aborts <= MAX_ABORTS);
    for (uint32_t i = 0; i < c->aborts; i++) {
        c->abort[i].root     = sva_monitor_abort_root(m, i);
        c->abort[i].mask     = sva_monitor_abort_mask(m, i)[0];
        c->abort[i].accept   = sva_monitor_abort_effect(m, i) == SVA_ABORT_ACCEPT;
        c->abort[i].resolves = sva_monitor_abort_resolves_attempt(m, i);
        c->abort[i].sampled  = sva_monitor_abort_sampled(m, i);
        c->abort[i].covers_trigger = sva_monitor_abort_covers_trigger(m, i);
    }
}

/* ---- The emitted code -------------------------------------------------- */

/* Sign-extend one bit to a full mask, so a constant can be selected without a
 * branch. A code generator emits this as `sext i1 to iN`. */
static sva_word spread(uint64_t bit) {
    return (sva_word)(-(int64_t)(bit & 1));
}

/* `satisfy`: fold over guards, not states. Each cube is a constant literal
 * list, so `holds` is a chain of shifts and ANDs over the root word and the
 * contribution is a constant mask. Guard 1 is the reserved unsatisfiable cube;
 * no surviving state carries it, so its mask is zero and it folds away. */
static sva_word satisfy(const Cubes *cubes, const Tables *t, sva_word roots) {
    sva_word sat = 0;
    for (uint32_t g = 0; g < cubes->count; g++) { /* unrolled at compile time */
        uint64_t holds = 1;
        for (uint32_t i = 0; i < cubes->nlits[g]; i++) {
            const uint32_t root = SVA_LITERAL_ROOT(cubes->lit[g][i]);
            const uint32_t neg  = SVA_LITERAL_NEGATED(cubes->lit[g][i]);
            holds &= ((uint64_t)(roots >> root) & 1u) ^ neg;
        }
        sat |= spread(holds) & t->guard_states[g];
    }
    return sat;
}

/* `advance`: the transposition. Instead of iterating the live states, fold over
 * every state and select its constant follow row by its own liveness bit. */
static sva_word advance(const Tables *t, sva_word current, sva_word satisfied) {
    sva_word next = 0;
    for (uint32_t s = 0; s < t->states; s++) /* unrolled at compile time */
        next |= spread(current >> s) & t->follow[s];
    return next & satisfied;
}

/* `chosen`: isolate the lowest set bit, which is how a counted backend picks the
 * ONE transition a count moves along.
 *
 * Several can be enabled at once even though a counted obligation is
 * determinised, because a determinised state is a (subset, cell) pair and two
 * cells may share a valuation. Such siblings are bisimilar - same subset, so the
 * same exits and the same successors - which is why taking the lowest is a
 * choice and not an approximation. Handing the count to every enabled sibling
 * instead manufactures attempts out of nothing, and does so only on obligations
 * whose cells happen to overlap.
 *
 * One AND and one NEG, no branch. A code generator has no excuse. */
static sva_word chosen(sva_word enabled) {
    return enabled & (~enabled + 1);
}

/* How many attempts an id set stands for. One instruction on every target that
 * matters, and the number a verdict is multiplied by: the trigger counts
 * antecedent matches by identity, so the size of the set IS the count. */
static uint64_t popcount64(uint64_t ids) {
    return (uint64_t)__builtin_popcountll(ids);
}

/* The counted discipline: one uint64_t per obligation state, the obligation
 * determinised, and a failure read out of `dead` rather than out of a state
 * vector running empty.
 *
 * Written the same way as the ring path - a fold over statically-known states
 * selecting constants by mask, no branch in the hot path - because the point of
 * this file is that the tables are enough to COMPILE the protocol, and the
 * counted step is the half that is easy to get subtly wrong.
 *
 * `report` comes in carrying whatever an abort covering the trigger already
 * decided, exactly as the ring path's does. */
static sva_report step_counted(const Compiled *c, Emitted *st, sva_word roots,
                               sva_word sat, uint64_t fires, sva_report report) {
    /* Aborts against the counts as they stand, BEFORE the advance. The ring
     * applies them after, which it can afford because an attempt-resolving abort
     * discards the post-advance vector anyway. Here it cannot: two states can
     * share a successor - everything that has just died shares the trap - and
     * once two counts have merged there is no telling which of them was armed.
     *
     * Armed is a question about the state, not about the attempt's history: the
     * reachable set is `follow[s]`, so one answer serves every attempt counted
     * there and retiring the whole count is exact. Outward, so the scan runs
     * backwards and an outer abort beats an inner one in the same cycle. */
    for (uint32_t s = 0; s < c->obligation.states; s++) { /* unrolled */
        if (st->counts[s] == 0) continue;
        for (uint32_t i = c->aborts; i-- > 0;) {
            if (!((roots >> c->abort[i].root) & 1u)) continue;
            if ((c->obligation.follow[s] & c->abort[i].mask) == 0) continue;
            /* A branch-scoped abort would have to clear states out of a vector,
             * and a counted attempt has none; lowering refuses that pairing. */
            CHECK(c->abort[i].resolves);
            if (c->abort[i].accept) report.matches += st->counts[s];
            else report.failures += st->counts[s];
            st->counts[s] = 0;
            break;
        }
    }

    /* The advance, transposed the way `advance` is: for every source state, the
     * one enabled successor, then a mask-select of that source's count into each
     * destination. `chosen` is what keeps it one successor. */
    uint64_t next[MAX_STATES] = { 0 };
    for (uint32_t s = 0; s < c->obligation.states; s++) { /* unrolled */
        const sva_word one = chosen(c->obligation.follow[s] & sat);
        /* Total, so a live count always has somewhere to go. */
        CHECK(st->counts[s] == 0 || one != 0);
        for (uint32_t t = 0; t < c->obligation.states; t++)
            next[t] += st->counts[s] & (uint64_t)spread(one >> t);
    }

    const uint64_t born = !c->has_trigger ? 1 : (c->overlap ? fires : st->fired);
    if (born != 0) {
        const sva_word one = chosen(c->obligation.first & sat);
        CHECK(one != 0);
        /* A fresh attempt is armed by `first` rather than by a follow row, and its
         * fate is settled before its count is added: afterwards it is
         * indistinguishable from whatever else landed in that state. All `born` of
         * them share one state and one arming, and the count is what says how many
         * verdicts that is worth. */
        uint64_t seeded = born;
        for (uint32_t i = c->aborts; i-- > 0;) {
            if (!((roots >> c->abort[i].root) & 1u)) continue;
            if ((c->obligation.first & c->abort[i].mask) == 0) continue;
            CHECK(c->abort[i].resolves);
            if (c->abort[i].accept) report.matches += born;
            else report.failures += born;
            seeded = 0;
            break;
        }
        for (uint32_t t = 0; t < c->obligation.states; t++)
            next[t] += seeded & (uint64_t)spread(one >> t);
    }
    st->fired = fires;

    /* Verdicts. An exit and a dead state cannot be the same state - dead stands
     * for the empty subset and an exit for a subset holding one of the operand's
     * exits - so testing the exit first mirrors the ring rather than imposing a
     * precedence. A count arriving is that many attempts resolving at once, which
     * is what the counters are for. */
    for (uint32_t t = 0; t < c->obligation.states; t++) { /* unrolled */
        const sva_word bit      = (sva_word)1 << t;
        const uint64_t matched  = (uint64_t)spread((c->obligation.last & bit) >> t);
        const uint64_t died     = (uint64_t)spread((c->obligation.dead & bit) >> t);
        report.matches += next[t] & matched;
        report.failures += next[t] & died;
        next[t] &= ~(matched | died);
    }
    memcpy(st->counts, next, sizeof next);
    return report;
}

static sva_report emitted_step(const Compiled *c, Emitted *st, sva_word roots) {
    sva_report report = { 0, 0, false };

    if ((roots >> c->reset_root) & 1u) {
        memset(st->pending, 0, sizeof st->pending);
        memset(st->attempts, 0, sizeof st->attempts);
        memset(st->counts, 0, sizeof st->counts);
        memset(st->trigger_ids, 0, sizeof st->trigger_ids);
        st->trigger  = 0;
        st->fired    = 0;
        report.reset = true;
        return report;
    }

    const sva_word sat = satisfy(&c->cubes, &c->obligation, roots);

    /* How many antecedent matches end this cycle, and how many attempts are
     * still matching one. A count, not a flag: an obligation is owed per match,
     * so a cycle three matches end in owes three verdicts.
     *
     * With no window the two are the same 0-or-1 and the ids fold away entirely,
     * which is the common case: an antecedent whose matches all span the same
     * number of cycles cannot end two of them together. */
    uint64_t fires = 0, in_flight = 0;
    if (c->has_trigger) {
        const sva_word holds = satisfy(&c->cubes, &c->trigger, roots);

        /* The ids ride the transitions the vector rides, so they move first, off
         * the same pre-advance set. Union, not addition: two threads of ONE
         * attempt reconverging are one attempt, and a match is what an attempt
         * owns. Summing here would report two matches for `a ##[1:2] b ##[1:2] c`
         * ending both ways in one cycle. */
        uint64_t next_ids[MAX_STATES] = { 0 };
        if (c->window != 0) {
            for (uint32_t s = 0; s < c->trigger.states; s++) { /* unrolled */
                const uint64_t ids = st->trigger_ids[s] & spread(st->trigger >> s);
                const sva_word enabled = c->trigger.follow[s] & holds;
                for (uint32_t t = 0; t < c->trigger.states; t++)
                    next_ids[t] |= ids & spread(enabled >> t);
            }
        }

        sva_word next = advance(&c->trigger, st->trigger, holds);
        next |= c->trigger.first & holds; /* an attempt starts every cycle */
        st->trigger = next;

        if (c->window == 0) {
            fires = (next & c->trigger.last) != 0;
            in_flight = fires;
        } else {
            /* This cycle's attempt takes the rotating id. Nothing retires it:
             * the window is the trigger's longest match and every trigger state
             * can reach an exit, so a thread of the attempt that held it that
             * many cycles ago would be a match longer than the longest. */
            const uint64_t id = (uint64_t)1 << st->id_now;
            st->id_now = st->id_now + 1 == c->window ? 0 : st->id_now + 1;
            const sva_word entering = c->trigger.first & holds;
            uint64_t ending = 0, alive = 0;
            for (uint32_t s = 0; s < c->trigger.states; s++) { /* unrolled */
                CHECK((next_ids[s] & id) == 0); /* the last holder is gone */
                next_ids[s] |= id & spread(entering >> s);
                st->trigger_ids[s] = next_ids[s];
                ending |= next_ids[s] & spread(c->trigger.last >> s);
                alive |= next_ids[s];
            }
            fires     = (uint64_t)popcount64(ending);
            in_flight = (uint64_t)popcount64(alive);
        }
    }

    /* An abort above the implication covers the antecedent too, so every attempt
     * still matching one is resolved by it and seeds nothing - the ones ending
     * now and the ones caught halfway alike, one verdict each. This is the cycle
     * the same abort written inside the consequent cannot see at all: there it
     * arms only once the consequent starts, which for `|=>` is the cycle after.
     *
     * Last wins. The list runs innermost to outermost, and an outer abort beats
     * an inner verdict landing in the same cycle. */
    if (in_flight != 0) {
        bool resolved = false, accept = false;
        for (uint32_t i = 0; i < c->aborts; i++) {
            if (!c->abort[i].covers_trigger) continue;
            if (!((roots >> c->abort[i].root) & 1u)) continue;
            resolved = true;
            accept   = c->abort[i].accept;
        }
        if (resolved) {
            if (accept) report.matches += in_flight;
            else report.failures += in_flight;
            fires = 0;
            /* Their threads go with them: an antecedent completing later would
             * impose an obligation on an attempt already resolved. */
            st->trigger = 0;
            memset(st->trigger_ids, 0, sizeof st->trigger_ids);
        }
    }

    /* The two disciplines part company here. Everything above is shared - the
     * reset, the satisfied mask, the trigger, and the verdict an abort covering
     * the trigger produces - and everything below is only about how the pending
     * obligation attempts are held. */
    if (c->counted) return step_counted(c, st, roots, sat, fires, report);

    /* Which aborts each attempt is inside for this cycle, guards aside: the
     * states it could reach, not the ones it ends up live in. An abort covers
     * the cycle its subtree is entered in and the cycle its last thread dies
     * in, and the live set has the subtree empty in both. */
    uint64_t armed[MAX_ABORTS] = { 0 };
    for (uint32_t i = 0; i < c->aborts; i++) {
        for (uint32_t k = 0; k < c->depth; k++) {
            if (st->attempts[k] == 0) continue;
            const sva_word reach = advance(&c->obligation, st->pending[k], ~(sva_word)0);
            if ((reach & c->abort[i].mask) != 0) armed[i] |= (uint64_t)1 << k;
        }
    }

    for (uint32_t k = 0; k < c->depth; k++) { /* unrolled at compile time */
        if (st->attempts[k] == 0) continue;
        st->pending[k] = advance(&c->obligation, st->pending[k], sat);
    }

    /* How many attempts the seeding stands for: one per cycle without a trigger,
     * and otherwise one per antecedent match that ended in the cycle it starts
     * from. Several of them share one slot, because they impose the same
     * obligation from the same cycle - one state vector, one advance, one verdict
     * cycle - and the count is what makes that verdict worth three. */
    const uint64_t born = !c->has_trigger ? 1 : (c->overlap ? fires : st->fired);
    if (born != 0) {
        /* A slot is always free: `depth` is the obligation's longest match, so
         * at most depth - 1 attempts survived into this cycle. */
        uint32_t slot = 0;
        while (slot < c->depth && st->attempts[slot] != 0) slot++;
        CHECK(slot < c->depth);
        st->pending[slot]  = c->obligation.first & sat;
        st->attempts[slot] = born;
        for (uint32_t i = 0; i < c->aborts; i++) {
            if ((c->obligation.first & c->abort[i].mask) != 0)
                armed[i] |= (uint64_t)1 << slot;
        }
    }
    st->fired = fires;

    /* Aborts resolve an attempt early: an accept discharges it, which is a pass,
     * and a reject fails it. After seeding, so an abort firing in the cycle an
     * attempt starts applies to that attempt, and before verdicts, because
     * resolving early is the point.
     *
     * Outward, so the scan runs backwards: the list is innermost-first, and when
     * two aborts resolve one attempt in one cycle the OUTER one wins.
     * `accept_on(a) (reject_on(b) p)` with both holding is TRUE, not FALSE. */
    for (uint32_t i = c->aborts; i-- > 0;) {
        if (!((roots >> c->abort[i].root) & 1u)) continue;
        for (uint32_t k = 0; k < c->depth; k++) {
            if (st->attempts[k] == 0) continue;
            if (!((armed[i] >> k) & 1u)) continue;
            if (c->abort[i].resolves) {
                /* One verdict per attempt the slot stands for: they share the
                 * state vector the abort is armed by, so it fires for all. */
                if (c->abort[i].accept) report.matches += st->attempts[k];
                else report.failures += st->attempts[k];
                st->pending[k]  = 0;
                st->attempts[k] = 0;
            } else {
                st->pending[k] &= ~c->abort[i].mask;
            }
        }
    }

    /* Seeding before taking verdicts is not incidental: under `|->` a one-cycle
     * consequent has to be able to discharge in the cycle it was created. */
    for (uint32_t k = 0; k < c->depth; k++) { /* unrolled at compile time */
        if (st->attempts[k] == 0) continue;
        if ((st->pending[k] & c->obligation.last) != 0) {
            report.matches += st->attempts[k];
            st->pending[k]  = 0;
            st->attempts[k] = 0;
        } else if (st->pending[k] == 0) {
            report.failures += st->attempts[k];
            st->attempts[k] = 0;
        }
    }
    return report;
}

static sva_report emitted_finish(const Compiled *c, Emitted *st) {
    sva_report report = { 0, 0, false };
    if (c->at_end == SVA_VERDICT_FAILS) {
        /* Every attempt still held anywhere is still pending, however many share
         * a counter or a state vector - which is the whole reason neither is a
         * bit. */
        if (c->counted) {
            for (uint32_t s = 0; s < c->obligation.states; s++)
                report.failures += st->counts[s];
        } else {
            for (uint32_t k = 0; k < c->depth; k++) report.failures += st->attempts[k];
        }
    }
    memset(st->pending, 0, sizeof st->pending);
    memset(st->attempts, 0, sizeof st->attempts);
    memset(st->counts, 0, sizeof st->counts);
    memset(st->trigger_ids, 0, sizeof st->trigger_ids);
    st->trigger  = 0;
    st->fired    = 0;
    return report;
}

/* ---- Shapes ------------------------------------------------------------ */

static uint64_t rng_state = 0x243f6a8885a308d3u;
static uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static sva_node atom_cycle(sva_ir *ir, uint32_t atom) {
    sva_node boolean = 0, seq = 0;
    BUILD(sva_bool_atom(ir, atom, &boolean));
    BUILD(sva_seq_boolean(ir, boolean, &seq));
    return seq;
}

typedef sva_node (*Build)(sva_ir *ir);

static sva_node build_delay_range(sva_ir *ir) { /* req |=> ack ##[1:2] done */
    sva_node body = 0, consequent = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0), ack = atom_cycle(ir, 1), done = atom_cycle(ir, 2);
    BUILD(sva_seq_delay(ir, ack, 1, 2, done, &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    return property;
}

static sva_node build_overlap(sva_ir *ir) { /* a ##1 b |-> c */
    sva_node antecedent = 0, consequent = 0, property = 0;
    const sva_node a = atom_cycle(ir, 0), b = atom_cycle(ir, 1), c = atom_cycle(ir, 2);
    BUILD(sva_seq_delay(ir, a, 1, 1, b, &antecedent));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, c, &consequent));
    BUILD(sva_prop_implies(ir, SVA_OVERLAPPING, antecedent, consequent, &property));
    return property;
}

static sva_node build_bare_strong(sva_ir *ir) { /* strong(a ##[1:3] b) */
    sva_node seq = 0, property = 0;
    const sva_node a = atom_cycle(ir, 0), b = atom_cycle(ir, 1);
    BUILD(sva_seq_delay(ir, a, 1, 3, b, &seq));
    BUILD(sva_prop_sequence(ir, SVA_STRONG, seq, &property));
    return property;
}

static sva_node build_repeat(sva_ir *ir) { /* req |=> ack[*1:3] ##1 done */
    sva_node repeat = 0, body = 0, consequent = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0), ack = atom_cycle(ir, 1), done = atom_cycle(ir, 2);
    BUILD(sva_seq_repeat(ir, SVA_REPEAT_CONSECUTIVE, ack, 1, 3, &repeat));
    BUILD(sva_seq_delay(ir, repeat, 1, 1, done, &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    return property;
}

/* `(a ##[1:3] b) |-> c`: the shape the whole trigger count exists for, and the
 * one a flag gets wrong. Three attempts can have their antecedent match END in
 * the same cycle — started three, two and one cycles earlier — and each owes a
 * verdict of its own there. A backend carrying one bit reports one failure where
 * three are owed, so this is the case that separates a compiled step that
 * maintains the id sets from one that does not.
 *
 * `|->` rather than `|=>` so the obligation resolves in the very cycle the
 * matches end, which puts all three verdicts in one report.
 *
 * The goto repetition that used to stand here — `ack[->2] |=> done` — is now
 * refused: an unbounded antecedent has unboundedly many attempts matching at
 * once, and no fixed set of ids counts them. See
 * SVA_ERROR_UNBOUNDED_ANTECEDENT, which smoke.c covers. */
static sva_node build_multi_match(sva_ir *ir) {
    sva_node antecedent = 0, consequent = 0, property = 0;
    const sva_node a = atom_cycle(ir, 0), b = atom_cycle(ir, 1), c = atom_cycle(ir, 2);
    BUILD(sva_seq_delay(ir, a, 1, 3, b, &antecedent));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, c, &consequent));
    BUILD(sva_prop_implies(ir, SVA_OVERLAPPING, antecedent, consequent, &property));
    return property;
}

static sva_node build_cubes(sva_ir *ir) { /* (a && !b) ##1 (c or d) */
    sva_node b = 0, negated = 0, conjunction = 0, head = 0, alternative = 0, seq = 0;
    sva_node a = 0, property = 0;
    BUILD(sva_bool_atom(ir, 0, &a));
    BUILD(sva_bool_atom(ir, 1, &b));
    BUILD(sva_bool_not(ir, b, &negated));
    BUILD(sva_bool_and(ir, a, negated, &conjunction));
    BUILD(sva_seq_boolean(ir, conjunction, &head));
    BUILD(sva_seq_or(ir, atom_cycle(ir, 2), atom_cycle(ir, 3), &alternative));
    BUILD(sva_seq_delay(ir, head, 1, 1, alternative, &seq));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, seq, &property));
    return property;
}

static sva_node build_intersect(sva_ir *ir) { /* (a ##[1:3] b) intersect (c ##2 d) */
    sva_node left = 0, right = 0, both = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 1, 3, atom_cycle(ir, 1), &left));
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 2, 2, atom_cycle(ir, 3), &right));
    BUILD(sva_seq_intersect(ir, left, right, &both));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, both, &property));
    return property;
}

/* `req |=> (en throughout (ack ##1 done))`. The boolean rides on every cycle of
 * the obligation's match, so it lands in each state's guard rather than in a
 * machine of its own — a backend sees one more root and no more states. */
static sva_node build_throughout(sva_ir *ir) {
    sva_node en = 0, body = 0, guarded = 0, consequent = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_bool_atom(ir, 1, &en));
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 1, 1, atom_cycle(ir, 3), &body));
    BUILD(sva_seq_throughout(ir, en, body, &guarded));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, guarded, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    return property;
}

/* `req |=> ((ack ##1 done) and rdy)`. The two sides end in different cycles, so
 * the obligation stays live after `rdy` has matched and is discharged only when
 * the longer side finishes — the case a plain product cannot express. */
static sva_node build_and(sva_ir *ir) {
    sva_node left = 0, both = 0, consequent = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0), rdy = atom_cycle(ir, 3);
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 1), 1, 1, atom_cycle(ir, 2), &left));
    BUILD(sva_seq_and(ir, left, rdy, &both));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, both, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    return property;
}

/* `(a ##1 b) within c[*1:3]`, which is sugar: sva_seq_within expands into
 * delays and an intersection, so this shape checks that what the expansion
 * lowers to still reaches a backend through the same tables. */
static sva_node build_within(sva_ir *ir) {
    sva_node inner = 0, outer = 0, nested = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 1, 1, atom_cycle(ir, 1), &inner));
    BUILD(sva_seq_repeat(ir, SVA_REPEAT_CONSECUTIVE, atom_cycle(ir, 2), 1, 3, &outer));
    BUILD(sva_seq_within(ir, inner, outer, &nested));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, nested, &property));
    return property;
}

/* `req |=> first_match(ack ##[1:3] done)`. The obligation is the operand
 * determinised and cut off at its first match, so its states are subsets and
 * cubes rather than letter occurrences — the one shape whose automaton is not
 * built by combining its operands' — and a backend has to reproduce it from
 * the same three tables as any other. Its exits have empty follow rows, which
 * is what suppression looks like from out here. */
static sva_node build_first_match(sva_ir *ir) {
    sva_node body = 0, earliest = 0, consequent = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 1), 1, 3, atom_cycle(ir, 2), &body));
    BUILD(sva_seq_first_match(ir, body, &earliest));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, earliest, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    return property;
}

/* The property layer above a sequence. None of it needs a new table or a new
 * opcode: a shift, a conjunction and a followed-by all land in the obligation
 * automaton, and only an abort adds anything a backend reads — a root, a mask
 * and two flags. */

static sva_node build_prop_and(sva_ir *ir) { /* req |=> (ack and (##1 done)) */
    sva_node left = 0, delayed = 0, right = 0, pair = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &left));
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 3), 1, 1, atom_cycle(ir, 2), &delayed));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, delayed, &right));
    BUILD(sva_prop_junction(ir, SVA_CONJUNCTION, left, right, &pair));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, pair, &property));
    return property;
}

static sva_node build_prop_or(sva_ir *ir) { /* req |=> (ack or (##1 done)) */
    sva_node left = 0, delayed = 0, right = 0, pair = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &left));
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 3), 1, 1, atom_cycle(ir, 2), &delayed));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, delayed, &right));
    BUILD(sva_prop_junction(ir, SVA_DISJUNCTION, left, right, &pair));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, pair, &property));
    return property;
}

static sva_node build_nexttime(sva_ir *ir) { /* req |=> nexttime[2] ack */
    sva_node consequent = 0, shifted = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 2, 2, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, shifted, &property));
    return property;
}

static sva_node build_eventually(sva_ir *ir) { /* req |=> eventually [1:3] ack */
    sva_node consequent = 0, shifted = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 1, 3, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, shifted, &property));
    return property;
}

/* `req |=> always [1:3] ack`. The conjunctive reading of the same range: the
 * obligation is a product of the operand shifted to each offset, which for a
 * one-cycle operand prunes to a chain. */
static sva_node build_always(sva_ir *ir) {
    sva_node consequent = 0, shifted = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 1, 3, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, shifted, &property));
    return property;
}

/* `(a ##[0:1] b) #-# c`, with no trigger at all: a followed-by never becomes
 * one, because it is a concatenation rather than an implication. An antecedent
 * that never matches fails it, where it would make an implication hold. */
static sva_node build_followed_by(sva_ir *ir) {
    sva_node antecedent = 0, consequent = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 0, 1, atom_cycle(ir, 1), &antecedent));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 2), &consequent));
    BUILD(sva_prop_followed_by(ir, SVA_OVERLAPPING, antecedent, consequent, &property));
    return property;
}

/* `req |=> accept_on(kill) (##1 ack)`. The abort is scoped to the obligation, so
 * a backend reads one mask and applies it per attempt; the antecedent keeps
 * matching in cycles `kill` holds, which is what a monitor-wide reset would not
 * do. Atom 4 is the condition. */
static sva_node build_accept_on(sva_ir *ir) {
    sva_node body = 0, consequent = 0, guarded = 0, kill = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_ACCEPT, kill, consequent, &guarded));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, guarded, &property));
    return property;
}

/* `sync_reject_on(kill) nexttime[2] ack`, strong. The abort sits under the
 * shift, so its mask is a proper subset of the obligation's states and the two
 * filler cycles are cycles it cannot fire in. */
static sva_node build_reject_on(sva_ir *ir) {
    sva_node consequent = 0, guarded = 0, shifted = 0, kill = 0;
    BUILD(sva_prop_sequence(ir, SVA_STRONG, atom_cycle(ir, 1), &consequent));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_SYNC_REJECT, kill, consequent, &guarded));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 2, 2, guarded, &shifted));
    return shifted;
}

/* `reject_on(kill) (req |=> (##1 ack))`. The abort is ABOVE the implication, so
 * it covers the antecedent as well: when `kill` holds in the cycle `req`
 * matches, the attempt is resolved right there and no obligation is ever seeded
 * from it. That is a cycle earlier than the same abort inside the consequent,
 * which is why a backend cannot infer this from the mask —
 * `sva_monitor_abort_covers_trigger` is a flag it has to read. Atom 4 is the
 * condition. */
static sva_node build_reject_over_implication(sva_ir *ir) {
    sva_node body = 0, consequent = 0, implication = 0, kill = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &implication));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_REJECT, kill, implication, &property));
    return property;
}

/* The same over `|->`, and with an accept, which discharges where the reject
 * fails. Those are the two halves of the trigger-covering path a backend picks
 * between, and overlapping seeds in the very cycle the abort resolves. */
static sva_node build_accept_over_implication(sva_ir *ir) {
    sva_node body = 0, consequent = 0, implication = 0, kill = 0, property = 0;
    const sva_node req = atom_cycle(ir, 0);
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_prop_implies(ir, SVA_OVERLAPPING, req, consequent, &implication));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_ACCEPT, kill, implication, &property));
    return property;
}

/* `accept_on(kill) (reject_on(other) (##1 ack))`, no trigger. Two aborts that
 * resolve the same attempt, so the compiled path has to agree on WHICH one wins
 * when both hold — the outer accept, which is the LAST entry of the list and not
 * the first. Scanning the list forwards reports a failure here where the
 * reference reports a pass. Atom 3 is the reject's condition, atom 4 the
 * accept's. */
static sva_node build_nested_aborts(sva_ir *ir) {
    sva_node body = 0, consequent = 0, inner = 0, outer = 0, kill = 0, other = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 2), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &consequent));
    BUILD(sva_bool_atom(ir, 3, &other));
    BUILD(sva_prop_abort(ir, SVA_ABORT_REJECT, other, consequent, &inner));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_ACCEPT, kill, inner, &outer));
    return outer;
}

/* ---- Counted mode ------------------------------------------------------- *
 *
 * The obligations with no bounded longest match. A backend reads exactly two
 * extra things for these — sva_monitor_mode and sva_automaton_dead — and then
 * emits `step_counted` instead of the ring path. Every shape below reaches it,
 * and the differential check against sva_run_step is the whole proof that the
 * tie-break and the dead mask are enough to compile it.
 */

/* `req |=> s_eventually ack`: the headline. A trigger ABOVE an unbounded
 * operator, which is fine — a trigger is existential and needs no bound — so the
 * counted obligation and the ring-free trigger coexist in one monitor. This is
 * the shape that used to be refused outright. */
static sva_node build_s_eventually(sva_ir *ir) {
    sva_node consequent = 0, shifted = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_STRONG, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 0, SVA_UNBOUNDED, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, atom_cycle(ir, 0), shifted, &property));
    return property;
}

/* `s_eventually (ack ##1 done)`, no trigger. The operand of an existential
 * unbounded shift may span any number of cycles, and this is what pins that:
 * only the universal operators want a single-cycle operand. */
static sva_node build_s_eventually_multi(sva_ir *ir) {
    sva_node body = 0, consequent = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 1), 1, 1, atom_cycle(ir, 2), &body));
    BUILD(sva_prop_sequence(ir, SVA_STRONG, body, &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 0, SVA_UNBOUNDED, consequent, &property));
    return property;
}

/* `req |=> always ack`. The dual of the above in every respect a backend can
 * see: the obligation has no exits at all, because "for ever" has no finite
 * witness, so it never reports a match and reports a failure the cycle `ack`
 * drops. Two overlapping triggers both fail in that cycle, which is exactly the
 * multiplicity the counters exist for and a merged vector could not report. */
static sva_node build_always_forever(sva_ir *ir) {
    sva_node consequent = 0, shifted = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 0, SVA_UNBOUNDED, consequent, &shifted));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, atom_cycle(ir, 0), shifted, &property));
    return property;
}

/* `nexttime[2] always ack`: the same, entered two cycles in, so the filler chain
 * sits in front of a fragment with no exits. */
static sva_node build_always_shifted(sva_ir *ir) {
    sva_node consequent = 0, forever = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 0, SVA_UNBOUNDED, consequent, &forever));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 2, 2, forever, &property));
    return property;
}

/* The four until forms over one-cycle operands, which is where they differ from
 * each other rather than from anything else: `_WITH` requires `a` in the cycle
 * `b` arrives, and the STRONG kinds fail at the end of a trace `b` never arrived
 * in where the weak ones hold. */
static sva_node build_until_kind(sva_ir *ir, sva_until kind) {
    sva_node left = 0, right = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 0), &left));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &right));
    BUILD(sva_prop_until(ir, kind, left, right, &property));
    return property;
}

static sva_node build_until(sva_ir *ir) { return build_until_kind(ir, SVA_UNTIL_WEAK); }
static sva_node build_until_with(sva_ir *ir) {
    return build_until_kind(ir, SVA_UNTIL_WEAK_WITH);
}
static sva_node build_s_until(sva_ir *ir) { return build_until_kind(ir, SVA_UNTIL_STRONG); }
static sva_node build_s_until_with(sva_ir *ir) {
    return build_until_kind(ir, SVA_UNTIL_STRONG_WITH);
}

/* `a until (ack ##1 done)`. The right operand is existential — one attempt of it
 * per candidate cycle, only one of which has to succeed — so it may span several
 * cycles even though the left one may not. */
static sva_node build_until_multi(sva_ir *ir) {
    sva_node left = 0, body = 0, right = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 0), &left));
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 1), 1, 1, atom_cycle(ir, 2), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &right));
    BUILD(sva_prop_until(ir, SVA_UNTIL_WEAK, left, right, &property));
    return property;
}

/* `accept_on(kill) (s_eventually ack)`, strong. An abort over a counted
 * obligation, which is carried through the determinisation as a mask over
 * subsets: a determinised state is armed when its subset meets the original
 * mask. Being armed is then a property of the state, so firing retires every
 * attempt counted there at once - which is exact, and is the case a backend that
 * retired "the" attempt would get wrong. Atom 4 is the condition. */
static sva_node build_counted_abort(sva_ir *ir) {
    sva_node consequent = 0, shifted = 0, property = 0, kill = 0;
    BUILD(sva_prop_sequence(ir, SVA_STRONG, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_SOME, 1, SVA_UNBOUNDED, consequent, &shifted));
    BUILD(sva_bool_atom(ir, 4, &kill));
    BUILD(sva_prop_abort(ir, SVA_ABORT_ACCEPT, kill, shifted, &property));
    return property;
}

/* `not weak(a ##1 b)`. The claim this shape exists to make is that a complement
 * reaches a backend as an ORDINARY automaton: the tables below are read by the
 * same `read_tables`, stepped by the same ring, and resolved by the same
 * emptiness rule as every shape above it. Nothing in the emitted code knows a
 * `not` was involved.
 *
 * What it does change is invisible from here and load-bearing anyway — the
 * exits are the cycles the operand ran out of futures, the vector empties in
 * the cycles the operand matched, and `at_end` came back inverted, which is why
 * `emitted_finish` reads a verdict rather than assuming a polarity. */
static sva_node build_not(sva_ir *ir) {
    sva_node body = 0, inner = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &inner));
    BUILD(sva_prop_not(ir, inner, &property));
    return property;
}

/* `not (always ack)`, which is `s_eventually !ack` and so has no longest match.
 * A complement borrows its operand's bound, so this one stays SVA_MODE_COUNTED
 * and the counted step drives a complement — the determinisation the mode wants
 * run over a fragment that has already been determinised once. */
static sva_node build_not_counted(sva_ir *ir) {
    sva_node consequent = 0, forever = 0, property = 0;
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 1), &consequent));
    BUILD(sva_prop_shift(ir, SVA_SHIFT_EVERY, 0, SVA_UNBOUNDED, consequent, &forever));
    BUILD(sva_prop_not(ir, forever, &property));
    return property;
}

/* `not not weak(a ##1 b)`: the involution, driven rather than argued. It must
 * report exactly what `build_not`'s operand reports, cycle for cycle. */
static sva_node build_not_not(sva_ir *ir) {
    sva_node body = 0, inner = 0, once = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 1, 1, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, body, &inner));
    BUILD(sva_prop_not(ir, inner, &once));
    BUILD(sva_prop_not(ir, once, &property));
    return property;
}

/* `strong(a ##[1:2] b) implies weak(c)`, through SVA_JUNCTION_IMPLIES. The two
 * operands carry OPPOSITE strengths, which is what the reduction to
 * `not p or q` requires: `not` turns the left one over, and the disjunction
 * underneath wants the pair to agree. Equal strengths are refused with
 * SVA_ERROR_MIXED_STRENGTH, which `sva_prop_not_refusals` in smoke.c covers. */
static sva_node build_implies(sva_ir *ir) {
    sva_node body = 0, left = 0, right = 0, property = 0;
    BUILD(sva_seq_delay(ir, atom_cycle(ir, 0), 1, 2, atom_cycle(ir, 1), &body));
    BUILD(sva_prop_sequence(ir, SVA_STRONG, body, &left));
    BUILD(sva_prop_sequence(ir, SVA_WEAK, atom_cycle(ir, 2), &right));
    BUILD(sva_prop_junction(ir, SVA_IMPLIES, left, right, &property));
    return property;
}

typedef struct {
    const char *name;
    Build       build;
    bool        with_reset;
} Shape;

/* Drive both implementations over the same random root valuations. */
static void run_shape(const Shape *shape, uint32_t traces, uint32_t length) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);
    const sva_node property = shape->build(ir);

    sva_node reset = 0;
    if (shape->with_reset) {
        BUILD(sva_bool_atom(ir, 7, &reset));
    } else {
        BUILD(sva_bool_const(ir, false, &reset));
    }

    sva_directive directive;
    memset(&directive, 0, sizeof directive);
    directive.kind     = SVA_ASSERT;
    directive.clock    = SVA_CLOCK_NONE;
    directive.reset    = reset;
    directive.property = property;

    uint32_t index = 0;
    BUILD(sva_ir_add(ir, &directive, &index));

    sva_monitor *monitor;
    if (sva_lower(&monitor, ir, index, NULL, NULL) != SVA_OK) {
        fprintf(stderr, "%s: lowering failed\n", shape->name);
        failures++;
        sva_ir_destroy(ir);
        return;
    }

    Compiled compiled;
    compile(monitor, &compiled);

    sva_run *run = sva_run_create(monitor);
    CHECK(run != NULL);

    /* A constant root must be driven to its own value, which is what a backend
     * emits for it; everything else is free. */
    const uint32_t  roots     = sva_monitor_root_count(monitor);
    const sva_node *root_node = sva_monitor_roots(monitor);
    sva_word        forced_on = 0, forced_off = 0;
    for (uint32_t r = 0; r < roots; r++) {
        sva_bool_view view;
        sva_bool_read(ir, root_node[r], &view);
        if (view.op != SVA_BOOL_CONST) continue;
        if (view.lhs != 0) forced_on |= (sva_word)1 << r;
        else forced_off |= (sva_word)1 << r;
    }
    const sva_word mask = roots >= 64 ? ~(sva_word)0 : (((sva_word)1 << roots) - 1);

    uint32_t steps = 0;
    for (uint32_t t = 0; t < traces && failures == 0; t++) {
        Emitted state;
        memset(&state, 0, sizeof state);
        sva_run_clear(run);

        for (uint32_t cycle = 0; cycle < length; cycle++) {
            sva_word bits = (rng() & mask & ~forced_off) | forced_on;

            const sva_report want = sva_run_step(run, &bits);
            const sva_report got  = emitted_step(&compiled, &state, bits);
            steps++;
            if (want.failures != got.failures || want.matches != got.matches ||
                want.reset != got.reset) {
                fprintf(stderr,
                        "%s: trace %u cycle %u roots 0x%llx: "
                        "reference f=%llu m=%llu r=%d, emitted f=%llu m=%llu r=%d\n",
                        shape->name, t, cycle, (unsigned long long)bits,
                        (unsigned long long)want.failures, (unsigned long long)want.matches,
                        (int)want.reset, (unsigned long long)got.failures,
                        (unsigned long long)got.matches, (int)got.reset);
                failures++;
                break;
            }
        }
        const sva_report want = sva_run_finish(run);
        const sva_report got  = emitted_finish(&compiled, &state);
        if (want.failures != got.failures || want.matches != got.matches) {
            fprintf(stderr, "%s: trace %u finish: reference f=%llu, emitted f=%llu\n",
                    shape->name, t, (unsigned long long)want.failures,
                    (unsigned long long)got.failures);
            failures++;
        }
    }

    printf("  %-26s %2u states, %s %u, %u guards, %u roots, %u steps\n", shape->name,
           sva_automaton_state_count(sva_monitor_obligation(monitor)),
           sva_monitor_mode(monitor) == SVA_MODE_COUNTED ? "counted" : "depth",
           sva_monitor_depth(monitor), sva_monitor_guard_count(monitor), roots, steps);

    sva_run_destroy(run);
    sva_monitor_destroy(monitor);
    sva_ir_destroy(ir);
}

/* Why the per-slot attempt count is not redundant with `pending != 0`.
 *
 * An attempt seeded into a cycle whose first letter does not hold has an empty
 * state vector immediately. That is a *failure*, and it is the single most
 * common one - it is what `a |=> b` reports when `b` is low. A backend that
 * derives occupancy from `pending != 0` cannot tell that attempt from an absent
 * one and silently reports nothing. */
static void empty_attempt_is_a_failure(void) {
    sva_ir *ir = sva_ir_create();
    CHECK(ir != NULL);

    sva_node consequent = 0, property = 0, never = 0;
    const sva_node req = atom_cycle(ir, 0), ack = atom_cycle(ir, 1);
    BUILD(sva_prop_sequence(ir, SVA_WEAK, ack, &consequent));
    BUILD(sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property));
    BUILD(sva_bool_const(ir, false, &never));

    sva_directive directive;
    memset(&directive, 0, sizeof directive);
    directive.kind = SVA_ASSERT;
    directive.clock = SVA_CLOCK_NONE;
    directive.reset = never;
    directive.property = property;
    uint32_t index = 0;
    BUILD(sva_ir_add(ir, &directive, &index));

    sva_monitor *monitor;
    BUILD(sva_lower(&monitor, ir, index, NULL, NULL));
    Compiled compiled;
    compile(monitor, &compiled);

    /* Find req's root so the trace does not assume a bit position. */
    uint32_t        req_root  = ~0u;
    const sva_node *root_node = sva_monitor_roots(monitor);
    for (uint32_t r = 0; r < sva_monitor_root_count(monitor); r++) {
        sva_bool_view view;
        sva_bool_read(ir, root_node[r], &view);
        if (view.op == SVA_BOOL_ATOM && view.lhs == 0) req_root = r;
    }
    CHECK(req_root != ~0u);

    Emitted state;
    memset(&state, 0, sizeof state);

    /* req high, then nothing: the obligation is seeded in the second cycle with
     * no live state and must be reported as a failure that cycle. */
    sva_word bits = (sva_word)1 << req_root;
    CHECK(emitted_step(&compiled, &state, bits).failures == 0);
    bits = 0;
    const sva_report second = emitted_step(&compiled, &state, bits);
    CHECK(second.failures == 1);
    /* And the slot it was reported from now holds nothing, which is exactly why
     * emptiness cannot also mean "free". */
    CHECK(state.pending[0] == 0);
    CHECK(state.attempts[0] == 0);
    /* One antecedent match per cycle here, so the slot only ever stood for one
     * attempt: `a |=> b` is the shape a zero window is for. */
    CHECK(compiled.window == 0);

    sva_monitor_destroy(monitor);
    sva_ir_destroy(ir);
}

int main(void) {
    const Shape shapes[] = {
        { "req |=> ack ##[1:2] done", build_delay_range, false },
        { "the same, with a reset",   build_delay_range, true  },
        { "a ##1 b |-> c",            build_overlap,     false },
        { "strong(a ##[1:3] b)",      build_bare_strong, false },
        { "req |=> ack[*1:3] ##1 dn", build_repeat,      false },
        { "(a ##[1:3] b) |-> c",      build_multi_match, false },
        { "(a && !b) ##1 (c or d)",   build_cubes,       false },
        { "intersect product",        build_intersect,   false },
        { "en throughout (a ##1 b)",  build_throughout,  false },
        { "(a ##1 b) and c",          build_and,         false },
        { "(a ##1 b) within c[*1:3]", build_within,      false },
        { "first_match(a ##[1:3] b)", build_first_match, false },
        { "req |=> ack and ##1 done", build_prop_and,    false },
        { "req |=> ack or ##1 done",  build_prop_or,     false },
        { "req |=> nexttime[2] ack",  build_nexttime,    false },
        { "req |=> eventually[1:3]",  build_eventually,  false },
        { "req |=> always[1:3] ack",  build_always,      false },
        { "(a ##[0:1] b) #-# c",      build_followed_by, false },
        { "accept_on(kill) ##1 ack",  build_accept_on,   false },
        { "sync_reject_on(kill) p",   build_reject_on,   false },
        { "an abort beside a reset",  build_accept_on,   true  },
        { "reject_on(kill)(req|=>p)", build_reject_over_implication, false },
        { "accept_on(kill)(req|->p)", build_accept_over_implication, false },
        { "accept_on(reject_on(p))",  build_nested_aborts, false },

        /* Counted mode: no bounded longest match, so a determinised obligation
         * and a counter per state rather than a ring of vectors. */
        { "req |=> s_eventually ack", build_s_eventually,       false },
        { "s_eventually (a ##1 b)",   build_s_eventually_multi, false },
        { "req |=> always ack",       build_always_forever,     false },
        { "counted, with a reset",    build_always_forever,     true  },
        { "nexttime[2] always ack",   build_always_shifted,     false },
        { "a until b",                build_until,              false },
        { "a until_with b",           build_until_with,         false },
        { "a s_until b",              build_s_until,            false },
        { "a s_until_with b",         build_s_until_with,       false },
        { "a until (b ##1 c)",        build_until_multi,        false },
        { "accept_on(kill) counted",  build_counted_abort,      false },
        { "not (a ##1 b)",            build_not,                false },
        { "the same, with a reset",   build_not,                true  },
        { "not (always ack)",         build_not_counted,        false },
        { "not not (a ##1 b)",        build_not_not,            false },
        { "strong(a##[1:2]b) impl c", build_implies,            false },
    };

    printf("compiled backend vs reference runtime:\n");
    for (uint32_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        run_shape(&shapes[i], 400, 24);

    empty_attempt_is_a_failure();

    if (failures != 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("ok: a transposed, branch-free backend reproduces the protocol\n");
    return 0;
}
