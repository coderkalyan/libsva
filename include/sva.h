/*
 * libsva — concurrent assertions.
 *
 * Temporal properties checked over a trace of discrete cycles. You build an
 * assertion, lower it to a monitor, and either read the monitor's tables to
 * generate code or step it directly.
 *
 *
 * OBJECTS
 *
 * An sva_ir, an sva_monitor and an sva_run are libsva's objects in your own
 * storage: declare one, initialise it in place, tear it down there. Only what
 * they grow internally is allocated, and libsva picks its own allocator, so
 * there is nothing to configure and nothing to pass. Their contents are opaque.
 *
 * An sva_automaton is only reached by pointer, so it has no size here at all.
 * Reading a monitor is direct: an accessor hands back a pointer to a whole array
 * and you index it. One call per table, not per element.
 *
 * Pointer arguments must not be null, except where a comment says otherwise.
 *
 *
 * FAILURE
 *
 * Anything that can fail returns a status and delivers its result through an out
 * parameter, written only on SVA_OK. The only thing a builder can report is an
 * allocation failure:
 *
 *     sva_ir ir;
 *     sva_ir_init(&ir);
 *
 *     sva_node req, ack, consequent, property;
 *     uint32_t directive;
 *     if (sva_bool_atom(&ir, 0, &req) != SVA_OK) goto fail;
 *     if (sva_seq_boolean(&ir, req, &req) != SVA_OK) goto fail;
 *     ...
 *     if (sva_ir_add(&ir, &d, &directive) != SVA_OK) goto fail;
 *
 *     sva_monitor monitor;
 *     if (sva_lower(&monitor, &ir, directive, NULL, NULL) == SVA_OK) { ... }
 *
 * Nothing else is checked. A malformed tree — an operand from the wrong layer, a
 * reversed range, goto over a compound sequence — is a bug in your lowering, not
 * a property of what you are compiling. There is no source location to report,
 * because the location is your own code. libsva asserts and aborts.
 *
 * Legal input libsva cannot lower yet — an unbounded obligation, a nested
 * implication, a limit tripped — is a property of the source, so sva_lower returns
 * a status and fills sva_diagnostic with the node that caused it, for you to map
 * back and report.
 *

 *
 *
 * ATOMS
 *
 * An atom is an opaque uint32_t you assign, dense from zero. libsva never
 * inspects one: evaluating and sampling atoms is your side of the fence. A clock
 * is a handle in its own space, since it delimits cycles rather than being
 * sampled in one.
 *
 * Atoms are assumed TWO-VALUED. sva_bool_not is boolean complement, evaluated as
 * !atom once the guard layer has pushed it down to a leaf. If your own booleans
 * are three-valued, do not use it: lower to negation normal form and give each
 * polarity its own atom. SystemVerilog is the motivating case - LRM 16.6 reads an
 * unknown expression as false, so both `a` and `!a` are false when a is x, and
 * complementing the collapsed bit is simply wrong. De Morgan is sound in
 * three-valued logic, so pushing negation to the leaves costs nothing else.
 * sva_bool_and and sva_bool_or need no such care: they commute with a collapse to
 * false, since an unknown operand cannot turn a known outcome around.
 */

#ifndef SVA_H
#define SVA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* An index into the IR's nodes. Nothing in the tree is optional: every edge is
   present, so there is no way of saying "no node" and no reserved value. */
typedef uint32_t sva_node;

/* An unclocked assertion. */
#define SVA_CLOCK_NONE UINT32_MAX

/* An open upper bound on a delay or repetition range. */
#define SVA_UNBOUNDED  UINT32_MAX

typedef enum sva_status {
    SVA_OK = 0,
    /* An allocation failed. */
    SVA_ERROR_OUT_OF_MEMORY = 1,
    /* An automaton would have exceeded sva_options.state_limit. */
    SVA_ERROR_STATE_LIMIT = 2,
    /* A monitor would have exceeded sva_options.depth_limit. */
    SVA_ERROR_DEPTH_LIMIT = 3,
    /* The obligation's matches have no bounded length, so the number of
       attempts pending at once is unbounded and no fixed pool tracks them.
       Sound tracking needs the obligation determinised, which is not
       implemented. */
    SVA_ERROR_UNBOUNDED_OBLIGATION = 4,
    /* A property operator's operand raises obligations of its own: an
       implication as a consequent, or under a conjunction, a followed-by or a
       disjunctive shift. A monitor has one trigger and there is nowhere to put
       a second. An abort is no longer in that list — it covers the one trigger
       instead of needing a second, subject to the second half of
       SVA_ERROR_UNSUPPORTED_ABORT_SCOPE. */
    SVA_ERROR_UNSUPPORTED_CONSEQUENT = 5,
    /* Goto or non-consecutive repetition over an operand that is not a single
       boolean. */
    SVA_ERROR_UNSUPPORTED_REPEAT_OPERAND = 6,
    /* sva_prop_junction over operands of different strength. The pair is pending
       until both sides resolve, and a pending pair with one weak and one strong
       side has no single end-of-trace disposition. */
    SVA_ERROR_MIXED_STRENGTH = 7,
    /* An abort whose scope libsva cannot carry, in either of two ways.

       It sits below an operator that renumbers its operand's states, so the
       mask naming the abort's subtree cannot be carried through: a conjunction,
       an SVA_SHIFT_EVERY range, or a followed-by. A disjunction and an
       SVA_SHIFT_SOME range are fine.

       Or it sits above an implication whose antecedent can match over more than
       one cycle. Covering a trigger works because the abort's scope is the whole
       trigger, which is the union of every attempt's antecedent — but only while
       no attempt can be caught halfway through matching one. An attempt that is
       owes a verdict of its own, and the union has nothing to count it with. A
       single-cycle antecedent cannot be caught halfway, and is accepted. */
    SVA_ERROR_UNSUPPORTED_ABORT_SCOPE = 8,
} sva_status;

/* ------------------------------------------------------------------------- *
 * Objects
 *
 * An sva_ir, an sva_monitor and an sva_run are libsva's objects in your storage:
 * declare one, initialise it in place, tear it down there. Only what they grow
 * internally is allocated, and libsva picks its own allocator, so there is
 * nothing to configure.
 *
 * The contents are opaque. The sizes here have headroom, and the library fails
 * its own build — naming the size it needs — if a struct outgrows one, so a
 * mismatch is never silent.
 * ------------------------------------------------------------------------- */

#define SVA_IR_SIZE      96
#define SVA_MONITOR_SIZE 288
#define SVA_RUN_SIZE     160

/* `alignas` is a keyword in C++11 and C23, and spelled `_Alignas` in C11. A C++
   consumer is the expected case, so this cannot just use the C form. */
#ifdef __cplusplus
#define SVA_ALIGNAS(n) alignas(n)
#else
#define SVA_ALIGNAS(n) _Alignas(n)
#endif

typedef struct sva_ir {
    SVA_ALIGNAS(8) unsigned char opaque[SVA_IR_SIZE];
} sva_ir;

typedef struct sva_monitor {
    SVA_ALIGNAS(8) unsigned char opaque[SVA_MONITOR_SIZE];
} sva_monitor;

typedef struct sva_run {
    SVA_ALIGNAS(8) unsigned char opaque[SVA_RUN_SIZE];
} sva_run;

/* Reached only by pointer, so it needs no size. */
typedef struct sva_automaton sva_automaton;

/* ------------------------------------------------------------------------- *
 * The IR
 * ------------------------------------------------------------------------- */

void sva_ir_init(sva_ir *ir);
void sva_ir_deinit(sva_ir *ir);

uint32_t sva_ir_node_count(const sva_ir *ir);
/* One past the highest atom handle mentioned, which is the width of the
   per-cycle valuation a backend allocates. */
uint32_t sva_ir_atom_count(const sva_ir *ir);

typedef enum sva_repeat {
    /* a[*lo:hi] — back-to-back matches. */
    SVA_REPEAT_CONSECUTIVE = 0,
    /* b[->lo:hi] — occurrences of a boolean, ending on the last. */
    SVA_REPEAT_GOTO = 1,
    /* b[=lo:hi] — as goto, but the match may extend past the last. */
    SVA_REPEAT_NONCONSECUTIVE = 2,
} sva_repeat;

typedef enum sva_strength {
    /* A pending attempt holds when the trace ends. */
    SVA_WEAK = 0,
    /* A pending attempt fails when the trace ends. */
    SVA_STRONG = 1,
} sva_strength;

typedef enum sva_implication {
    /* s |-> p — the consequent starts on the antecedent's last cycle. */
    SVA_OVERLAPPING = 0,
    /* s |=> p — the consequent starts the cycle after. */
    SVA_NONOVERLAPPING = 1,
} sva_implication;

typedef enum sva_junction {
    /* p and q — both hold, each on its own schedule. */
    SVA_CONJUNCTION = 0,
    /* p or q — either holding is enough. */
    SVA_DISJUNCTION = 1,
} sva_junction;

typedef enum sva_shift {
    /* The range read disjunctively: the property need hold at only one offset
       in it. nexttime[n] p is lo == hi == n; eventually [m:n] p is wider. */
    SVA_SHIFT_SOME = 0,
    /* The range read conjunctively: always [m:n] p, the property at every
       offset. */
    SVA_SHIFT_EVERY = 1,
} sva_shift;

typedef enum sva_abort {
    /* accept_on(b) p — b holding discharges p: it holds. */
    SVA_ABORT_ACCEPT = 0,
    /* reject_on(b) p — b holding fails p. */
    SVA_ABORT_REJECT = 1,
    /* sync_accept_on(b) p — as SVA_ABORT_ACCEPT, b read from sampled values. */
    SVA_ABORT_SYNC_ACCEPT = 2,
    /* sync_reject_on(b) p — as SVA_ABORT_REJECT, b read from sampled values. */
    SVA_ABORT_SYNC_REJECT = 3,
} sva_abort;

typedef enum sva_kind {
    SVA_ASSERT = 0,
    SVA_ASSUME = 1,
    SVA_COVER = 2,
    SVA_RESTRICT = 3,
} sva_kind;

/* Boolean layer: the alphabet. */
sva_status sva_bool_const(sva_ir *ir, bool value, sva_node *out);
sva_status sva_bool_atom(sva_ir *ir, uint32_t atom, sva_node *out);
/* Two-valued complement. See ATOMS above before using this from a front-end
   whose booleans admit an unknown value. */
sva_status sva_bool_not(sva_ir *ir, sva_node operand, sva_node *out);
sva_status sva_bool_and(sva_ir *ir, sva_node lhs, sva_node rhs, sva_node *out);
sva_status sva_bool_or(sva_ir *ir, sva_node lhs, sva_node rhs, sva_node *out);

/* How a boolean node is shaped. */
typedef enum sva_bool_op {
    SVA_BOOL_CONST = 0,
    SVA_BOOL_ATOM = 1,
    SVA_BOOL_NOT = 2,
    SVA_BOOL_AND = 3,
    SVA_BOOL_OR = 4,
} sva_bool_op;

typedef struct sva_bool_view {
    sva_bool_op op;
    /* SVA_BOOL_CONST: 0 or 1. SVA_BOOL_ATOM: the atom handle. SVA_BOOL_NOT: the
       operand. SVA_BOOL_AND and SVA_BOOL_OR: the left operand. */
    uint32_t lhs;
    /* SVA_BOOL_AND and SVA_BOOL_OR: the right operand. Otherwise zero. */
    uint32_t rhs;
} sva_bool_view;

/* Read a boolean node back. This is the other half of the alphabet contract: a
   monitor's roots are boolean nodes, and a backend walks each one to emit its
   evaluation. `node` must be one. */
void sva_bool_read(const sva_ir *ir, sva_node node, sva_bool_view *out);

/* Sequence layer: regular patterns over cycles. Every sequence matches at least
   one cycle — there is no empty match, so a zero repetition count is rejected
   and a source language that has one expands it away first. */
sva_status sva_seq_boolean(sva_ir *ir, sva_node boolean, sva_node *out);

/* left, then right after a delay of lo..hi cycles. The delay is the gap
   *between* the two matches, so 0 overlaps them by a cycle and 1 is plain
   concatenation.
 *
 * A delay measured from the start of the attempt instead — SVA's `##[lo:hi] b`
 * with nothing to its left — is this with a one-cycle `true` to the left, exactly
 * and with no adjustment to the bounds. It is not a second operator. */
sva_status sva_seq_delay(sva_ir *ir, sva_node left, uint32_t lo, uint32_t hi,
                         sva_node right, sva_node *out);

sva_status sva_seq_or(sva_ir *ir, sva_node lhs, sva_node rhs, sva_node *out);
/* Both sides match from the same start cycle, and the match ends at the later
   of the two ends: whichever side finishes first waits for the other. */
sva_status sva_seq_and(sva_ir *ir, sva_node lhs, sva_node rhs, sva_node *out);
/* Both sides match from the same start cycle *and* the same end cycle. */
sva_status sva_seq_intersect(sva_ir *ir, sva_node lhs, sva_node rhs, sva_node *out);
/* `exp` holds in every cycle a match of `sequence` spans. `exp` is a BOOLEAN
   node, not a sequence — it is a condition on each cycle of the match rather
   than something matching alongside it, and it is the only sequence-layer
   operand that is.
 *
 * The LRM writes this as `(exp[*0:$]) intersect sequence`, which cannot be
 * built here because that repetition admits an empty match. Since every
 * sequence matches at least one cycle, `(exp[*1:$]) intersect sequence` is the
 * same language, and is what this builds — for free, by conjoining `exp` into
 * every state of the sequence's automaton. */
sva_status sva_seq_throughout(sva_ir *ir, sva_node exp, sva_node sequence,
                              sva_node *out);

/* A match of `inner` lies somewhere inside a match of `outer`, and the match
   ends where `outer` does.
 *
 * Sugar, not an operator of its own: this builds
 * `(true ##[0:$] inner ##[0:$] true) intersect outer` from tags the IR already
 * has, so nothing downstream has to learn a new one. The LRM pads with
 * `1[*0:$]`, which is an empty match this IR cannot write; a `0..$` delay
 * against a one-cycle `true` is the same padding without one, because delay 0
 * fuses that cycle into the neighbouring match. */
sva_status sva_seq_within(sva_ir *ir, sva_node inner, sva_node outer,
                          sva_node *out);
/* Only the shortest match of `sequence` from each start cycle: as soon as one
   match completes, every longer match of the same attempt is suppressed.
 *
 * The one sequence operator that is not a language operation on matches taken
 * one at a time — whether a match survives depends on whether a shorter one
 * covers the same cycles — so it lowers by determinising its operand rather
 * than by combining automata, and it is the one operator whose state count can
 * be exponential in its operand's. `SVA_ERROR_STATE_LIMIT` names it if it is. */
sva_status sva_seq_first_match(sva_ir *ir, sva_node sequence, sva_node *out);
/* lo must be at least 1. Goto and non-consecutive repetition additionally
   require `sequence` to be an sva_seq_boolean, because they are defined in terms
   of the cycles in which the operand does not hold. */
sva_status sva_seq_repeat(sva_ir *ir, sva_repeat kind, sva_node sequence,
                          uint32_t lo, uint32_t hi, sva_node *out);

/* Property layer. */
sva_status sva_prop_sequence(sva_ir *ir, sva_strength strength, sva_node sequence,
                             sva_node *out);
sva_status sva_prop_implies(sva_ir *ir, sva_implication kind, sva_node antecedent,
                            sva_node consequent, sva_node *out);
/* Both operands must carry the same strength, which is a property of their
   subtrees rather than of their own tags: sva_lower reports
   SVA_ERROR_MIXED_STRENGTH, this call does not. */
sva_status sva_prop_junction(sva_ir *ir, sva_junction kind, sva_node lhs,
                             sva_node rhs, sva_node *out);
/* A property shifted by lo..hi cycles. SVA_UNBOUNDED as hi spells the unbounded
   `always p` / `s_eventually p`: it builds, and sva_lower refuses it with
   SVA_ERROR_UNBOUNDED_OBLIGATION. */
sva_status sva_prop_shift(sva_ir *ir, sva_shift kind, uint32_t lo, uint32_t hi,
                          sva_node property, sva_node *out);
/* s #-# p and s #=# p. The existential dual of sva_prop_implies: an antecedent
   that never matches makes an implication hold vacuously and makes this fail. */
sva_status sva_prop_followed_by(sva_ir *ir, sva_implication kind,
                                sva_node antecedent, sva_node consequent,
                                sva_node *out);
/* An abort scoped to `property`, rather than to the whole monitor the way a
   directive's reset is. The other difference is the verdict: a reset *abandons*
   an attempt, leaving none, while an accept *discharges* it, which is a pass.

   The SVA_ABORT_SYNC_* kinds read `cond` from sampled values and the others from
   live values, exactly as a reset does. Passing a term that matches the kind is
   yours to get right; sva_monitor_abort_sampled reports which was asked for. */
sva_status sva_prop_abort(sva_ir *ir, sva_abort kind, sva_node cond,
                          sva_node property, sva_node *out);

typedef struct sva_directive {
    /* Yours, to resolve against your own tables. */
    uint32_t id;
    sva_kind kind;
    /* SVA_CLOCK_NONE if unclocked. */
    uint32_t clock;
    /* A boolean node. Whenever it holds, every attempt in flight is abandoned
       and fresh ones start from the next cycle. An assertion that never resets
       uses a constant false here — there is no absent edge to give it. */
    sva_node reset;
    sva_node property;
} sva_directive;

/* `out` receives the directive's index, for sva_lower. */
sva_status sva_ir_add(sva_ir *ir, const sva_directive *directive, uint32_t *out);

/* ------------------------------------------------------------------------- *
 * Monitors
 *
 * An assertion compiles to one obligation automaton plus, for an implication, a
 * trigger automaton saying when to start one.
 *
 * The two are tracked differently. The trigger is existential — all that matters
 * is whether *some* antecedent match ends this cycle — so the union of every
 * trigger attempt is sound and one state vector does. The obligation is
 * per-attempt: merging attempts is unsound, because one attempt's threads can
 * die while another's live and the union hides the failure. Each pending attempt
 * gets its own state vector, and sva_monitor_depth says how many there can be.
 * ------------------------------------------------------------------------- */

/* A state mask is sva_automaton_words of these, least significant bit of word
   zero first. */
typedef uint64_t sva_word;

/* A literal is a root, possibly complemented. */
typedef uint32_t sva_literal;
#define SVA_LITERAL_ROOT(literal)    ((literal) >> 1)
#define SVA_LITERAL_NEGATED(literal) ((literal) & 1u)

typedef struct sva_options {
    /* 0 selects the library default. */
    uint32_t state_limit;
    uint32_t depth_limit;
} sva_options;

typedef struct sva_diagnostic {
    /* The node whose lowering failed. */
    sva_node node;
    /* The size that tripped a limit. */
    uint64_t size;
} sva_diagnostic;

/* Lower one directive into `out`. `options` and `diagnostic` may be NULL; `out` is
   written only on success and `diagnostic` only on failure. The IR may be torn
   down afterwards: a monitor does not refer back to it, except that its root
   nodes are only meaningful against it. */
sva_status sva_lower(sva_monitor *out, const sva_ir *ir, uint32_t directive,
                     const sva_options *options, sva_diagnostic *diagnostic);
void sva_monitor_deinit(sva_monitor *monitor);

uint32_t sva_monitor_id(const sva_monitor *monitor);
sva_kind sva_monitor_kind(const sva_monitor *monitor);
uint32_t sva_monitor_clock(const sva_monitor *monitor);

/* Obligation attempts that can be pending at once: the obligation's longest
   match, and so the number of state vectors a backend allocates. An attempt
   started in cycle t has either matched or run out of live states by cycle
   t + depth - 1. */
uint32_t sva_monitor_depth(const sva_monitor *monitor);
sva_strength sva_monitor_strength(const sva_monitor *monitor);
/* Meaningless without a trigger. */
sva_implication sva_monitor_implication(const sva_monitor *monitor);
/* Root index of the reset condition. Always a real root: an assertion that never
   resets has a constant false there, which folds to nothing in a backend. */
uint32_t sva_monitor_reset(const sva_monitor *monitor);

/* Aborts scoped to part of the obligation — accept_on and friends. Zero for a
   property with none, which is the common case.

   An abort is *armed* for one pending attempt exactly when that attempt has a
   live state in sva_monitor_abort_mask. That is what confines it to the cycles
   the property beneath it is actually being evaluated in, so `nexttime[2]
   (reject_on(b) p)` cannot reject during the two cycles it spends getting to p,
   and nothing has to track a window to arrange it.

   Apply them after seeding a new attempt and before taking verdicts — except an
   abort with sva_monitor_abort_covers_trigger, which also acts earlier, on the
   trigger. */
uint32_t sva_monitor_abort_count(const sva_monitor *monitor);
uint32_t sva_monitor_abort_root(const sva_monitor *monitor, uint32_t abort);
/* SVA_ABORT_ACCEPT discharges the attempt, which is a pass a cover counts;
   SVA_ABORT_REJECT fails it. Never an SVA_ABORT_SYNC_* value: whether the
   condition was sampled is sva_monitor_abort_sampled. */
sva_abort sva_monitor_abort_effect(const sva_monitor *monitor, uint32_t abort);
/* Whether firing resolves the whole attempt, or only clears the states in the
   mask and leaves the rest of the attempt running — a reject under a
   disjunction kills its own disjunct and no more. */
bool sva_monitor_abort_resolves_attempt(const sva_monitor *monitor, uint32_t abort);
/* Whether the condition wants the sampled valuation rather than the live one.
   Nothing in this library can check that you honour it.

   False — the plain accept_on/reject_on — means the condition is live, and live
   means EVERY time step, not just the ones carrying a clocking event. A backend
   that steps only on the clock edge therefore has to LATCH it: set a bit
   whenever the condition holds at any time slot, then consume and clear that bit
   at the next edge. Reading the condition at the edge alone is not the live
   valuation, and the failure is silent — it collapses the plain forms onto the
   sync_ ones, since a condition that pulses strictly between two ticks is
   exactly what separates them. */
bool sva_monitor_abort_sampled(const sva_monitor *monitor, uint32_t abort);
/* Whether the abort covers the trigger as well as the obligation: it was written
   above the implication rather than inside its consequent, so it also covers the
   cycles the antecedent is being matched in. Always false without a trigger.

   When it holds in a cycle an antecedent match ends, resolve that match with the
   abort's effect and seed no attempt from it. This is not the same property as
   the abort written inside the consequent, which arms only once the consequent
   starts — a cycle later, for |=>. So an abort above an implication cannot be
   pushed down into it.

   If several of these hold at once, the LAST in the list wins: the list runs
   innermost to outermost, and an outer abort beats an inner verdict landing in
   the same cycle. */
bool sva_monitor_abort_covers_trigger(const sva_monitor *monitor, uint32_t abort);
/* sva_automaton_words(sva_monitor_obligation(monitor)) words. */
const uint64_t *sva_monitor_abort_mask(const sva_monitor *monitor, uint32_t abort);

/* The boolean nodes to evaluate once per cycle, one bit each, in this order.
   This is the whole of what a backend has to understand; everything past it is
   bit arithmetic over the result. Walk one with sva_bool_read — a root is any
   boolean expression, not necessarily an atom. */
uint32_t sva_monitor_root_count(const sva_monitor *monitor);
const sva_node *sva_monitor_roots(const sva_monitor *monitor);

/* Interned guard cubes, shared by both automata so a term the antecedent and the
   consequent share is evaluated once. Guard 0 is the empty cube, which always
   holds and has no literals. */
uint32_t sva_monitor_guard_count(const sva_monitor *monitor);
/* The literals of one guard. Writes the count and returns them, or writes 0 and
   returns NULL for the empty cube. */
const sva_literal *sva_monitor_guard(const sva_monitor *monitor, uint32_t guard,
                                     uint32_t *count);

/* Valid while the monitor is. The trigger is NULL when the property is a bare
   sequence and an obligation starts every cycle. */
const sva_automaton *sva_monitor_obligation(const sva_monitor *monitor);
const sva_automaton *sva_monitor_trigger(const sva_monitor *monitor);

/* A position automaton: one state per letter occurrence, no epsilon transitions,
   and homogeneous — every transition entering a state carries the same guard, so
   a guard belongs to the state. There is no start state; `first` is a mask. */
uint32_t sva_automaton_state_count(const sva_automaton *automaton);
/* Words in one state mask, and the row stride of follow and guard_states. */
uint32_t sva_automaton_words(const sva_automaton *automaton);
/* [state_count] — one guard index per state. */
const uint32_t *sva_automaton_guards(const sva_automaton *automaton);
/* [words] — the states enterable one cycle after `state`. */
const sva_word *sva_automaton_follow(const sva_automaton *automaton, uint32_t state);

/* [words] — states a match may begin in. */
const sva_word *sva_automaton_first(const sva_automaton *automaton);
/* [words] — states that complete a match. */
const sva_word *sva_automaton_last(const sva_automaton *automaton);
/* [words] — the states carrying `guard`. Building the satisfied mask folds over
   guards rather than gathering per state. */
const sva_word *sva_automaton_guard_states(const sva_automaton *automaton,
                                           uint32_t guard);

/* ------------------------------------------------------------------------- *
 * Running
 *
 * A reference implementation of the per-cycle protocol. A compiling backend reads
 * the tables above and emits this itself; this is the contract it has to
 * reproduce, and something to differentially test against.
 *
 * One cycle:
 *   1. Evaluate the roots into a bit each.
 *   2. If the reset root holds, abandon everything in flight and stop.
 *   3. Turn the root bits into a satisfied-state mask per automaton.
 *   4. Advance the trigger, then seed it. Note whether a match now ends.
 *   5. Advance every pending obligation attempt.
 *   6. Seed a new attempt: every cycle with no trigger, in the cycle the trigger
 *      fires for |->, and the cycle after for |=>.
 *   7. Take verdicts. An accepting state discharges an attempt; no live states
 *      fails it.
 *   8. At the end of a finite trace, a pending attempt holds if the property is
 *      weak and fails if it is strong.
 *
 * Seeding before taking verdicts is not incidental: under |-> the obligation
 * starts on the trigger's last cycle, so a one-cycle consequent has to be able
 * to discharge in the cycle it was created.
 * ------------------------------------------------------------------------- */

typedef struct sva_report {
    /* Obligation attempts that ran out of live states. */
    uint32_t failures;
    /* Attempts discharged by a completed match. A cover counts these. */
    uint32_t matches;
    /* The reset condition held, so everything in flight was abandoned. */
    bool reset;
} sva_report;

/* The monitor must outlive the run and must not move. */
sva_status sva_run_init(sva_run *run, const sva_monitor *monitor);
void sva_run_deinit(sva_run *run);

/* Abandon everything in flight, as a reset does. */
void sva_run_clear(sva_run *run);

/* Advance one cycle. `root_bits` holds one bit per sva_monitor_root_count, in
   order, packed into sva_words. */
sva_report sva_run_step(sva_run *run, const sva_word *root_bits);

/* Resolve what is still pending because the trace ended. */
sva_report sva_run_finish(sva_run *run);

#ifdef __cplusplus
}
#endif

#endif /* SVA_H */
