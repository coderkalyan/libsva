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
 * An sva_ir, an sva_monitor and an sva_run are incomplete types: libsva
 * allocates one and hands back a pointer, and you hand that pointer back to be
 * destroyed. libsva picks its own allocator, so there is nothing to configure
 * and nothing to pass, and nothing out here depends on how big any of them is.
 * The ABI section below says what that last part replaced.
 *
 * An sva_automaton is reached only by pointer and is owned by its monitor.
 * Reading a monitor is direct: an accessor hands back a pointer to a whole array
 * and you index it. One call per table, not per element.
 *
 * Pointer arguments must not be null, except where a comment says otherwise —
 * and every destroying call accepts null, so a failure path needs one
 * unconditional cleanup rather than a branch per object it got as far as.
 *
 *
 * FAILURE
 *
 * Anything that can fail returns a status and delivers its result through an out
 * parameter, written only on SVA_OK. The only thing a builder can report is an
 * allocation failure:
 *
 *     sva_ir *ir = sva_ir_create();
 *     if (ir == NULL) goto fail;
 *
 *     sva_node req, ack, consequent, property;
 *     uint32_t directive;
 *     if (sva_bool_atom(ir, 0, &req) != SVA_OK) goto fail;
 *     if (sva_seq_boolean(ir, req, &req) != SVA_OK) goto fail;
 *     ...
 *     if (sva_ir_add(ir, &d, &directive) != SVA_OK) goto fail;
 *
 *     sva_monitor *monitor;
 *     if (sva_lower(&monitor, ir, directive, NULL, NULL) == SVA_OK) { ... }
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
    /* An obligation with no bounded longest match that counter tracking cannot
       represent either. One shape, in two spellings: the operand of an unbounded
       `always` (SVA_SHIFT_EVERY with hi == SVA_UNBOUNDED), and the LEFT operand
       of any sva_prop_until form, must each match in a single cycle. The
       diagnostic names the offending operand.

       Both of those operators are universal over cycles — they say the operand
       holds at every cycle of a window — which is a statement about one attempt
       of the operand per cycle, all overlapping. A counted attempt is a single
       state and cannot carry a set of sub-attempts that grows with the trace.
       When the operand matches in exactly one cycle each sub-attempt resolves
       inside its own cycle, the set never grows, and the shape is accepted.

       Existential operators carry no such restriction: the operand of an
       unbounded s_eventually (SVA_SHIFT_SOME) and the RIGHT operand of an until
       may span any number of cycles. */
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
    /* sva_prop_junction over operands whose strengths do not meet that
       junction's requirement. The pair is pending until both sides resolve, and
       one strength per monitor cannot say what the end of a trace makes of a
       pending pair whose two sides would answer differently.

       SVA_CONJUNCTION and SVA_DISJUNCTION want the two strengths EQUAL.
       SVA_IMPLIES wants them to DIFFER — the same rule seen through the
       reduction, since `p implies q` is `not p or q` and `not` inverts
       strength. SVA_IFF can never satisfy it: the two directions carry q's and
       p's strengths, and each direction already forced those to be opposite, so
       the outer conjunction always sees a mixed pair. Every iff is refused.

       None of this is conservative. Take p and q both weak and `p implies q`.
       On a trace where q has definitively failed and p is still pending, the
       end of the trace makes p hold, so q was required and the implication
       fails; on a trace where both are pending it holds. Same monitor, two live
       state configurations, opposite dispositions, one bit to carry them in.

       Lifting it needs a per-state end-of-trace disposition rather than one
       sva_monitor_strength per monitor. That would delete this status and let
       iff lower, and it would change what sva_monitor_strength means. */
    SVA_ERROR_MIXED_STRENGTH = 7,
    /* An abort whose scope libsva cannot carry, in any of three ways.

       It sits below an operator that renumbers its operand's states, so the
       mask naming the abort's subtree cannot be carried through: a conjunction,
       an SVA_SHIFT_EVERY range, or a followed-by. A disjunction and an
       SVA_SHIFT_SOME range are fine.

       Or it does not resolve the whole attempt — sva_monitor_abort_resolves_-
       attempt would be false — over a monitor that came out SVA_MODE_COUNTED.
       Killing part of an attempt is a mask subtracted from its state vector, and
       a counted attempt has no vector: its single state stands for a whole
       subset of the operand's states, and no state stands for that subset minus
       the mask. An abort that resolves the whole attempt is carried through
       determinisation and is fine.

       Or it sits under an sva_prop_not, whatever its scope. Not for want of a
       mask — that moves through the determinisation perfectly well — but
       because ARMING does not survive. An abort is armed from the unfiltered
       one-step image of where an attempt is, deliberately, so that it covers
       the cycle its subtree's last thread dies in; that cycle is the whole
       point of an accept, which discharges an attempt the property was about to
       fail. A determinised successor is already filtered by the valuation, and
       a complement then deletes the one state that image would have been read
       off, so the abort would read as unarmed in exactly the cycle it should
       fire. Lowering it would turn every such accept silently into a failure. */
    SVA_ERROR_UNSUPPORTED_ABORT_SCOPE = 8,
    /* A trigger strictly below an unbounded operator: always (a |=> b),
       s_eventually (a |=> b), (a |=> b) until c. The diagnostic names the
       unbounded operator.

       Tracking mode is one per monitor and this shape wants both at once. The
       implication's own obligation is bounded, so it is SVA_MODE_RING; the
       operator above it is not, so it is SVA_MODE_COUNTED. For a *bounded* range
       the shift lands on the antecedent instead, which is the only correct place
       for it, and an unbounded range has no filler chain to grow there.

       A trigger ABOVE an unbounded operator is fine and is the point of the
       mode: req |=> s_eventually ack lowers. A trigger is existential, one state
       vector shared by all of its attempts, so it needs no bound and composes
       with either discipline. */
    SVA_ERROR_MODE_MIX = 9,
    /* An sva_prop_not libsva cannot carry, in either of two ways.

       It complements an operand that contributes a trigger of its own — the
       complement of an implication, `not (s |-> p)`. A trigger is the UNION of
       every attempt of the antecedent, which is sound only for the existential
       question a trigger is ever asked; complementing needs a per-attempt
       answer that the union has already discarded. Lowering it as `s #-# not p`
       is the intended lift — LRM 16.12.9 defines followed-by as
       `not (seq |-> not prop)`, so that is the same identity read backwards —
       but it is a rewrite on the tree, which libsva does not do.

       Or it sits below an operator that renumbers its operand's states, so the
       mask naming the complement's still-pending states cannot be carried
       through: a conjunction, an SVA_SHIFT_EVERY range, an until, or a
       followed-by. A disjunction and an SVA_SHIFT_SOME range relocate states as
       a block and are fine, which is what `implies` needs. Exactly the operator
       set that refuses an abort mask, for exactly that reason. */
    SVA_ERROR_UNSUPPORTED_COMPLEMENT = 10,
    /* An implication whose antecedent has no bounded longest match:
       (a ##[1:$] b) |-> c, (b[=2]) |-> c. The diagnostic names the antecedent.

       An obligation is owed per antecedent MATCH, so the cycle a match ends in
       has to report how many attempts ended one there — LRM 16.14.1 starts an
       evaluation of the consequent per match, and three antecedent matches
       ending together are three of them. Telling those attempts apart is what
       sva_monitor_trigger_window is for, and an unbounded antecedent has
       unboundedly many alive at once to tell apart.

       Refused rather than answered with the number of matching CYCLES, which is
       what a trigger carrying one bit answers and is wrong by however many
       attempts ended together. Both tracking modes refuse it: the obligation's
       discipline has no bearing on the antecedent. */
    SVA_ERROR_UNBOUNDED_ANTECEDENT = 11,
    /* One operand of an intersect or an and assigns a local variable slot that
       the sibling operand reads from the incoming context. LRM F.5.5 evaluates
       both operands from the SAME input context, so the sibling must not see
       that write; one valuation per thread cannot express both views at once,
       and the two operands of a product are one thread here. The diagnostic
       names the product. */
    SVA_ERROR_LOCAL_CROSS_READ = 12,
    /* A local variable assignment under an operator that determinises its
       operand — sva_seq_first_match or sva_prop_not. A determinised state
       stands for a SET of operand threads, and with an assignment beneath it
       those threads hold different valuations, which is exactly what one
       valuation per state cannot carry. Reading a local under either is fine:
       with no assignment there, every operand thread carries the valuation
       that flowed in. The diagnostic names the determinising operator. */
    SVA_ERROR_LOCAL_UNDER_DETERMINISATION = 13,
    /* A zero-delay fusion whose right-hand component reads a local variable
       slot the left-hand component assigns in the same cycle. Fusion conjoins
       the two guards into one cube, and this shape needs the cube split around
       the assignment: the left half is read before it and the right half
       after. The diagnostic names the delay. */
    SVA_ERROR_LOCAL_FUSED_READ = 14,
    /* Local variables under a universal operator with no upper bound — an
       unbounded sva_prop_always range, or either operand of an sva_prop_until.
       Those leave one attempt per cycle pending for ever, which SVA_MODE_COUNTED
       exists to collapse into a counter per state and a valuation makes
       uncollapsible: two attempts in one state with different valuations have
       different futures. An unbounded *sequence* obligation is not this shape
       and is supported. The diagnostic names the operator. */
    SVA_ERROR_LOCAL_UNBOUNDED_ATTEMPTS = 15,
    /* The IR declares local variable slots and this libsva does not lower them
       yet: the projection onto states, the flow sets and the threaded group step
       are unimplemented (../clotho-svalocal/docs/design/sva-local-variables.md
       is the design they are being built to). Every table below would read as
       empty, which a backend would take for "no local variables" and simulate as
       though the assignments were not there -- a wrong answer rather than a
       missing feature, so it is refused here instead.

       TODO: LRM 16.10. Delete this status once the group discipline lands. */
    SVA_ERROR_LOCAL_INCOMPLETE = 16,
} sva_status;

/* ------------------------------------------------------------------------- *
 * Objects, and the ABI contract that replaced their sizes
 *
 * These three are incomplete types. libsva allocates one and returns a pointer;
 * you hand that pointer back to be destroyed. Nothing here says how big any of
 * them is, because nothing outside libsva has any use for that.
 *
 * That is the whole point of the shape. They used to be byte arrays in your
 * storage, sized by constants in this header and checked when libsva was built
 * against the structs they stood for. The check only ever saw one side of the
 * pair: a header from one revision compiled against an archive from another
 * passed every build and then overran the caller's stack at run time, with
 * nothing to catch it, because a buffer the caller allocates cannot be
 * validated by the library that writes past the end of it. A size nobody
 * outside libsva knows cannot be got wrong.
 *
 * What replaces it is a version that can actually be compared. SVA_ABI_VERSION
 * is bumped whenever a signature, a layout or the meaning of a value changes.
 * A consumer MUST check it once at startup:
 *
 *     if (sva_abi_version() != SVA_ABI_VERSION) { ...refuse to run... }
 *
 * The macro is what your translation unit compiled against and the call is what
 * it linked, so the two disagreeing is precisely a stale header or a stale
 * archive — reported at the first call rather than executed.
 * ------------------------------------------------------------------------- */

#define SVA_ABI_VERSION 5

uint32_t sva_abi_version(void);

typedef struct sva_ir sva_ir;
typedef struct sva_monitor sva_monitor;
typedef struct sva_run sva_run;
/* Owned by the monitor it was read out of, and valid for as long as that is. */
typedef struct sva_automaton sva_automaton;

/* ------------------------------------------------------------------------- *
 * The IR
 * ------------------------------------------------------------------------- */

/* create/destroy, not init/deinit: those named setting an object up in storage
   you already had, and there is no such storage now. NULL if the allocation
   failed, which is the only way this can fail. */
sva_ir *sva_ir_create(void);
/* Accepts NULL. */
void sva_ir_destroy(sva_ir *ir);

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
    /* p implies q — LRM 16.12.2's *property* implication, which is exactly
       `not p or q`.

       Not sva_prop_implies, which is |-> and |=>: those take a SEQUENCE
       antecedent, quantify over its matches and contribute a trigger. This
       takes two properties and contributes nothing but a verdict. */
    SVA_IMPLIES = 2,
    /* p iff q — the conjunction of the two directions. Always refused with
       SVA_ERROR_MIXED_STRENGTH; the tag exists so the refusal can name iff. */
    SVA_IFF = 3,
} sva_junction;

/* What a monitor knows about one attempt.

   Three values, because a monitor watches a finite prefix and a prefix does not
   decide every property: an attempt that has neither completed a match nor run
   out of futures is PENDING, and pending is a verdict rather than the absence
   of one.

   sva_prop_not is what forces the lattice into the ABI. Complementing an
   automaton exchanges match for no-match, which is two of these; the third has
   no automaton to invert it and needs the rule written down. See
   sva_verdict_not. */
typedef enum sva_verdict {
    SVA_VERDICT_PENDING = 0,
    SVA_VERDICT_HOLDS = 1,
    SVA_VERDICT_FAILS = 2,
} sva_verdict;

typedef enum sva_shift {
    /* The range read disjunctively: the property need hold at only one offset
       in it. nexttime[n] p is lo == hi == n; eventually [m:n] p is wider. */
    SVA_SHIFT_SOME = 0,
    /* The range read conjunctively: always [m:n] p, the property at every
       offset. */
    SVA_SHIFT_EVERY = 1,
} sva_shift;

typedef enum sva_until {
    /* p until q — weak, and p is not required at the cycle q holds in. */
    SVA_UNTIL_WEAK = 0,
    /* p until_with q — weak, and p is required there too. */
    SVA_UNTIL_WEAK_WITH = 1,
    /* p s_until q — as SVA_UNTIL_WEAK, but q must arrive before the trace
       ends. */
    SVA_UNTIL_STRONG = 2,
    /* p s_until_with q — as SVA_UNTIL_WEAK_WITH, and q must arrive. */
    SVA_UNTIL_STRONG_WITH = 3,
} sva_until;

typedef enum sva_mode {
    /* One state vector per pending obligation attempt, sva_monitor_depth of
       them. */
    SVA_MODE_RING = 0,
    /* One uint64_t counter per obligation state, and the obligation
       determinised. */
    SVA_MODE_COUNTED = 1,
    /* One valuation group list per pending obligation attempt: a group is a
       state mask plus a valuation of the monitor's local variable slots. Always
       the mode when sva_monitor_local_count is non-zero, and never otherwise.
       sva_monitor_depth is the attempt bound where the obligation has one and
       zero where it does not. See the LOCAL VARIABLES section. */
    SVA_MODE_THREADED = 2,
} sva_mode;

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

/* A state mask is sva_automaton_words of these, least significant bit of word
   zero first. A slot mask is sva_ir_local_words of them, read the same way. */
typedef uint64_t sva_word;

/* A literal is a root, possibly complemented. */
typedef uint32_t sva_literal;
#define SVA_LITERAL_ROOT(literal)    ((literal) >> 1)
#define SVA_LITERAL_NEGATED(literal) ((literal) & 1u)

/* ------------------------------------------------------------------------- *
 * LOCAL VARIABLES
 *
 * A local variable is per evaluation ATTEMPT and per THREAD within it (LRM
 * 16.10): two threads of one attempt may hold different values for the same
 * variable and continue separately. libsva owns where a value moves; it never
 * owns what a value is. A SLOT is an opaque numbered box, a TYPE is a uint32_t
 * libsva stores and never reads, and an assignment's right-hand side is an
 * opaque HANDLE the backend evaluates.
 *
 * Slots are dense from zero in declaration order, which is the order the LRM
 * gives initialization: local variable formal arguments first, then
 * assertion_variable_declarations, each in source order. An initializer runs
 * when an attempt is created, in slot order.
 *
 * A local variable makes a thread a (position, valuation) pair, and a state
 * mask cannot tell two valuations apart, so a monitor with slots is lowered
 * SVA_MODE_THREADED: per attempt, a list of GROUPS, each a valuation plus the
 * mask of the states reachable with it. Every position sharing a valuation
 * shares a mask, so the mask arithmetic of the other two modes is unchanged and
 * a group list of one is exactly them. The step is in the RUNNING section.
 *
 * Reads are declared, not discovered. sva_atom_reads tells libsva which slots
 * an atom's expression reads; libsva propagates that to roots and states, so
 * sva_monitor_root_local says which roots a backend must evaluate once per
 * group rather than once per cycle, and so the four SVA_ERROR_LOCAL_* shapes
 * can be found rather than mislowered. An atom with no declared reads is
 * assumed to read none.
 * ------------------------------------------------------------------------- */

/* A slot with no initializer: unassigned when the attempt begins. LRM 16.10 is
   explicit that local variables have no default initial value, and a reference
   to an unassigned one is a static error in the source language rather than
   something with a value here. */
#define SVA_LOCAL_NO_INIT UINT32_MAX

/* Declare one local variable slot. `type` and `init` are opaque; `*out`
   receives the slot number, dense from zero in declaration order. */
sva_status sva_local_declare(sva_ir *ir, uint32_t type, uint32_t init,
                             uint32_t *out);
uint32_t sva_ir_local_count(const sva_ir *ir);
/* Words in a slot mask: sva_ir_local_count rounded up to 64. */
uint32_t sva_ir_local_words(const sva_ir *ir);

/* The slots `atom` reads. `slots` is sva_ir_local_words words, least
   significant bit of word zero being slot zero. Every slot named must already
   be declared. Calling this twice for one atom replaces the mask, and not
   calling it leaves the atom local-free. */
sva_status sva_atom_reads(sva_ir *ir, uint32_t atom, const sva_word *slots);

/* One local variable assignment: `slot = handle`, where `handle` is opaque and
   the backend evaluates it against the arriving valuation. A compound
   assignment (`x += e`) and an increment are this, with the read folded into
   the handle's own expression — libsva does not model operators on values. */
typedef struct sva_assign {
    uint32_t slot;
    uint32_t handle;
} sva_assign;

/* `sequence`, with `count` assignments performed in order at the end of every
   nonempty match of it — LRM 16.10's `( sequence_expr, sequence_match_item )`.
   They run AFTER the last cycle's own boolean holds, so an assignment and a
   guard reading the same slot in one match item see the old value in the guard
   and the new one after it.
 *
 * `sequence` must not admit an empty match, which this IR guarantees for every
 * sequence it can build, so there is nothing to check here that 16.10's
 * prohibition would catch.
 *
 * count of zero is legal and returns `sequence` unchanged, so a front-end need
 * not special-case an empty match item list. */
sva_status sva_seq_match(sva_ir *ir, sva_node sequence, const sva_assign *assigns,
                         uint32_t count, sva_node *out);

/* Property layer. */
sva_status sva_prop_sequence(sva_ir *ir, sva_strength strength, sva_node sequence,
                             sva_node *out);
sva_status sva_prop_implies(sva_ir *ir, sva_implication kind, sva_node antecedent,
                            sva_node consequent, sva_node *out);
/* not p — LRM 16.12.2. The operand's verdict, complemented in the three-valued
   lattice of sva_verdict: holds where `property` fails, fails where it holds,
   pending where it is pending.

   Nothing to do with sva_bool_not, which complements a boolean TERM within one
   cycle. This inverts a whole attempt's verdict, and with it the strength
   `property` contributes — a not over a weak operand is strong.

   It costs a determinisation of the operand, so it is the second operator whose
   state count can be exponential in its operand's; SVA_ERROR_STATE_LIMIT names
   the not if it is. It does NOT force SVA_MODE_COUNTED: a complement borrows
   its operand's bound, so a ring-tracked operand stays ring-tracked. */
sva_status sva_prop_not(sva_ir *ir, sva_node property, sva_node *out);
/* One of the four property junctions. Each constrains the strengths its
   operands end up carrying, which is a property of their subtrees rather than
   of their own tags: sva_lower reports SVA_ERROR_MIXED_STRENGTH, this call does
   not. SVA_CONJUNCTION and SVA_DISJUNCTION want the two equal, SVA_IMPLIES
   wants them to differ, and SVA_IFF is always refused. */
sva_status sva_prop_junction(sva_ir *ir, sva_junction kind, sva_node lhs,
                             sva_node rhs, sva_node *out);
/* A property shifted by lo..hi cycles. SVA_UNBOUNDED as hi spells the unbounded
   `always p` / `s_eventually p`, which lowers to an SVA_MODE_COUNTED monitor.
   SVA_SHIFT_EVERY additionally wants an operand matching in a single cycle and
   reports SVA_ERROR_UNBOUNDED_OBLIGATION otherwise; SVA_SHIFT_SOME does not. */
sva_status sva_prop_shift(sva_ir *ir, sva_shift kind, uint32_t lo, uint32_t hi,
                          sva_node property, sva_node *out);
/* One of the four `until` forms over two *properties*: lhs holds up to the cycle
   rhs holds in, inclusively for the _WITH kinds, and the STRONG kinds require
   that cycle to arrive.

   The kind carries the end-of-trace disposition, so unlike sva_prop_junction
   this does not care whether the operands' own strengths agree — LRM 16.12.10
   spells four operators out precisely to say which. An until always lowers to an
   SVA_MODE_COUNTED monitor: nothing bounds how long lhs may have to hold.

   lhs must match in a single cycle, and sva_lower reports
   SVA_ERROR_UNBOUNDED_OBLIGATION naming it if not. rhs may span any number of
   cycles. */
sva_status sva_prop_until(sva_ir *ir, sva_until kind, sva_node lhs, sva_node rhs,
                          sva_node *out);
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
 * The two are tracked differently. The trigger is asked how many antecedent
 * matches end this cycle, so one state vector holds every attempt of it and each
 * state carries the set of attempts whose threads sit there — see
 * sva_monitor_trigger_window. The obligation is per-attempt: merging attempts is
 * unsound, because one attempt's threads can die while another's live and the
 * union hides the failure. What DOES collapse is the obligation itself. Matches
 * ending in one cycle impose the identical future, so one instance of it serves
 * them all, carrying the number of attempts it stands for; what does not
 * collapse is that number, and reporting one verdict for three attempts is a
 * wrong answer rather than a coarse one.
 *
 * How the pending attempts are held is sva_monitor_mode, and there are two
 * disciplines because there are two kinds of obligation.
 *
 * SVA_MODE_RING. Each pending attempt gets its own state vector, and
 * sva_monitor_depth says how many there can be. That bound is the obligation's
 * longest match, so it is exact rather than a heuristic: an attempt seeded in
 * cycle t has matched or run out of live states by cycle t + depth - 1.
 *
 * SVA_MODE_COUNTED. An obligation with no bounded longest match has no such ring
 * — always p, s_eventually p and the until family can leave attempts pending
 * arbitrarily long, so arbitrarily many are in flight. Their obligation is
 * *determinised*, and totally: the death of an attempt is a state, not the
 * absence of one. One determinised state then fully characterises an attempt's
 * future, so attempts sitting in the same state may share it, and a backend
 * carries one uint64_t count per obligation state instead of a vector per
 * attempt. The count is what the verdict needs: two attempts is two failures,
 * and a merged vector could only ever report one.
 *
 * The counted step is, per cycle:
 *
 *     next[*] = 0
 *     for each state s with cnt[s] != 0:
 *         next[chosen(follow(s) & satisfied)] += cnt[s]
 *     if seeding: next[chosen(first & satisfied)] += fires
 *     for each state t with next[t] != 0:
 *         if      (last & bit(t)) matches  += next[t], next[t] = 0
 *         else if (dead & bit(t)) failures += next[t], next[t] = 0
 *     cnt = next
 *
 * and two rules in it are not optional.
 *
 * ONE SUCCESSOR, LOWEST INDEX WINS. `chosen(m)` above is the lowest set bit of
 * m, and a count MUST move along exactly one enabled transition. Several can be
 * enabled at once even though the obligation is determinised: a determinised
 * state is a (subset, cell) pair and two cells may share a valuation. Such
 * siblings are bisimilar — same subset, so the same exits and the same
 * successors — which is why taking the lowest is a choice rather than an
 * approximation. Summing a count into every enabled successor instead
 * manufactures attempts out of nothing, and does so only on obligations whose
 * cells happen to overlap. Isolating the lowest set bit is branch-free
 * (`m & -m`), so there is no reason to approximate it.
 *
 * DEAD, NOT EMPTINESS. The construction is total, so a count never falls off the
 * automaton and "no live states" never happens. Failure is sva_automaton_dead.
 *
 * COUNTING THE TRIGGER. `fires` above is a COUNT, not a flag: it is how many
 * antecedent matches end this cycle, and an obligation instance is worth that
 * many verdicts. When sva_monitor_trigger_window is zero it is 0 or 1 and
 * (current & last) != 0 is the whole of it. When it is W, a backend carries a
 * W-bit id set per trigger state and maintains, per cycle:
 *
 *     ids'[*] = {}
 *     for each state s with ids[s] != {}:
 *         for each t in follow(s) & satisfied: ids'[t] |= ids[s]
 *     for each t in first & satisfied: ids'[t] |= { cycle mod W }
 *     fires = |union of ids'[t] over t in last|
 *
 * UNION, NOT ADDITION, and this is the rule the whole mechanism exists for. A
 * match is an interval [start, end] (LRM 16.9), so what is being counted is
 * distinct starts. One attempt can reach an exit along several paths in one
 * cycle — `a ##[1:2] b ##[1:2] c` matches [t, t+3] through either gap — and that
 * is ONE match. Adding per-state thread counts reports two; a set union reports
 * one, and still tells two attempts apart. An id may be reused every W cycles
 * without being retired: every trigger state can reach an exit, so a thread
 * outliving the longest match does not exist.
 *
 * An abort with sva_monitor_abort_covers_trigger reads the same ids: it resolves
 * every attempt with a live antecedent thread, which is the union over ALL
 * states rather than over the exits, one verdict each, and clears them.
 *
 * Counters are uint64_t and are not saturated: at seven and a half million ticks
 * a second a uint32_t overflows in about ten minutes of wall clock, which is a
 * reachable simulation, and uint64_t is not reachable at all.
 * ------------------------------------------------------------------------- */

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

/* Lower one directive. `options` and `diagnostic` may be NULL; `*out` receives a
   monitor to destroy with sva_monitor_destroy, and is written only on success,
   while `diagnostic` is written only on failure. The IR may be torn down
   afterwards: a monitor does not refer back to it, except that its root nodes
   are only meaningful against it.

   This one keeps its status return and its out parameter rather than handing
   the monitor back the way sva_ir_create does, because running out of memory is
   one of eleven things that can go wrong here and the other ten have to be
   told apart. */
sva_status sva_lower(sva_monitor **out, const sva_ir *ir, uint32_t directive,
                     const sva_options *options, sva_diagnostic *diagnostic);
/* Accepts NULL. */
void sva_monitor_destroy(sva_monitor *monitor);

uint32_t sva_monitor_id(const sva_monitor *monitor);
sva_kind sva_monitor_kind(const sva_monitor *monitor);
uint32_t sva_monitor_clock(const sva_monitor *monitor);

/* How the monitor's pending obligation attempts are held, and so which step a
   backend emits. See the section comment above for the two rules
   SVA_MODE_COUNTED puts on one. */
sva_mode sva_monitor_mode(const sva_monitor *monitor);

/* Obligation attempts that can be pending at once: the obligation's longest
   match, and so the number of state vectors a backend allocates. An attempt
   started in cycle t has either matched or run out of live states by cycle
   t + depth - 1.

   ZERO, and meaningless, in SVA_MODE_COUNTED. There is no bound to report there
   — that is what the mode exists for — and a counted backend allocates one
   uint64_t per obligation state instead. Do not read a zero as a lowering bug. */
uint32_t sva_monitor_depth(const sva_monitor *monitor);
/* Antecedent attempts the trigger tells apart: the number of attempt ids a
   backend rotates through, with one id set per trigger state. See COUNTING THE
   TRIGGER above for what maintaining them costs and why a union is not a sum.

   ZERO when one bit answers instead, which is the common case and folds the
   whole mechanism away. That is an antecedent matching in a single cycle, and
   equally one whose matches all span the SAME number of cycles: distinct start
   cycles then end in distinct cycles, so at most one match can end per cycle.
   `a |=> b` and `(a ##1 b) |-> c` both report zero. So does every monitor
   without a trigger.

   An unbounded antecedent is refused rather than tracked to a bound — see
   SVA_ERROR_UNBOUNDED_ANTECEDENT — so a non-zero window is always a real
   bound. */
uint32_t sva_monitor_trigger_window(const sva_monitor *monitor);
sva_strength sva_monitor_strength(const sva_monitor *monitor);
/* The disposition of an attempt still pending when the trace ends — the whole
   content of sva_monitor_strength, said as a verdict. Never
   SVA_VERDICT_PENDING: a trace that has ended decides every attempt.

   Prefer this to reading the strength and branching on it. It is the same
   number and it cannot be got backwards. */
sva_verdict sva_monitor_pending(const sva_monitor *monitor);
/* The negation table of the verdict lattice, and the normative one: anything
   reporting three-valued results must agree with it rather than re-deriving it.

   The row that matters is SVA_VERDICT_PENDING, which maps to itself. Negating
   an undecided attempt leaves it undecided, however certain its outcome has
   become — which is what makes `not` an involution, and is why a complement in
   libsva never resolves an attempt one cycle before its operand does. */
sva_verdict sva_verdict_not(sva_verdict verdict);
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

   When it holds, resolve EVERY attempt with a live antecedent thread with the
   abort's effect — the ones whose match ends this cycle and the ones still
   matching alike, one verdict each and not one between them — clear their
   threads out of the trigger so that none of them completes an antecedent later,
   and seed no attempt from any of them. The count is the union of the id sets
   over every trigger state; without a window there is at most one such attempt
   and it is the one matching now.

   An attempt with no live antecedent thread is not one of them: it can no longer
   match, so the implication already holds vacuously and there is nothing under
   evaluation left to abort.

   This is not the same property as the abort written inside the consequent,
   which arms only once the consequent starts — a cycle later, for |=>. So an
   abort above an implication cannot be pushed down into it.

   If several of these hold at once, the LAST in the list wins: the list runs
   innermost to outermost, and an outer abort beats an inner verdict landing in
   the same cycle. */
bool sva_monitor_abort_covers_trigger(const sva_monitor *monitor, uint32_t abort);
/* sva_automaton_words(sva_monitor_obligation(monitor)) words. */
const uint64_t *sva_monitor_abort_mask(const sva_monitor *monitor, uint32_t abort);

/* The boolean nodes to evaluate, one bit each, in this order. This is the whole
   of what a backend has to understand; everything past it is bit arithmetic over
   the result. Walk one with sva_bool_read — a root is any boolean expression,
   not necessarily an atom.

   Once per cycle, except for the roots sva_monitor_root_local names, which read
   a local variable and are evaluated once per valuation group. */
uint32_t sva_monitor_root_count(const sva_monitor *monitor);
const sva_node *sva_monitor_roots(const sva_monitor *monitor);
/* [(root_count + 63) / 64] — the roots that read a local variable slot, and so
   must be evaluated per group against that group's valuation rather than once
   for the cycle. All-zero without local variables, which is what keeps a
   local-free monitor's step exactly what it was. */
const sva_word *sva_monitor_root_local(const sva_monitor *monitor);

/* Local variable slots, in declaration order. Zero for a monitor without any,
   which is then never SVA_MODE_THREADED. */
uint32_t sva_monitor_local_count(const sva_monitor *monitor);
/* Words in a slot mask. */
uint32_t sva_monitor_local_words(const sva_monitor *monitor);
/* The opaque type the slot was declared with. */
uint32_t sva_monitor_local_type(const sva_monitor *monitor, uint32_t slot);
/* The opaque handle of the slot's initializer, or SVA_LOCAL_NO_INIT. Run them in
   slot order when an attempt is created, before the attempt's first cycle is
   evaluated: an initializer may read a lower-numbered slot's initial value. */
uint32_t sva_monitor_local_init(const sva_monitor *monitor, uint32_t slot);

/* Assignment sites, dense from zero. A site belongs to whichever states'
   programs name it, and its handle is evaluated against the valuation arriving
   at that state. */
uint32_t sva_monitor_assign_count(const sva_monitor *monitor);
uint32_t sva_monitor_assign_slot(const sva_monitor *monitor, uint32_t site);
uint32_t sva_monitor_assign_handle(const sva_monitor *monitor, uint32_t site);

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
/* [words] — states an obligation attempt has FAILED in. Meaningful in
   SVA_MODE_COUNTED; all-zero otherwise.

   It is the states standing for the empty subset of the operand's — no
   continuation of the trace can match from here. Do not re-derive it as "states
   with no path to an exit": that is a wider set, and for `always p`, which has
   no exits at all because "for ever" has no finite witness, it is every state. */
const sva_word *sva_automaton_dead(const sva_automaton *automaton);
/* [words] — the states carrying `guard`. Building the satisfied mask folds over
   guards rather than gathering per state. */
const sva_word *sva_automaton_guard_states(const sva_automaton *automaton,
                                           uint32_t guard);

/* [count] — the assignment sites to perform, in order, on arriving in `state`,
   and after that state's own guard has been found to hold. Writes the count and
   returns the sites, or writes 0 and returns NULL for a state that assigns
   nothing, which is every state of a monitor without local variables.

   A state produced by a zero-delay fusion carries the concatenation of the
   fused positions' programs, left to right. */
const uint32_t *sva_automaton_program(const sva_automaton *automaton,
                                      uint32_t state, uint32_t *count);
/* [sva_monitor_local_words] — the slots whose values may still be read from
   `state` onward: F.5.4's flow set, projected onto the automaton. A valuation is
   restricted to this before two groups are compared, which is what lets groups
   differing only in dead slots merge again. All-zero without local variables. */
const sva_word *sva_automaton_live(const sva_automaton *automaton, uint32_t state);

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
 *   4. Advance the trigger, then seed it. Count the matches that now end.
 *   5. If an abort covering the trigger holds, resolve every attempt with a live
 *      antecedent thread with its effect, clear them, and seed nothing.
 *   6. Advance every pending obligation attempt.
 *   7. Seed a new attempt: every cycle with no trigger, in the cycle the trigger
 *      fires for |->, and the cycle after for |=>. It stands for as many attempts
 *      as step 4 counted matches.
 *   8. Take verdicts. An accepting state discharges an attempt; no live states
 *      fails it — or in SVA_MODE_COUNTED, a count arriving in a dead state,
 *      because there the automaton is total and no live states never happens.
 *      Either way it is worth as many verdicts as it stands for attempts.
 *   9. At the end of a finite trace, a pending attempt holds if the property is
 *      weak and fails if it is strong. Every attempt still pending counts, in
 *      either mode, however many share a state vector or a counter.
 *
 * Seeding before taking verdicts is not incidental: under |-> the obligation
 * starts on the trigger's last cycle, so a one-cycle consequent has to be able
 * to discharge in the cycle it was created.
 *
 * SVA_MODE_THREADED changes steps 3, 6, 7 and 8 and nothing else. A pending
 * attempt is a list of GROUPS rather than one state vector; a group is a
 * valuation of the local variable slots plus the mask of states holding it.
 * Per group:
 *
 *   3'. The roots sva_monitor_root_local names are evaluated against this
 *       group's valuation; the rest are the cycle's, evaluated once. The
 *       satisfied mask then folds over guards exactly as before.
 *   6'. next = advance(group.mask) & satisfied(group), unchanged arithmetic.
 *   6". SPLIT ON ASSIGNMENT. For each state in `next` with a non-empty
 *       sva_automaton_program, emit a new group carrying that state alone and
 *       the valuation the program produces, and clear that state from this
 *       group's mask. States that assign nothing stay where they are.
 *   6"'. RESTRICT AND MERGE. Mask each group's valuation to the union of
 *       sva_automaton_live over its states, then merge groups with equal
 *       valuations by OR-ing their masks. Merging is not an optimisation the
 *       way it looks: without it an accumulating assignment under a repetition
 *       grows a group per cycle for ever.
 *   7'. A seeded attempt starts as one group whose valuation is the
 *       initializers of sva_monitor_local_init, or — under an implication — the
 *       valuation that flowed out of the antecedent match that seeded it. Two
 *       antecedent matches ending in one cycle with DIFFERENT valuations seed
 *       two attempts, not one attempt worth two verdicts.
 *   8'. An attempt holds when any of its groups meets `last` and fails when
 *       every group's mask is empty. Verdicts are per attempt, as before.
 *
 * A trigger with local variables carries groups too, each with the id set of
 * step 4, so the valuation reaching an exit is the one handed to the attempt.
 *
 * The group count is the resource local variables consume and it is the caller's
 * to bound: nothing here caps it.
 * ------------------------------------------------------------------------- */

typedef struct sva_report {
    /* Obligation attempts that ran out of live states, or in SVA_MODE_COUNTED
       reached a dead state.

       Wide because one cycle can resolve many attempts at once: one dead state
       may hold the count of every attempt that has just failed together, and one
       state vector may stand for every antecedent match that seeded it. */
    uint64_t failures;
    /* Attempts discharged by a completed match. A cover counts these. */
    uint64_t matches;
    /* The reset condition held, so everything in flight was abandoned. */
    bool reset;
} sva_report;

/* The monitor must outlive the run. NULL if the allocation failed. */
sva_run *sva_run_create(const sva_monitor *monitor);
/* Accepts NULL. */
void sva_run_destroy(sva_run *run);

/* Abandon everything in flight, as a reset does. */
void sva_run_clear(sva_run *run);

/* Values, for the reference run only.

   libsva stores no value and computes on none, so a run of an SVA_MODE_THREADED
   monitor asks for the two things it cannot know: what an assignment site
   produces, and whether a local-reading root holds. A valuation is
   sva_monitor_local_count uint64_t slots, in slot order, which is a stand-in for
   whatever the backend's own values are — enough to differential-test the group
   discipline, not a value model anyone has to adopt.

   `frame` is the valuation arriving at the state, so an assignment reading its
   own slot (`x += e`) or a lower one (`y = x && f`) reads it there. Sites of one
   program are called in order and each sees the previous one's write.

   Binding is required before stepping a threaded monitor and pointless
   otherwise. */
typedef struct sva_locals {
    uint64_t (*assign)(void *user, uint32_t site, const uint64_t *frame);
    bool (*root)(void *user, uint32_t root, const uint64_t *frame);
    void *user;
} sva_locals;

void sva_run_bind(sva_run *run, const sva_locals *locals);

/* Advance one cycle. `root_bits` holds one bit per sva_monitor_root_count, in
   order, packed into sva_words. */
sva_report sva_run_step(sva_run *run, const sva_word *root_bits);

/* Resolve what is still pending because the trace ended. */
sva_report sva_run_finish(sva_run *run);

#ifdef __cplusplus
}
#endif

#endif /* SVA_H */
