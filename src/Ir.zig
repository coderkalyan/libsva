//! libsva intermediate representation.
//!
//! # Structure
//!
//! A flattened tree: every node lives in one array and refers to other nodes by
//! index. Three layers share that array, distinguished by `Node.Tag`:
//!
//!   1. **Boolean layer** — the "letters" of the automaton alphabet: boolean
//!      combinations of opaque atoms. Each maximal boolean node is the guard on
//!      an NFA transition.
//!   2. **Sequence layer** — regular patterns over cycles: delayed
//!      concatenation, disjunction, conjunction, length-matching intersection,
//!      a boolean required throughout a match, shortest-match selection, and
//!      the three flavors of repetition.
//!   3. **Property layer** — sequences promoted to properties, implication,
//!      and the verdict combinators over them, negation included.
//!
//! A leaf of the boolean layer is an `Atom`: an opaque `u32` handle the
//! front-end assigns. The IR never inspects one — it stores no text, no types,
//! and no widths, and it does not know how an atom is sampled or evaluated.
//! That is entirely the caller's business; a handle is only meaningful within
//! the IR instance that produced it.
//!
//! **Atoms are assumed two-valued.** `bnot` is boolean complement: the guard
//! layer pushes it down to a leaf and evaluates it as `!atom`. A front-end whose
//! own booleans are *three*-valued must therefore not emit `bnot` over such a
//! term. SystemVerilog is the motivating case: LRM 16.6 interprets an unknown
//! expression as false, so under `a = 1'bx` both `a` and `!a` are false and
//! `a || !a` does not hold — whereas `bnot` of the collapsed bit would report it
//! as true. Such a front-end lowers to negation normal form instead, pushing
//! negation to the leaves itself and supplying each polarity as its own atom;
//! De Morgan is sound in three-valued logic, so nothing else changes. `band` and
//! `bor` need no such care: they commute with the collapse to false, because an
//! unknown operand can never turn a known outcome around.
//!
//! The cost of that encoding is that two polarities of one term are unrelated
//! atoms, so a cube holding both is not recognised as unsatisfiable. Teaching
//! the guard layer that a pair of atoms is mutually exclusive would recover it
//! without this file learning anything about widths or unknown values, and is
//! the intended move if state counts ever justify it.
//!
//! Operands always point *backwards*: a node's operands are nodes that were
//! built before it, so an index is strictly less than the index of the node
//! referring to it. The tree is therefore acyclic and already in topological
//! order — a consumer can fold over `nodes` front to back in one pass, and a
//! producer must build bottom-up. The constructors assert this.
//!
//! Two encoding details make a hand-rolled `{lhs, rhs}` walk wrong, so a pass
//! that needs a node's edges must call `operandsOf`: `bool_const` and `atom`
//! hold a payload rather than an index in `lhs`, and `seq_delay` keeps its
//! right operand in `extra` while its `rhs` is an `extra` offset.
//!
//! # Semantics
//!
//! ## Traces, cycles and clocks
//!
//! A *trace* is a sequence of cycles, finite (a terminated simulation) or
//! infinite (a formal model). A cycle assigns a boolean value to every atom,
//! and the engine is stepped once per cycle.
//!
//! What delimits a cycle is outside the alphabet: `Assertion.clock` is a
//! `Clock`, a handle space of its own, because a clock is not a term the trace
//! assigns a value to per cycle — it is what fixes the sampling points that
//! give the atoms their per-cycle values in the first place. Choosing an edge
//! is likewise the front-end's job: the rising and falling edge of one physical
//! signal are two distinct `Clock` handles.
//!
//! ## Atoms
//!
//! The atom space must be *dense*: an IR that mentions `n` atoms uses exactly
//! the handles `0..n - 1`, and `atom_count` is that `n`, maintained as nodes
//! are added. libsva never remaps, so a backend can allocate a per-cycle
//! valuation of `atom_count` bits and index it by handle directly. Holes are
//! not rejected — they just waste a bit apiece.
//!
//! Every atom the IR mentions is the payload of an `atom` node, so a scan over
//! `nodes` finds all of them; clocks are a separate space, reachable from the
//! assertion table.
//!
//! ## Attempts
//!
//! An assertion is **implicitly quantified over start cycles**: one *attempt*
//! is seeded at every cycle of the trace, independently, and the directive's
//! verdict is over all of them — an `assert`/`assume` holds only if every
//! attempt holds, a `cover` reports each attempt that matches. That outer
//! quantification is not written in the tree and cannot be opted out of, which
//! is why there is no `always` operator: a property node describes a single
//! attempt starting at a single cycle.
//!
//! ## Verdicts and strength
//!
//! Within an attempt, a sequence property (`prop_seq_weak`/`prop_seq_strong`)
//! **holds** as soon as its sequence completes a match, and **fails** as soon
//! as the sequence can no longer match — for a subset-simulation monitor, the
//! cycle its state set goes empty. Until one of those happens the attempt is
//! *pending*.
//!
//! An implication attempt holds vacuously if its antecedent never completes a
//! match; otherwise every antecedent match obliges the consequent property,
//! and the attempt fails if any of those consequent attempts fails.
//! Followed-by is the same shape read existentially: it needs *some* match of
//! the antecedent with the consequent holding after it, so an antecedent that
//! never matches makes it fail where it would make an implication hold.
//!
//! `prop_and` and `prop_or` combine verdicts rather than matches — each
//! operand keeps its own schedule, and the conjunction is pending until both
//! have resolved. `prop_delay` and `prop_always` shift a property by a range of
//! cycles, disjunctively and conjunctively respectively; an unbounded range
//! spells `s_eventually p` and `always p`. The four `prop_until*` tags are the
//! LRM's `until` family: the left operand holds up to the cycle the right one
//! does, inclusively for the `_with` forms, and the `s_` forms require that
//! cycle to arrive.
//!
//! `prop_not` complements a verdict. There are three of them — holds, fails and
//! pending — and negation swaps the first two while fixing the third, so a
//! complemented attempt resolves in exactly the cycle its operand resolved and
//! not a cycle sooner. `prop_implies` and `prop_iff` are reductions onto it:
//! `not p or q`, and the conjunction of the two directions.
//!
//! A finite trace ends with attempts still pending, and their disposition is
//! the property's **strength**:
//!
//!   - A *weak* property discharges a pending obligation: it holds.
//!   - A *strong* property requires the obligation to have been met within the
//!     trace: it fails.
//!
//! Strength is carried by `prop_seq_weak`/`prop_seq_strong`; every operator
//! above them takes its strength from its operands, the `until` family
//! excepted — there the tag fixes it. Implication takes it from
//! the consequent, and a never-completed antecedent is vacuous under either —
//! strength governs the consequent, not the trigger. `prop_and` and `prop_or`
//! require their two operands to agree, because a pending conjunction of a
//! weak and a strong operand has no single end-of-trace disposition. On an
//! infinite trace the distinction is only observable for obligations that are
//! never resolved.
//!
//! `prop_not` inverts strength, LRM 16.12.2: a pending `not p` at the end of a
//! trace is a pending `p`, whose disposition is then complemented. That is why
//! negation cannot be a pure automaton operation, and it is what makes
//! `prop_implies` demand operands of *differing* strength — `not p or q` puts
//! the inverted strength next to `q`'s under `prop_or`'s agreement rule.
//!
//! ## Aborts
//!
//! An abort (`prop_accept_on`, `prop_reject_on`, and their `sync_` forms)
//! resolves the property beneath it the moment its condition holds, while that
//! property is still being evaluated: an accept resolves it as holding, a
//! reject as failing. It is scoped to that subtree, so an abort below an
//! implication leaves the antecedent alone and later matches of it still raise
//! obligations.
//!
//! This is the distinction against `reset` below, and it is not a shade of
//! meaning: a reset **abandons**, leaving no verdict at all, while an accept
//! **discharges**, which is a pass a `cover` counts.
//!
//! ## Empty matches
//!
//! Every sequence in the IR matches at least one cycle. Repetition counts are
//! `>= 1` and nothing else can introduce an empty match, so `seq_delay`,
//! `seq_and`, `seq_intersect`, `seq_throughout`, implication antecedents and
//! property promotion need no empty-match special cases and have none.
//!
//! For two of them the ban is load-bearing rather than incidental.
//! `seq_throughout` is `(exp[*1:$]) intersect seq` only because no match is
//! empty — the LRM's `[*0:$]` would otherwise say something weaker — and
//! `seqWithin` can pad with a `0..$` delay against a one-cycle `true` where
//! the LRM pads with `1[*0:$]`, because a delay of 0 fuses that cycle into the
//! match beside it.
//!
//! A source language with a zero-count repetition must therefore eliminate it
//! before building the IR — expanding into a disjunction whose empty
//! alternative is folded into the surrounding context, per that language's own
//! empty-match rules, which tend to be subtle and sensitive to how the
//! concatenation associates. libsva declines to guess: the constructors reject
//! a zero count outright.
//!
//! ## Reset
//!
//! If an assertion's `reset` condition holds in some cycle, every attempt of
//! that assertion still in flight is abandoned at that cycle: abandoned
//! attempts neither hold nor fail nor match, whatever their strength, and a
//! fresh attempt is seeded again from the next cycle onward.
//!
//! # Checking
//!
//! Every convenience constructor asserts the well-formedness it can check for
//! free — layer typing of each operand, backward-pointing in-range indices,
//! ordered and non-degenerate ranges. Those are `std.debug.assert`s: they are
//! the enforcement mechanism for everything above, so build against libsva in
//! `ReleaseSafe` unless you have a reason not to. They vanish in `ReleaseFast`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

/// A local variable slot: an opaque type and an optional initializer. Both
/// handles are the front-end's; libsva stores them and never inspects them.
pub const Local = struct {
    /// Opaque type handle. Only the front-end knows what a slot holds.
    type: u32,
    /// The initializer's opaque handle, or `local_no_init`. LRM 16.10: the
    /// initialization assignment is performed when an attempt creates the
    /// slot's copy, in declaration order, so an initializer may read a
    /// lower-numbered slot that was already initialised this attempt.
    init: u32,
};

/// A slot with no initializer. LRM 16.10: "Local variables do not have default
/// initial values" — an uninitialised slot is *unassigned* at the start of the
/// attempt, and reading one before it is assigned is a static error the
/// front-end rules out; libsva does not represent an unassigned value.
pub const local_no_init: u32 = std.math.maxInt(u32);

const Ir = @This();

/// The boolean/sequence/property tree.
nodes: std.MultiArrayList(Node),
/// Top-level directives, each rooted at a property node.
assertions: std.MultiArrayList(Assertion),
/// Side pool of `u32` payloads for nodes with a variable or multi-field shape
/// (delay ranges, repetition counts).
extra: std.ArrayListUnmanaged(u32),
/// Width of the atom space: one past the highest `Atom` handle any node
/// mentions, and so the size of the per-cycle valuation a backend allocates.
/// Maintained by `atom`.
atom_count: u32,
/// Local variable slots in declaration order, dense from zero. Empty for the
/// local-free assertions that are the overwhelming majority, and every table
/// and code path keyed on locals is gated on this being non-empty.
locals: std.ArrayListUnmanaged(Local),
/// Per-atom read masks: a flat `atom_count * local_words` array of `u64`,
/// atom-major, so atom `a`'s mask is `[a * local_words ..][0..local_words]`.
/// Least-significant bit of word zero is slot zero. LRM 16.10 lets a local
/// appear in any expression the atom stands for, and which slots an atom reads
/// is declared by the front-end rather than discovered here.
atom_reads: std.ArrayListUnmanaged(u64),
/// Words in one slot mask: `locals.len` rounded up to 64. Cached because it is
/// the stride of `atom_reads` and every mask read recomputes it otherwise.
local_words: u32,

/// An empty IR. Allocates nothing until the first `add*` call.
pub fn init() Ir {
    return .{
        .nodes = .{},
        .assertions = .{},
        .extra = .empty,
        .atom_count = 0,
        .locals = .empty,
        .atom_reads = .empty,
        .local_words = 0,
    };
}

pub fn deinit(ir: *Ir, gpa: Allocator) void {
    ir.nodes.deinit(gpa);
    ir.assertions.deinit(gpa);
    ir.extra.deinit(gpa);
    ir.locals.deinit(gpa);
    ir.atom_reads.deinit(gpa);
}

/// An opaque handle to a boolean-layer leaf term. The front-end assigns these
/// and is solely responsible for evaluating and sampling them; the IR treats a
/// handle as an uninterpreted letter and only ever compares them for equality.
///
/// Handles must be dense from zero (see the semantics section above), and
/// `maxInt(u32)` is not one: the space is `0..maxInt(u32) - 1` so that
/// `atom_count` can always be formed.
pub const Atom = enum(u32) { _ };

/// An opaque handle to whatever delimits a cycle for an assertion. Distinct
/// from `Atom`: a clock is not a term with a value each cycle, it is what
/// defines the cycles. Edge selection belongs to the front-end, which allocates
/// a separate handle per edge of a signal.
pub const Clock = enum(u32) {
    /// The assertion is unclocked and simply steps with the engine.
    none = std.math.maxInt(u32),
    _,

    pub fn unwrap(c: Clock) ?Clock {
        return if (c == .none) null else c;
    }
};

/// One entry in the flattened tree. `tag` selects how `data` is interpreted;
/// see `Tag` for the per-tag encoding.
pub const Node = struct {
    /// Which operator this node is, and how `data` is encoded.
    tag: Tag,
    /// Operands/payload, interpreted per `tag`.
    data: Data,

    /// Index into `Ir.nodes`. Every edge in the tree is present: nothing here
    /// is optional, so there is no sentinel and every index names a node.
    pub const Index = enum(u32) { _ };

    /// Raw payload whose meaning depends on `tag`. `extern` so the layout is
    /// fixed for hashing and on-disk use.
    pub const Data = extern struct {
        /// First operand: a `Node.Index`, an `Atom`, or a literal.
        lhs: u32 = 0,
        /// Second operand: a `Node.Index`, an `Extra.Index`, or unused.
        rhs: u32 = 0,
    };

    /// The operator set. Only what the lowering implements is listed: there is
    /// no reserved space, and the tags are not stable across versions —
    /// `format_version` is bumped whenever this enum changes.
    ///
    /// Ranges below are written `lo..hi` and are inclusive at both ends;
    /// `hi == unbounded` means no upper bound.
    pub const Tag = enum(u8) {
        // == Boolean layer ====================================================

        /// Boolean constant. `lhs` is 0 (false) or 1 (true), not an index.
        bool_const,
        /// An opaque front-end term, read as a sampled boolean. `lhs` = `Atom`,
        /// not an index.
        atom,
        /// Logical negation. `lhs` = operand boolean node.
        bnot,
        /// Logical conjunction. `lhs`, `rhs` = boolean nodes.
        band,
        /// Logical disjunction. `lhs`, `rhs` = boolean nodes.
        bor,

        // == Sequence layer ===================================================

        /// A boolean used as a length-1 sequence: matches a single cycle in
        /// which the boolean holds. `lhs` = boolean node.
        seq_boolean,
        /// Delayed concatenation: `lhs` matches, then after a delay of
        /// `lo..hi` cycles `right` matches. `lhs` = left sequence, `rhs` =
        /// `extra` index of `[right, lo, hi]`, where `right` is a sequence
        /// `Node.Index` — the one operand in the IR that lives outside `Data`.
        ///
        /// The delay is the gap *between* the two matches: 0 is an overlap, so
        /// the last cycle of the left match and the first of the right coincide,
        /// and 1 is plain concatenation.
        ///
        /// A delay measured from the start of the attempt instead — SVA's
        /// `##[lo:hi] b` with nothing to its left — is this operator with a
        /// one-cycle `true` to its left, exactly and with no adjustment to the
        /// bounds. It is not a second operator.
        seq_delay,
        /// Sequence disjunction: matches if either side matches. `lhs`, `rhs` =
        /// sequence nodes.
        seq_or,
        /// Both sides match from the same start cycle, and the match ends at
        /// the *later* of the two ends: a side that finishes early waits for
        /// the other rather than constraining it. `lhs`, `rhs` = sequence
        /// nodes. `seq_intersect` is this operator plus the requirement that
        /// the two ends coincide.
        seq_and,
        /// Length-matching intersection: both sides match from the same start
        /// cycle *and* the same end cycle. `lhs`, `rhs` = sequence nodes.
        seq_intersect,
        /// A boolean that must hold in every cycle of a match of a sequence.
        /// `lhs` = boolean node — the one sequence-layer tag whose operands are
        /// not all sequences — and `rhs` = sequence node.
        ///
        /// The LRM writes this as `(exp[*0:$]) intersect seq`, which this IR
        /// cannot build: the repetition admits an empty match. Because every
        /// sequence here matches at least one cycle, `(exp[*1:$]) intersect
        /// seq` is the same language, and it is what the operator means.
        seq_throughout,
        /// Only the *shortest* match of a sequence from each start cycle: as
        /// soon as one match completes, every longer match of the same attempt
        /// is suppressed. `lhs` = the sequence node.
        ///
        /// This is the one sequence-layer operator that is not a language
        /// operation on its operand's matches taken one at a time — whether a
        /// match survives depends on whether a *shorter* one exists over the
        /// same cycles, so lowering it determinises the operand rather than
        /// composing fragments. See `Glushkov.lowerFirstMatch`.
        seq_first_match,
        /// Consecutive repetition: `lo..hi` back-to-back matches of the
        /// operand, `lo >= 1`. `lhs` = the repeated sequence, `rhs` = `extra`
        /// index of `[lo, hi]`.
        seq_repeat_consec,
        /// Goto repetition: `lo..hi` matches of the operand, not necessarily
        /// back to back, with the match ending on the last one. Payload as
        /// `seq_repeat_consec`. The operand must be a `seq_boolean`: the gaps
        /// between occurrences are cycles in which the operand does *not*
        /// hold, which needs its negation, and "not necessarily back to back"
        /// has no reading for an operand whose matches can overlap.
        seq_repeat_goto,
        /// Non-consecutive repetition: as goto, except the match may extend
        /// past the last occurrence. Payload as `seq_repeat_consec`, and the
        /// operand is a `seq_boolean` for the same reason.
        seq_repeat_nonconsec,

        /// A sequence with local variable assignments: `(sequence, assignments)`
        /// from LRM 16.10. Assignments run at the end of every nonempty match.
        /// An empty assignment list returns the sequence unchanged; a non-empty
        /// list requires the sequence to not admit empty matches (libsva
        /// guarantees this for every sequence it can build). `lhs` = sequence
        /// node (backward-pointing, asserted), `rhs` = `extra` offset holding
        /// `[count, slot0, handle0, slot1, handle1, ...]`, where each `slot`
        /// is dense from zero within the current IR's locals table.
        seq_match,

        // == Property layer ===================================================

        /// A sequence promoted to a weak property: it holds when the sequence
        /// matches, fails when the sequence can no longer match, and holds if
        /// the trace ends while a match is still possible. `lhs` = sequence
        /// node.
        prop_seq_weak,
        /// A sequence promoted to a strong property: as `prop_seq_weak`,
        /// except that a trace ending while the match is still pending fails.
        /// `lhs` = sequence node.
        prop_seq_strong,
        /// Overlapping implication: for every match of the antecedent
        /// sequence, the consequent property must hold starting at the match's
        /// last cycle. `lhs` = antecedent sequence, `rhs` = consequent
        /// property.
        prop_implies_overlap,
        /// Non-overlapping implication: as `prop_implies_overlap`, but the
        /// consequent starts at the cycle after the match. `lhs` = antecedent
        /// sequence, `rhs` = consequent property.
        prop_implies_nonoverlap,

        /// Property conjunction: both operands must hold, each on its own
        /// schedule. `lhs`, `rhs` = property nodes.
        ///
        /// Not the same shape as `seq_and` one layer down even though it
        /// lowers onto it: this is a conjunction of *verdicts*, and it is only
        /// `seq_and` because a sequence property holds exactly when its
        /// sequence has a match, so "both hold" is "both have a match" is a
        /// match of `seq_and`. Both operands must carry the same strength —
        /// a pending conjunction of a weak and a strong operand has no single
        /// end-of-trace disposition, and the lowering rejects it.
        prop_and,
        /// Property disjunction: either operand holding is enough. `lhs`,
        /// `rhs` = property nodes. Same strength requirement as `prop_and`.
        prop_or,
        /// The operand property, evaluated `lo..hi` cycles later than this
        /// node. `lhs` = property node, `rhs` = `extra` index of `[lo, hi]`.
        ///
        /// The range is *disjunctive*: the property need only hold at some one
        /// offset in it. So `nexttime[n] p` is `lo == hi == n`, and
        /// `eventually [m:n] p` is `lo = m, hi = n`. `always [m:n] p` is the
        /// conjunctive reading and is `prop_always`.
        ///
        /// `hi == unbounded` spells the unbounded `s_eventually p`. It leaves
        /// the obligation without a longest match, so the lowering tracks it
        /// with counters rather than with a ring of state vectors; the operand
        /// may span any number of cycles, because the range is existential.
        prop_delay,
        /// The operand property, holding at *every* offset in `lo..hi`. `lhs` =
        /// property node, `rhs` = `extra` index of `[lo, hi]`.
        ///
        /// The conjunctive counterpart of `prop_delay`; the two coincide when
        /// `lo == hi`. `hi == unbounded` spells the unbounded `always p`, which
        /// is counter-tracked for the same reason — but, being universal, it
        /// requires an operand matching in a single cycle. "`p` holds at every
        /// cycle" is a statement about overlapping attempts of `p`, and one
        /// counter cannot carry a set of them; the lowering says so.
        prop_always,
        /// Overlapping followed-by: *some* match of the antecedent sequence is
        /// followed by the consequent property holding from that match's last
        /// cycle. `lhs` = antecedent sequence, `rhs` = consequent property.
        ///
        /// The existential dual of `prop_implies_overlap`, and not a verdict
        /// inversion of it: where implication is vacuously true when the
        /// antecedent never matches, followed-by *fails* there, because it
        /// asserts a witness exists. LRM 16.12.9 defines it as
        /// `not (seq |-> not prop)`, and negating both quantifier and vacuity
        /// is what turns it into a plain concatenation — see `Glushkov`.
        prop_followed_by_overlap,
        /// Non-overlapping followed-by: as `prop_followed_by_overlap`, but the
        /// consequent starts at the cycle after the match. `lhs` = antecedent
        /// sequence, `rhs` = consequent property.
        prop_followed_by_nonoverlap,
        /// `accept_on`: the operand property is *discharged* — it holds — from
        /// the moment the condition holds, whatever the operand was going to
        /// do. `lhs` = condition boolean, `rhs` = property node.
        ///
        /// This is what distinguishes an abort from `Assertion.reset`: a reset
        /// *abandons* an attempt, which neither holds nor fails, while an
        /// accept resolves it in favour of holding.
        ///
        /// The condition is read from **live** values, like a reset and unlike
        /// every other boolean in the tree (LRM 16.15). The `sync_` form below
        /// is the sampled one. Which of the two a root wants is recorded on
        /// `Monitor.Abort` for a backend to honour; nothing in this IR can
        /// check it.
        prop_accept_on,
        /// `reject_on`: the operand property *fails* from the moment the
        /// condition holds. `lhs` = condition boolean, `rhs` = property node.
        /// Live-valued, as `prop_accept_on`.
        prop_reject_on,
        /// `sync_accept_on`: `prop_accept_on` with the condition read from
        /// **sampled** values. `lhs` = condition boolean, `rhs` = property.
        prop_sync_accept_on,
        /// `sync_reject_on`: `prop_reject_on` with the condition read from
        /// **sampled** values. `lhs` = condition boolean, `rhs` = property.
        prop_sync_reject_on,

        /// `p until q` — the weak, non-inclusive form. `lhs`, `rhs` = property
        /// nodes.
        ///
        /// LRM 16.12.10: `q` holds at some cycle `k` at or after this one, and
        /// `p` holds at every cycle strictly before `k`. Weak, so a trace that
        /// ends with `p` still holding and `q` not yet seen holds rather than
        /// fails.
        ///
        /// The four forms are separate tags rather than one tag carrying flags
        /// because the LRM makes them four operators, and the weak/strong
        /// distinction here is the *operator's* own: unlike every other property
        /// tag, whose end-of-trace disposition comes from its leaves, an until
        /// fixes it. See `Glushkov.propStrength`.
        prop_until,
        /// `p until_with q` — weak, inclusive. `lhs`, `rhs` = property nodes.
        ///
        /// As `prop_until`, but `p` must hold at cycle `k` as well. LRM 16.12.10
        /// gives the identity `p until_with q ≡ p until (q and p)`, which is
        /// what the lowering builds.
        prop_until_with,
        /// `p s_until q` — strong, non-inclusive. `lhs`, `rhs` = property nodes.
        ///
        /// `prop_until`'s obligation with the opposite end-of-trace disposition:
        /// `q` is *required* to arrive, so a trace that ends without it fails.
        prop_s_until,
        /// `p s_until_with q` — strong, inclusive. `lhs`, `rhs` = property
        /// nodes.
        prop_s_until_with,

        /// `not p` — LRM 16.12.2. `lhs` = property node.
        ///
        /// The operand's verdict, complemented in the three-valued lattice of
        /// `Monitor.Verdict`: this holds where `p` fails, fails where `p`
        /// holds, and is pending where `p` is pending. That third clause is not
        /// a convention — it is what makes `not` an involution, and the
        /// lowering pays a real price to honour it.
        ///
        /// Nothing to do with `bnot`, which is two-valued complement of a
        /// boolean *term* one cycle at a time. This complements a whole
        /// attempt's verdict, and the two are not interchangeable at any depth:
        /// `not (a ##1 b)` is not a boolean operation on anything.
        ///
        /// It also inverts the strength of everything below it, so a `not` over
        /// a weak operand is strong. See `Glushkov.lowerPropNot`.
        prop_not,
        /// `p implies q` — LRM 16.12.2, the *property* implication. `lhs`,
        /// `rhs` = property nodes.
        ///
        /// Exactly `not p or q`, and lowered as that. It is a separate tag so
        /// that a refusal names the operator the front-end's user wrote rather
        /// than the disjunction libsva reduced it to.
        ///
        /// Not to be confused with `prop_implies_overlap` and its non-
        /// overlapping twin, which are `|->` and `|=>`: those take a *sequence*
        /// antecedent, quantify over its matches, and contribute a trigger.
        /// This takes two properties and contributes nothing but a verdict.
        ///
        /// The reduction has a consequence worth stating here rather than
        /// leaving to be discovered: `not` inverts strength and `prop_or`
        /// requires its operands to agree, so this lowers only when its two
        /// operands' strengths *differ*. `Glushkov.Error.MixedStrength` carries
        /// the derivation and the lift.
        prop_implies,
        /// `p iff q` — LRM 16.12.2. `lhs`, `rhs` = property nodes.
        ///
        /// `(p implies q) and (q implies p)`, and lowered as that.
        ///
        /// Which never succeeds today, for a reason that is arithmetic rather
        /// than incidental: the left conjunct carries `q`'s strength and the
        /// right one carries `p`'s, and each direction already required those
        /// two to be opposite, so the outer conjunction always sees a mixed
        /// pair. The tag exists so the refusal can name `iff`; see
        /// `Glushkov.Error.MixedStrength` for what would lift it.
        prop_iff,

        /// Which of the three layers a tag belongs to. Operands are checked
        /// against this when a node is built.
        pub const Layer = enum(u8) { boolean, sequence, property };

        pub fn layer(tag: Tag) Layer {
            return switch (tag) {
                .bool_const, .atom, .bnot, .band, .bor => .boolean,
                .seq_boolean,
                .seq_delay,
                .seq_or,
                .seq_and,
                .seq_intersect,
                .seq_throughout,
                .seq_first_match,
                .seq_repeat_consec,
                .seq_repeat_goto,
                .seq_repeat_nonconsec,
                .seq_match,
                => .sequence,
                .prop_seq_weak,
                .prop_seq_strong,
                .prop_implies_overlap,
                .prop_implies_nonoverlap,
                .prop_and,
                .prop_or,
                .prop_delay,
                .prop_always,
                .prop_followed_by_overlap,
                .prop_followed_by_nonoverlap,
                .prop_accept_on,
                .prop_reject_on,
                .prop_sync_accept_on,
                .prop_sync_reject_on,
                .prop_until,
                .prop_until_with,
                .prop_s_until,
                .prop_s_until_with,
                .prop_not,
                .prop_implies,
                .prop_iff,
                => .property,
            };
        }
    };
};

comptime {
    // The encoding the hash and the on-disk format blit is byte-exact.
    std.debug.assert(@sizeOf(Node.Tag) == 1);
    std.debug.assert(@sizeOf(Node.Data) == 8);
    std.debug.assert(@sizeOf(Assertion.Kind) == 1);
    std.debug.assert(@sizeOf(Clock) == 4);

    // Tags are declared grouped by layer, boolean first. Reading a layer off a
    // tag is then a range check, and a tag added to the wrong group here is a
    // compile error rather than a subtly mis-typed tree.
    var prev: u8 = 0;
    for (@typeInfo(Node.Tag).@"enum".fields) |field| {
        const tag: Node.Tag = @enumFromInt(field.value);
        const cur = @intFromEnum(tag.layer());
        std.debug.assert(cur >= prev);
        prev = cur;
    }
}

/// Sentinel for an open upper bound in a delay or repetition range.
pub const unbounded: u32 = std.math.maxInt(u32);

// -- Assertions --------------------------------------------------------------

/// A top-level directive: a clocked property plus its disposition. One attempt
/// of `prop` is seeded per cycle; see the semantics section at the top of this
/// file.
pub const Assertion = struct {
    /// Caller-assigned identity (see `Id`).
    id: Id,
    /// The directive's disposition.
    kind: Kind,
    /// What delimits a cycle for this assertion, or `.none` if it is unclocked
    /// and steps with the engine. A runtime backend is stepped once per cycle
    /// by its caller and never reads this; a formal backend emits it into the
    /// model.
    clock: Clock,
    /// Reset condition — a boolean node. Whenever it holds, every in-flight
    /// attempt of this assertion is abandoned. An assertion that never resets
    /// uses a `bool_const` of false, rather than the field being absent.
    reset: Node.Index,
    /// The root property node, one attempt of which starts each cycle.
    prop: Node.Index,

    /// Index into `Ir.assertions`.
    pub const Index = enum(u32) { _ };

    /// Opaque per-IR identity handle the caller assigns; resolve it against the
    /// front-end's own symbol table. The IR stores no text.
    pub const Id = enum(u32) { _ };

    pub const Kind = enum(u8) {
        /// The property must hold on every attempt; a failing attempt is an
        /// error.
        assert,
        /// The property is taken as a constraint on the environment rather
        /// than checked.
        assume,
        /// Coverage: report each attempt that matches.
        cover,
        /// A restriction on the traces a formal backend explores; ignored by a
        /// runtime backend.
        restrict,
    };
};

// -- Index helpers -----------------------------------------------------------

pub const Extra = struct {
    /// Element offset into `Ir.extra`.
    pub const Index = enum(u32) { _ };
};

// -- Construction ------------------------------------------------------------
//
// Low-level builders the front-end sits on top of. `addNode` and `addExtra` are
// the unchecked escape hatch; the typed constructors below them assert layer
// and operand-order well-formedness as they go.

pub fn addNode(ir: *Ir, gpa: Allocator, node: Node) Allocator.Error!Node.Index {
    const index: u32 = @intCast(ir.nodes.len);
    try ir.nodes.append(gpa, node);
    return @enumFromInt(index);
}

/// Append `items` to the `extra` pool and return the start offset.
pub fn addExtra(ir: *Ir, gpa: Allocator, items: []const u32) Allocator.Error!Extra.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.appendSlice(gpa, items);
    return @enumFromInt(start);
}

pub fn addAssertion(ir: *Ir, gpa: Allocator, a: Assertion) Allocator.Error!Assertion.Index {
    ir.assertOperand(a.prop, .property);
    ir.assertOperand(a.reset, .boolean);

    const index: u32 = @intCast(ir.assertions.len);
    try ir.assertions.append(gpa, a);
    return @enumFromInt(index);
}

/// Assert that `node` is an already-built node of layer `want`. Because a node
/// is only appended once its operands exist, "already built" is exactly the
/// backward-edge invariant: the operand's index is below the index this node
/// will get. Compiled out when safety is off.
fn assertOperand(ir: *const Ir, node: Node.Index, want: Node.Tag.Layer) void {
    if (!std.debug.runtime_safety) return;
    std.debug.assert(@intFromEnum(node) < ir.nodes.len);
    std.debug.assert(ir.tagOf(node).layer() == want);
}

// -- Convenience node constructors: boolean layer ----------------------------

pub fn boolConst(ir: *Ir, gpa: Allocator, value: bool) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .bool_const, .data = .{ .lhs = @intFromBool(value) } });
}

/// Reference an opaque front-end term as a boolean, widening `atom_count` to
/// cover it.
pub fn atom(ir: *Ir, gpa: Allocator, a: Atom) Allocator.Error!Node.Index {
    const handle = @intFromEnum(a);
    std.debug.assert(handle != std.math.maxInt(u32));
    ir.atom_count = @max(ir.atom_count, handle + 1);
    return ir.addNode(gpa, .{ .tag = .atom, .data = .{ .lhs = handle } });
}

pub fn bNot(ir: *Ir, gpa: Allocator, operand: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(operand, .boolean);
    return ir.addNode(gpa, .{ .tag = .bnot, .data = .{ .lhs = @intFromEnum(operand) } });
}

/// `band` or `bor`. Xor, implication and friends are left to the front-end to
/// expand.
pub fn bBinary(ir: *Ir, gpa: Allocator, tag: Node.Tag, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    std.debug.assert(tag == .band or tag == .bor);
    ir.assertOperand(lhs, .boolean);
    ir.assertOperand(rhs, .boolean);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

// -- Convenience node constructors: sequence layer ---------------------------

/// Lift a boolean into a one-cycle sequence.
pub fn seqBoolean(ir: *Ir, gpa: Allocator, b: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(b, .boolean);
    return ir.addNode(gpa, .{ .tag = .seq_boolean, .data = .{ .lhs = @intFromEnum(b) } });
}

/// Delayed concatenation: `left`, then `right` after a delay of `lo..hi`
/// cycles. `hi == Ir.unbounded` leaves the upper bound open.
pub fn seqDelay(
    ir: *Ir,
    gpa: Allocator,
    left: Node.Index,
    lo: u32,
    hi: u32,
    right: Node.Index,
) Allocator.Error!Node.Index {
    ir.assertOperand(left, .sequence);
    ir.assertOperand(right, .sequence);
    std.debug.assert(lo <= hi);

    const start = try ir.addExtra(gpa, &.{ @intFromEnum(right), lo, hi });
    return ir.addNode(gpa, .{ .tag = .seq_delay, .data = .{
        .lhs = @intFromEnum(left),
        .rhs = @intFromEnum(start),
    } });
}

/// Sequence disjunction: matches if either side matches.
pub fn seqOr(ir: *Ir, gpa: Allocator, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(lhs, .sequence);
    ir.assertOperand(rhs, .sequence);
    return ir.addNode(gpa, .{ .tag = .seq_or, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

/// Both sides match from the same start cycle, and the match ends at the later
/// of the two ends: whichever side finishes first waits for the other.
pub fn seqAnd(ir: *Ir, gpa: Allocator, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(lhs, .sequence);
    ir.assertOperand(rhs, .sequence);
    return ir.addNode(gpa, .{ .tag = .seq_and, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

/// Length-matching intersection: both sides match over the same cycle span.
pub fn seqIntersect(ir: *Ir, gpa: Allocator, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(lhs, .sequence);
    ir.assertOperand(rhs, .sequence);
    return ir.addNode(gpa, .{ .tag = .seq_intersect, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

/// `exp` holds in every cycle a match of `seq` spans.
///
/// The left operand is a *boolean*, not a sequence — it is a condition on each
/// cycle of the match rather than something matching alongside it — and is the
/// only sequence-layer operand that is.
pub fn seqThroughout(ir: *Ir, gpa: Allocator, exp: Node.Index, seq: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(exp, .boolean);
    ir.assertOperand(seq, .sequence);
    return ir.addNode(gpa, .{ .tag = .seq_throughout, .data = .{
        .lhs = @intFromEnum(exp),
        .rhs = @intFromEnum(seq),
    } });
}

/// A match of `inner` lies somewhere inside a match of `outer`, and the match
/// ends where `outer` does.
///
/// Sugar, not a tag: this builds `(true ##[0:$] inner ##[0:$] true) intersect
/// outer` out of operators the IR already has. The LRM pads with `1[*0:$]`,
/// which this IR cannot write because that repetition admits an empty match,
/// but a `0..$` delay against a one-cycle `true` is the same padding without
/// one — delay 0 fuses that cycle into the neighbouring match, so the padding
/// can cost nothing, and delay `k` places the neighbour `k` cycles away. A tag
/// every consumer of the IR would carry forever buys nothing over that.
pub fn seqWithin(ir: *Ir, gpa: Allocator, inner: Node.Index, outer: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(inner, .sequence);
    ir.assertOperand(outer, .sequence);

    const cycle = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    const lead = try ir.seqDelay(gpa, cycle, 0, unbounded, inner);
    const padded = try ir.seqDelay(gpa, lead, 0, unbounded, cycle);
    return ir.seqIntersect(gpa, padded, outer);
}

/// Only the shortest match of `seq` from each start cycle; a longer match of
/// the same attempt is suppressed.
///
/// Unlike the operators around it this one cannot be spelled with the others:
/// keeping the shortest match is a set difference against the operand's own
/// longer matches, which is a complement, and no product of fragments is one.
pub fn seqFirstMatch(ir: *Ir, gpa: Allocator, seq: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(seq, .sequence);
    return ir.addNode(gpa, .{ .tag = .seq_first_match, .data = .{ .lhs = @intFromEnum(seq) } });
}

/// Repetition of `seq` `lo..hi` times, in the flavor named by `tag`
/// (`seq_repeat_consec`, `_goto` or `_nonconsec`). `hi == Ir.unbounded` leaves
/// the upper bound open.
///
/// `lo >= 1`: the IR has no empty match, so a zero count has to be expanded by
/// the front-end into a disjunction over the surrounding context. See the
/// empty-match section at the top of this file.
///
/// Goto and non-consecutive repetition additionally require `seq` to be a
/// `seq_boolean`, because they are defined in terms of the cycles in which the
/// operand does *not* hold.
pub fn seqRepeat(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    seq: Node.Index,
    lo: u32,
    hi: u32,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .seq_repeat_consec or tag == .seq_repeat_goto or tag == .seq_repeat_nonconsec);
    ir.assertOperand(seq, .sequence);
    // Guarded like `assertOperand`: the tag read is only in range once the
    // operand check above has passed, which it only does in a safe build.
    if (std.debug.runtime_safety and tag != .seq_repeat_consec) {
        std.debug.assert(ir.tagOf(seq) == .seq_boolean);
    }
    std.debug.assert(lo >= 1);
    std.debug.assert(lo <= hi);

    const start = try ir.addExtra(gpa, &.{ lo, hi });
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(seq),
        .rhs = @intFromEnum(start),
    } });
}

/// Local variable assignments on a sequence: `(sequence, assignments)` from
/// LRM 16.10. Assignments run at the end of every nonempty match. An empty
/// assignment list returns `seq` unchanged. A non-empty list requires `seq` to
/// not admit empty matches; libsva guarantees this for every sequence it builds.
/// `assigns_raw` is a flat array of `[count, slot0, handle0, slot1, handle1, ...]`,
/// where slots are dense from zero within the current IR's locals table.
pub fn seqMatch(
    ir: *Ir,
    gpa: Allocator,
    seq: Node.Index,
    assigns_raw: []const u32,
) Allocator.Error!Node.Index {
    ir.assertOperand(seq, .sequence);
    std.debug.assert(assigns_raw.len % 2 == 0); // pairs of (slot, handle)
    const count = assigns_raw.len / 2;

    if (std.debug.runtime_safety) {
        for (0..count) |i| {
            const slot = assigns_raw[i * 2];
            std.debug.assert(slot < ir.localCount());
        }
    }

    // Empty: return the sequence unchanged.
    if (count == 0) return seq;

    // Build [count, slot0, handle0, slot1, handle1, ...] into extra.
    var payload: std.ArrayListUnmanaged(u32) = .empty;
    defer payload.deinit(gpa);
    try payload.append(gpa, @intCast(count));
    try payload.appendSlice(gpa, assigns_raw);
    const start = try ir.addExtra(gpa, payload.items);
    return ir.addNode(gpa, .{ .tag = .seq_match, .data = .{
        .lhs = @intFromEnum(seq),
        .rhs = @intFromEnum(start),
    } });
}

/// One assignment of a `seq_match` node: the slot written and the opaque handle
/// the front-end evaluates for its value.
pub const Assign = struct { slot: u32, handle: u32 };

/// The assignment list of a `seq_match` node, in source order. LRM 16.10 performs
/// them in that order at the end of every nonempty match of the operand, so the
/// order is semantic: a later item may read an earlier one's write.
pub fn matchOf(ir: *const Ir, node: Node.Index, buf: *std.ArrayListUnmanaged(Assign), gpa: Allocator) Allocator.Error!void {
    std.debug.assert(ir.tagOf(node) == .seq_match);
    const start = ir.dataOf(node).rhs;
    const count = ir.extra.items[start];
    buf.clearRetainingCapacity();
    try buf.ensureTotalCapacity(gpa, count);
    for (0..count) |i| {
        buf.appendAssumeCapacity(.{
            .slot = ir.extra.items[start + 1 + i * 2],
            .handle = ir.extra.items[start + 2 + i * 2],
        });
    }
}

// -- Convenience node constructors: property layer ---------------------------

/// Promote a sequence to a property, weak or strong per `tag`
/// (`prop_seq_weak` or `prop_seq_strong`). Both hold on a match and fail once
/// no match remains possible; they differ only at the end of a finite trace,
/// where a weak property forgives a still-pending match and a strong one fails
/// it.
pub fn propSeq(ir: *Ir, gpa: Allocator, tag: Node.Tag, seq: Node.Index) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_seq_weak or tag == .prop_seq_strong);
    ir.assertOperand(seq, .sequence);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(seq) } });
}

/// Implication, overlapping or not per `tag` (`prop_implies_overlap` or
/// `prop_implies_nonoverlap`): every match of `antecedent` obliges
/// `consequent`, starting at the match's last cycle or the one after it.
pub fn propImplies(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    antecedent: Node.Index,
    consequent: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_implies_overlap or tag == .prop_implies_nonoverlap);
    ir.assertOperand(antecedent, .sequence);
    ir.assertOperand(consequent, .property);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(antecedent),
        .rhs = @intFromEnum(consequent),
    } });
}

/// `not p`: the operand's verdict, complemented in `Monitor.Verdict`.
///
/// Holds where `prop` fails, fails where `prop` holds, pending where `prop` is
/// pending — and inverts the strength `prop` contributes, since a pending
/// attempt is still pending under negation and only its end-of-trace
/// disposition turns over.
///
/// Not `bnot`. That complements a boolean term within one cycle; this
/// complements a whole attempt's verdict.
pub fn propNot(ir: *Ir, gpa: Allocator, prop: Node.Index) Allocator.Error!Node.Index {
    ir.assertOperand(prop, .property);
    return ir.addNode(gpa, .{ .tag = .prop_not, .data = .{ .lhs = @intFromEnum(prop) } });
}

/// A binary property junction per `tag`: `prop_and`, `prop_or`, `prop_implies`
/// or `prop_iff`.
///
/// The latter two are reductions onto the former — `p implies q` is
/// `not p or q`, and `p iff q` is the conjunction of the two directions — and
/// are tags of their own only so that a refusal names the operator as written.
///
/// Every one of the four constrains the strengths its operands end up carrying,
/// and none of those constraints is checkable here: strength is a property of a
/// subtree, not of an operand's own tag, so the lowering is what reports it.
pub fn propJunction(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    lhs: Node.Index,
    rhs: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_and or tag == .prop_or or
        tag == .prop_implies or tag == .prop_iff);
    ir.assertOperand(lhs, .property);
    ir.assertOperand(rhs, .property);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(lhs),
        .rhs = @intFromEnum(rhs),
    } });
}

/// A property shifted by a range of cycles, per `tag`.
///
///   - `prop_delay` reads `lo..hi` disjunctively: `prop` need hold at only one
///     offset in it. `nexttime[n]` is `lo == hi == n`; `eventually [m:n]` is
///     the wider range.
///   - `prop_always` reads it conjunctively: `prop` must hold at every offset.
///
/// `hi == unbounded` is accepted here and selects the counter-tracked
/// discipline in the lowering, which is also where the one restriction it
/// carries — `prop_always` wants a single-cycle operand — is diagnosed.
pub fn propRange(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    lo: u32,
    hi: u32,
    prop: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_delay or tag == .prop_always);
    ir.assertOperand(prop, .property);
    std.debug.assert(lo <= hi);

    const start = try ir.addExtra(gpa, &.{ lo, hi });
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(prop),
        .rhs = @intFromEnum(start),
    } });
}

/// Followed-by, overlapping or not per `tag` (`prop_followed_by_overlap` or
/// `prop_followed_by_nonoverlap`): *some* match of `antecedent` is followed by
/// `consequent` holding, from that match's last cycle or the one after it.
///
/// The existential dual of `propImplies`. An antecedent that never matches
/// makes an implication hold vacuously and makes this fail.
pub fn propFollowedBy(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    antecedent: Node.Index,
    consequent: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_followed_by_overlap or tag == .prop_followed_by_nonoverlap);
    ir.assertOperand(antecedent, .sequence);
    ir.assertOperand(consequent, .property);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(antecedent),
        .rhs = @intFromEnum(consequent),
    } });
}

/// An abort scoped to `prop`: `accept_on`, `reject_on`, or either one's `sync_`
/// form, per `tag`. While `prop` is being evaluated, `cond` holding resolves it
/// immediately — as holding for an accept, as failing for a reject.
///
/// The `sync_` tags read `cond` from sampled values and the plain ones from
/// live values, exactly as `Assertion.reset` does. Supplying a term that
/// matches the tag is the front-end's job; see `Monitor.Abort.sampled`.
pub fn propAbort(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    cond: Node.Index,
    prop: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_accept_on or tag == .prop_reject_on or
        tag == .prop_sync_accept_on or tag == .prop_sync_reject_on);
    ir.assertOperand(cond, .boolean);
    ir.assertOperand(prop, .property);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(cond),
        .rhs = @intFromEnum(prop),
    } });
}

/// One of the four `until` forms per `tag`: `lhs` must hold up to the cycle
/// `rhs` holds in, inclusively for the `_with` tags, and the `s_` tags require
/// `rhs` to arrive.
///
/// Unlike `propJunction`, the operands' own strengths are not required to
/// agree and are not consulted: the tag carries the end-of-trace disposition
/// (LRM 16.12.10 spells the four operators out precisely to say which), so
/// there is no ambiguity for a mixed pair to create.
///
/// The lowering restricts `lhs` to a property matching in a single cycle. See
/// `Glushkov.lowerUntil` for why, and for the diagnostic that says so.
pub fn propUntil(
    ir: *Ir,
    gpa: Allocator,
    tag: Node.Tag,
    lhs: Node.Index,
    rhs: Node.Index,
) Allocator.Error!Node.Index {
    std.debug.assert(tag == .prop_until or tag == .prop_until_with or
        tag == .prop_s_until or tag == .prop_s_until_with);
    ir.assertOperand(lhs, .property);
    ir.assertOperand(rhs, .property);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{
        .lhs = @intFromEnum(lhs),
        .rhs = @intFromEnum(rhs),
    } });
}

// -- Accessors ---------------------------------------------------------------

pub fn tagOf(ir: *const Ir, node: Node.Index) Node.Tag {
    return ir.nodes.items(.tag)[@intFromEnum(node)];
}

pub fn dataOf(ir: *const Ir, node: Node.Index) Node.Data {
    return ir.nodes.items(.data)[@intFromEnum(node)];
}

/// Most operands a node can have.
pub const max_operands = 2;

/// Write the nodes `node` refers to into `buf` and return the filled prefix —
/// the exhaustive walk any def-use pass must use, rather than reading `lhs` and
/// `rhs` itself. It knows the two shapes that break a naive walk: `bool_const`
/// and `atom` carry a payload in `lhs` and have no operands at all, and
/// `seq_delay` holds its right operand in `extra`, not in `rhs`.
///
/// Every index it returns names a real node: nothing in the tree is optional.
pub fn operandsOf(ir: *const Ir, node: Node.Index, buf: *[max_operands]Node.Index) []Node.Index {
    const d = ir.dataOf(node);
    switch (ir.tagOf(node)) {
        // Payload in `lhs`, no edges.
        .bool_const, .atom => return buf[0..0],
        // One operand in `lhs`; `rhs` is an `extra` offset for the repetitions
        // and for the ranged property operators.
        .bnot,
        .seq_boolean,
        .seq_first_match,
        .seq_repeat_consec,
        .seq_repeat_goto,
        .seq_repeat_nonconsec,
        .seq_match,
        .prop_seq_weak,
        .prop_seq_strong,
        .prop_delay,
        .prop_always,
        .prop_not,
        => {
            buf[0] = @enumFromInt(d.lhs);
            return buf[0..1];
        },
        // Both operands in `data`.
        .band,
        .bor,
        .seq_or,
        .seq_and,
        .seq_intersect,
        .seq_throughout,
        .prop_implies_overlap,
        .prop_implies_nonoverlap,
        .prop_and,
        .prop_or,
        .prop_followed_by_overlap,
        .prop_followed_by_nonoverlap,
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
        .prop_until,
        .prop_until_with,
        .prop_s_until,
        .prop_s_until_with,
        .prop_implies,
        .prop_iff,
        => {
            buf[0] = @enumFromInt(d.lhs);
            buf[1] = @enumFromInt(d.rhs);
            return buf[0..2];
        },
        // Left operand in `lhs`, right operand in `extra`.
        .seq_delay => {
            buf[0] = @enumFromInt(d.lhs);
            buf[1] = @enumFromInt(ir.extra.items[d.rhs]);
            return buf[0..2];
        },
    }
}

/// The `Atom` an `atom` node names.
pub fn atomOf(ir: *const Ir, node: Node.Index) Atom {
    std.debug.assert(ir.tagOf(node) == .atom);
    return @enumFromInt(ir.dataOf(node).lhs);
}

/// Decoded payload of a `seq_delay` node.
pub const Delay = struct {
    /// Left sequence.
    left: Node.Index,
    /// Right sequence.
    right: Node.Index,
    /// Minimum delay in cycles; `0` means the two matches overlap by a cycle.
    lo: u32,
    /// Maximum delay in cycles, or `unbounded`.
    hi: u32,
};

pub fn delayOf(ir: *const Ir, node: Node.Index) Delay {
    std.debug.assert(ir.tagOf(node) == .seq_delay);
    const d = ir.dataOf(node);
    const p = ir.extra.items[d.rhs..][0..3];
    return .{ .left = @enumFromInt(d.lhs), .right = @enumFromInt(p[0]), .lo = p[1], .hi = p[2] };
}

/// Decoded payload of a `seq_repeat_*` node. The flavor is the node's tag.
pub const Repeat = struct {
    /// The repeated sequence.
    seq: Node.Index,
    /// Minimum repetition count, at least 1.
    lo: u32,
    /// Maximum repetition count, or `unbounded`.
    hi: u32,
};

pub fn repeatOf(ir: *const Ir, node: Node.Index) Repeat {
    const tag = ir.tagOf(node);
    std.debug.assert(tag == .seq_repeat_consec or tag == .seq_repeat_goto or tag == .seq_repeat_nonconsec);
    const d = ir.dataOf(node);
    const p = ir.extra.items[d.rhs..][0..2];
    return .{ .seq = @enumFromInt(d.lhs), .lo = p[0], .hi = p[1] };
}

/// Decoded payload of a `prop_delay` or `prop_always` node. Which of the two
/// reads the range disjunctively is the node's tag.
pub const PropRange = struct {
    /// The shifted property.
    prop: Node.Index,
    /// Minimum offset in cycles. `0` is the property at this very cycle.
    lo: u32,
    /// Maximum offset in cycles, or `unbounded`.
    hi: u32,
};

pub fn propRangeOf(ir: *const Ir, node: Node.Index) PropRange {
    const tag = ir.tagOf(node);
    std.debug.assert(tag == .prop_delay or tag == .prop_always);
    const d = ir.dataOf(node);
    const p = ir.extra.items[d.rhs..][0..2];
    return .{ .prop = @enumFromInt(d.lhs), .lo = p[0], .hi = p[1] };
}

// -- Hashing -----------------------------------------------------------------

/// A Blake3 content digest — the full 256-bit output, so callers may treat it
/// as collision-free.
pub const Digest = [Blake3.digest_length]u8;

/// Content hash over every flat array. Structurally identical IRs hash equal,
/// so the pipeline can key a compiled-monitor cache on this. Section lengths
/// are folded in so that shifting a boundary between two arrays cannot collide.
pub fn hash(ir: *const Ir) Digest {
    var st = Blake3.init(.{});

    var scratch: [4]u8 = undefined;
    std.mem.writeInt(u32, &scratch, ir.atom_count, .little);
    st.update(&scratch);

    inline for (.{
        std.mem.sliceAsBytes(ir.nodes.items(.tag)),
        std.mem.sliceAsBytes(ir.nodes.items(.data)),
        std.mem.sliceAsBytes(ir.assertions.items(.id)),
        std.mem.sliceAsBytes(ir.assertions.items(.kind)),
        std.mem.sliceAsBytes(ir.assertions.items(.clock)),
        std.mem.sliceAsBytes(ir.assertions.items(.reset)),
        std.mem.sliceAsBytes(ir.assertions.items(.prop)),
        std.mem.sliceAsBytes(ir.extra.items),
        std.mem.sliceAsBytes(ir.locals.items),
        std.mem.sliceAsBytes(ir.atom_reads.items),
    }) |section| {
        std.mem.writeInt(u32, &scratch, @intCast(section.len), .little);
        st.update(&scratch);
        st.update(section);
    }

    var out: Digest = undefined;
    st.final(&out);
    return out;
}

// -- Serialization -----------------------------------------------------------
//
// On-disk form:
//
//     [magic: 4 bytes]
//     [format_version: u32]
//     [atom_count: u32]
//     [ per section: u32 element count, then the section's raw little-endian
//       bytes ]
//     [checksum: Blake3 digest over everything above]
//
// The checksum is verified *before* any length is trusted, so a corrupt or
// tampered cache file fails cleanly rather than driving an allocation off a
// garbage count. Endianness is not converted (a local build cache is
// same-endian).

/// Identifies a libsva cache blob.
pub const magic: [4]u8 = .{ 'S', 'V', 'A', 'B' };
/// On-disk format version. The IR reserves no space for future operators, so
/// this is bumped on any layout *or* `Node.Tag` change; readers reject a
/// mismatch and the caller recompiles from source.
pub const format_version: u32 = 11;

const checksum_len = Blake3.digest_length;
/// Bytes preceding the first payload word: `magic` + `format_version`.
const header_len = magic.len + @sizeOf(u32);

/// Serialize into a freshly allocated byte buffer owned by the caller.
pub fn serialize(ir: *const Ir, gpa: Allocator) Allocator.Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(gpa);

    try out.appendSlice(gpa, &magic);
    try writeU32(&out, gpa, format_version);
    try writeU32(&out, gpa, ir.atom_count);

    try writeU32(&out, gpa, @intCast(ir.nodes.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try writeU32(&out, gpa, @intCast(ir.assertions.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.id)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.kind)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.clock)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.reset)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.prop)));

    try writeU32(&out, gpa, @intCast(ir.extra.items.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.extra.items));

    try writeU32(&out, gpa, @intCast(ir.locals.items.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.locals.items));

    try writeU32(&out, gpa, @intCast(ir.atom_reads.items.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.atom_reads.items));

    var digest: Digest = undefined;
    Blake3.hash(out.items, &digest, .{});
    try out.appendSlice(gpa, &digest);

    return out.toOwnedSlice(gpa);
}

pub const DeserializeError = error{
    /// Buffer does not start with `magic`.
    BadMagic,
    /// `format_version` is not one this build understands.
    UnsupportedVersion,
    /// The trailing checksum does not match the payload.
    ChecksumMismatch,
    /// Buffer ends mid-record.
    Truncated,
} || Allocator.Error;

/// Reconstruct an `Ir` from bytes produced by `serialize`. The result owns its
/// storage and must be `deinit`ed.
pub fn deserialize(gpa: Allocator, bytes: []const u8) DeserializeError!Ir {
    if (bytes.len < header_len + checksum_len) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..magic.len], &magic)) return error.BadMagic;
    if (std.mem.readInt(u32, bytes[4..8], .little) != format_version) return error.UnsupportedVersion;

    const payload = bytes[0 .. bytes.len - checksum_len];
    const stored = bytes[bytes.len - checksum_len ..];
    var digest: Digest = undefined;
    Blake3.hash(payload, &digest, .{});
    // Integrity check only (local cache); a constant-time compare is unneeded.
    if (!std.mem.eql(u8, &digest, stored)) return error.ChecksumMismatch;

    var ir: Ir = .init();
    errdefer ir.deinit(gpa);

    // Counts are trusted only now, after the checksum has been verified.
    var cur: Cursor = .{ .bytes = payload, .pos = header_len };

    ir.atom_count = try cur.readU32();

    try ir.nodes.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try ir.assertions.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.id)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.kind)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.clock)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.reset)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.prop)));

    try ir.extra.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.extra.items));

    try ir.locals.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.locals.items));

    try ir.atom_reads.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.atom_reads.items));

    // Recompute the cached local_words from locals.len.
    ir.local_words = @intCast((ir.locals.items.len + 63) / 64);

    return ir;
}

fn writeU32(out: *std.ArrayListUnmanaged(u8), gpa: Allocator, v: u32) Allocator.Error!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try out.appendSlice(gpa, &buf);
}

// -- Local variable management -----------------------------------------------

/// Declare one local variable slot. Returns its slot number (dense from zero in
/// declaration order). LRM 16.10: each slot begins uninitialized, and the
/// optional initializer runs when the attempt is created. `type_` and `init` are
/// opaque to libsva; the front-end assigns both. An attempt's slots are
/// independent: an initializer may read a lower-numbered slot already
/// initialized this attempt.
pub fn localDeclare(ir: *Ir, gpa: Allocator, type_: u32, init_: u32) Allocator.Error!u32 {
    try ir.locals.append(gpa, .{ .type = type_, .init = init_ });
    ir.local_words = @intCast((ir.locals.items.len + 63) / 64);
    return @intCast(ir.locals.items.len - 1);
}

/// Count of local variable slots declared in this IR, dense from zero.
pub fn localCount(ir: *const Ir) u32 {
    return @intCast(ir.locals.items.len);
}

/// Words in one local slot's read mask: `localCount()` rounded up to 64. Dense
/// atoms use this as their stride into `atom_reads.items`.
pub fn localWords(ir: *const Ir) u32 {
    return ir.local_words;
}

/// Record which local variable slots one atom's expression reads. `slots` must
/// be exactly `localWords()` words, least-significant bit of word zero being
/// slot zero. Every slot mentioned must exist (`< localCount()`). A backend
/// must declare all locals *before* recording any atom's reads; libsva does not
/// re-stride existing masks if locals grow later.
pub fn atomReads(ir: *Ir, gpa: Allocator, atom_id: Atom, slots: []const u64) Allocator.Error!void {
    const a = @intFromEnum(atom_id);
    std.debug.assert(a < ir.atom_count);
    const lw = ir.localWords();
    std.debug.assert(slots.len == lw);

    if (lw == 0) return; // no locals declared, nothing to store

    // Grow `atom_reads` to cover this atom if needed: `atom_count * local_words` u64s.
    const current_capacity = ir.atom_reads.items.len / lw;
    if (current_capacity <= a) {
        try ir.atom_reads.resize(gpa, (a + 1) * lw);
    }

    const start = a * lw;
    @memcpy(ir.atom_reads.items[start..][0..lw], slots);
}

/// Read mask for one atom: which local slots it reads. Returns null if never
/// recorded for this atom (implicitly all-zero). Slice is `localWords()` words.
pub fn atomReadsOf(ir: *const Ir, atom_id: Atom) ?[]const u64 {
    const a = @intFromEnum(atom_id);
    const lw = ir.localWords();
    if (a >= ir.atom_count or lw == 0) return null;
    const offset: usize = @intCast(a * lw);
    if (offset >= ir.atom_reads.items.len) return null;
    const len: usize = @intCast(lw);
    if (offset + len > ir.atom_reads.items.len) return null;
    return ir.atom_reads.items[offset..][0..len];
}

const Cursor = struct {
    /// Buffer being read, checksum already stripped and verified.
    bytes: []const u8,
    /// Read offset into `bytes`.
    pos: usize = 0,

    fn take(c: *Cursor, n: usize) error{Truncated}![]const u8 {
        if (c.pos + n > c.bytes.len) return error.Truncated;
        defer c.pos += n;
        return c.bytes[c.pos..][0..n];
    }

    fn readU32(c: *Cursor) error{Truncated}!u32 {
        const s = try c.take(4);
        return std.mem.readInt(u32, s[0..4], .little);
    }

    /// Copy the next `dst.len` bytes into `dst`.
    fn copyInto(c: *Cursor, dst: []u8) error{Truncated}!void {
        @memcpy(dst, try c.take(dst.len));
    }
};

// -- Tests -------------------------------------------------------------------

test "build, hash, and round-trip an assertion" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // Every attempt: if `a` holds and `b` holds the cycle after, then `c` must
    // hold in that same cycle.
    const a = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(0)));
    const b = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(1)));
    const ante = try ir.seqDelay(gpa, a, 1, 1, b);
    const c = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(2)));
    const cons = try ir.propSeq(gpa, .prop_seq_weak, c);
    const prop = try ir.propImplies(gpa, .prop_implies_overlap, ante, cons);

    _ = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(7),
        .kind = .assert,
        .clock = @enumFromInt(4),
        .reset = try ir.boolConst(gpa, false),
        .prop = prop,
    });

    // Serialize -> deserialize preserves structure and content hash.
    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    var ir2 = try Ir.deserialize(gpa, bytes);
    defer ir2.deinit(gpa);

    try std.testing.expectEqual(ir.nodes.len, ir2.nodes.len);
    try std.testing.expectEqual(ir.assertions.len, ir2.assertions.len);
    try std.testing.expectEqual(@as(u32, 3), ir2.atom_count);

    const h1 = ir.hash();
    const h2 = ir2.hash();
    try std.testing.expectEqualSlices(u8, &h1, &h2);

    // Opaque handles round-trip verbatim.
    try std.testing.expectEqual(@as(Assertion.Id, @enumFromInt(7)), ir2.assertions.items(.id)[0]);
    try std.testing.expectEqual(@as(Clock, @enumFromInt(4)), ir2.assertions.items(.clock)[0]);
    const a_atom: Node.Index = @enumFromInt(ir2.dataOf(a).lhs);
    try std.testing.expectEqual(@as(Atom, @enumFromInt(0)), ir2.atomOf(a_atom));

    // The decoded delay payload survives.
    const d = ir2.delayOf(ante);
    try std.testing.expectEqual(@as(u32, 1), d.lo);
    try std.testing.expectEqual(@as(u32, 1), d.hi);
    try std.testing.expectEqual(b, d.right);
}

test "atom_count bounds the atom space and is part of the content" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    try std.testing.expectEqual(@as(u32, 0), ir.atom_count);
    _ = try ir.atom(gpa, @enumFromInt(5));
    try std.testing.expectEqual(@as(u32, 6), ir.atom_count);
    // A lower handle does not shrink the bound.
    _ = try ir.atom(gpa, @enumFromInt(2));
    try std.testing.expectEqual(@as(u32, 6), ir.atom_count);

    // Same tree over a denser atom space is a different IR.
    var dense: Ir = .init();
    defer dense.deinit(gpa);
    _ = try dense.atom(gpa, @enumFromInt(1));
    _ = try dense.atom(gpa, @enumFromInt(2));
    try std.testing.expectEqual(@as(u32, 3), dense.atom_count);
    try std.testing.expect(!std.mem.eql(u8, &ir.hash(), &dense.hash()));
}

test "operandsOf is the exhaustive walk" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // Build one node of every shape, including an atom whose handle collides
    // with a valid node index, and a delay off a `true` — which is how a delay
    // measured from the start of an attempt is written now that no edge is
    // optional.
    const x = try ir.atom(gpa, @enumFromInt(0));
    const y = try ir.atom(gpa, @enumFromInt(1));
    const both = try ir.bBinary(gpa, .band, x, try ir.bNot(gpa, y));
    const sx = try ir.seqBoolean(gpa, both);
    const sy = try ir.seqBoolean(gpa, y);
    const rep = try ir.seqRepeat(gpa, .seq_repeat_goto, sy, 1, unbounded);
    const always = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    const lead = try ir.seqDelay(gpa, always, 2, 3, rep);
    const cat = try ir.seqDelay(gpa, sx, 1, 1, lead);
    const alt = try ir.seqOr(gpa, cat, try ir.seqIntersect(gpa, sx, cat));
    const soonest = try ir.seqFirstMatch(gpa, alt);
    const prop = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        soonest,
        try ir.propSeq(gpa, .prop_seq_strong, sx),
    );
    // The property layer above it. `propRange` is the second shape that keeps an
    // operand outside `data`: its range lives in `extra`, so `rhs` is an offset
    // and a naive walk would follow it as an edge.
    const shifted = try ir.propRange(gpa, .prop_delay, 1, 2, prop);
    const bounded = try ir.propRange(gpa, .prop_always, 0, 3, shifted);
    const pair = try ir.propJunction(gpa, .prop_or, bounded, shifted);
    const witnessed = try ir.propFollowedBy(gpa, .prop_followed_by_overlap, sx, pair);
    const guarded = try ir.propAbort(gpa, .prop_sync_reject_on, y, witnessed);

    // The whole graph is reachable from the root, every edge points backwards,
    // and no leaf payload is mistaken for an edge.
    var buf: [max_operands]Node.Index = undefined;
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(gpa, ir.nodes.len);
    defer seen.deinit(gpa);

    var stack: std.ArrayListUnmanaged(Node.Index) = .empty;
    defer stack.deinit(gpa);
    try stack.append(gpa, guarded);
    while (stack.pop()) |node| {
        seen.set(@intFromEnum(node));
        for (ir.operandsOf(node, &buf)) |operand| {
            try std.testing.expect(@intFromEnum(operand) < @intFromEnum(node));
            try stack.append(gpa, operand);
        }
    }
    try std.testing.expectEqual(ir.nodes.len, seen.count());

    // The two shapes a naive {lhs, rhs} walk gets wrong.
    try std.testing.expectEqual(@as(usize, 0), ir.operandsOf(y, &buf).len);
    try std.testing.expectEqual(@as(usize, 0), ir.operandsOf(try ir.boolConst(gpa, true), &buf).len);
    try std.testing.expectEqualSlices(Node.Index, &.{ always, rep }, ir.operandsOf(lead, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{ sx, lead }, ir.operandsOf(cat, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{alt}, ir.operandsOf(soonest, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{prop}, ir.operandsOf(shifted, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{shifted}, ir.operandsOf(bounded, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{ sx, pair }, ir.operandsOf(witnessed, &buf));
    try std.testing.expectEqualSlices(Node.Index, &.{ y, witnessed }, ir.operandsOf(guarded, &buf));

    // The range survives the round trip through `extra`.
    try std.testing.expectEqual(PropRange{ .prop = prop, .lo = 1, .hi = 2 }, ir.propRangeOf(shifted));
    try std.testing.expectEqual(
        PropRange{ .prop = shifted, .lo = 0, .hi = 3 },
        ir.propRangeOf(bounded),
    );
}

test "sequence properties carry strength" {
    const gpa = std.testing.allocator;

    var weak: Ir = .init();
    defer weak.deinit(gpa);
    var strong: Ir = .init();
    defer strong.deinit(gpa);

    for ([_]struct { *Ir, Node.Tag }{ .{ &weak, .prop_seq_weak }, .{ &strong, .prop_seq_strong } }) |pair| {
        const ir, const tag = pair;
        const seq = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(0)));
        const prop = try ir.propSeq(gpa, tag, seq);
        try std.testing.expectEqual(tag, ir.tagOf(prop));
        try std.testing.expectEqual(Node.Tag.Layer.property, ir.tagOf(prop).layer());
    }

    // Strength is part of the content, so the two cache apart.
    try std.testing.expect(!std.mem.eql(u8, &weak.hash(), &strong.hash()));
}

test "repetition flavor lives in the tag" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    const seq = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(0)));
    const consec = try ir.seqRepeat(gpa, .seq_repeat_consec, seq, 1, unbounded);
    const goto = try ir.seqRepeat(gpa, .seq_repeat_goto, seq, 3, 3);

    try std.testing.expectEqual(Node.Tag.seq_repeat_consec, ir.tagOf(consec));
    try std.testing.expectEqual(Repeat{ .seq = seq, .lo = 1, .hi = unbounded }, ir.repeatOf(consec));
    try std.testing.expectEqual(Node.Tag.seq_repeat_goto, ir.tagOf(goto));
    try std.testing.expectEqual(Repeat{ .seq = seq, .lo = 3, .hi = 3 }, ir.repeatOf(goto));

    // Two flavors of the same range are distinct trees, so they hash apart.
    var other: Ir = .init();
    defer other.deinit(gpa);
    const oseq = try other.seqBoolean(gpa, try other.atom(gpa, @enumFromInt(0)));
    _ = try other.seqRepeat(gpa, .seq_repeat_consec, oseq, 1, unbounded);
    _ = try other.seqRepeat(gpa, .seq_repeat_nonconsec, oseq, 3, 3);

    try std.testing.expect(!std.mem.eql(u8, &ir.hash(), &other.hash()));
}

test "operands are layer-checked and point backwards" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    const b = try ir.atom(gpa, @enumFromInt(0));
    const seq = try ir.seqBoolean(gpa, b);
    const prop = try ir.propSeq(gpa, .prop_seq_strong, seq);

    try std.testing.expectEqual(Node.Tag.Layer.boolean, ir.tagOf(b).layer());
    try std.testing.expectEqual(Node.Tag.Layer.sequence, ir.tagOf(seq).layer());
    try std.testing.expectEqual(Node.Tag.Layer.property, ir.tagOf(prop).layer());

    // Every edge points at a lower index, so a front-to-back fold sees each
    // node after its operands.
    try std.testing.expect(@intFromEnum(b) < @intFromEnum(seq));
    try std.testing.expect(@intFromEnum(seq) < @intFromEnum(prop));

    // A well-formed root is accepted; the asserts fire on anything else.
    _ = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(0),
        .kind = .cover,
        .clock = .none,
        .reset = b,
        .prop = try ir.propImplies(gpa, .prop_implies_nonoverlap, seq, prop),
    });
}

test "cache format is versioned and checksum-protected" {
    const gpa = std.testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    _ = try ir.boolConst(gpa, true);

    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    try std.testing.expectEqualSlices(u8, &magic, bytes[0..magic.len]);
    try std.testing.expectEqual(format_version, std.mem.readInt(u32, bytes[4..8], .little));

    // A short buffer is rejected before anything is parsed.
    try std.testing.expectError(error.Truncated, Ir.deserialize(gpa, bytes[0..8]));

    // Flipping a payload byte trips the checksum.
    const tampered = try gpa.dupe(u8, bytes);
    defer gpa.free(tampered);
    tampered[header_len + 2] ^= 0xff;
    try std.testing.expectError(error.ChecksumMismatch, Ir.deserialize(gpa, tampered));

    // Wrong magic.
    var bad_magic = [_]u8{0} ** (header_len + checksum_len);
    bad_magic[0] = 'X';
    try std.testing.expectError(error.BadMagic, Ir.deserialize(gpa, &bad_magic));

    // Recognized magic but an unknown version.
    var bad_version = [_]u8{0} ** (header_len + checksum_len);
    @memcpy(bad_version[0..magic.len], &magic);
    std.mem.writeInt(u32, bad_version[4..8], format_version + 1, .little);
    try std.testing.expectError(error.UnsupportedVersion, Ir.deserialize(gpa, &bad_version));
}

test {
    std.testing.refAllDecls(@This());
}
