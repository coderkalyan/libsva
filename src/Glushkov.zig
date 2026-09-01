//! Lowers a sequence to a position automaton.
//!
//! # Method
//!
//! Glushkov's construction, generalised slightly. Rather than computing
//! `first`/`last`/`follow` as syntactic functions of an expression, this builds
//! a small algebra of **fragments** — homogeneous ε-free automata carrying
//! exactly those three sets — with one combinator per sequence tag. The base
//! cases are Glushkov's; `intersect` and `and` are synchronised products, and
//! the product of two homogeneous automata is homogeneous (the pair state is
//! entered under the conjunction of the two guards), so it composes as just
//! another fragment.
//!
//! The `Ir` guarantees every sequence matches at least one cycle, which is why
//! no fragment carries a `nullable` flag: the empty-match disjuncts that make
//! up most of the textbook equations are all identically absent.
//!
//! # Per-operator shape
//!
//! | tag | construction | states added |
//! |-----|--------------|--------------|
//! | `seq_boolean` | one state | 1 |
//! | `seq_or` | disjoint union | 0 |
//! | `seq_delay`, delay 0 | *fusion*: a state per `last × first` pair, guards conjoined | `|last| · |first|` |
//! | `seq_delay`, delay ≥ 1 | a chain of `true`-guarded filler states | `hi - 1`, or `lo - 1` when open |
//! | `seq_repeat_consec` | unrolled copies, looped on the last when open | `hi · |A|` |
//! | `seq_repeat_goto` | a `b` state and a `¬b` gap state per occurrence | `2 · hi` |
//! | `seq_repeat_nonconsec` | as goto, plus a trailing gap | `2 · hi + 1` |
//! | `seq_intersect` | reachable product | `|A| · |B|` |
//! | `seq_throughout` | a conjunction into every guard | 0 |
//! | `seq_and` | product, plus a copy of each operand to run on alone | `|A| · |B| + |A| + |B|` |
//! | `seq_first_match` | the operand determinised, cut off at its first match | `|subsets| · |cells|` |
//!
//! A delay range is *both* shapes at once when it spans zero: `a ##[0:2] b` is
//! the fusion states and the filler chain in one fragment, not a disjunction of
//! two copies.
//!
//! Every row of that table is a blow-up vector, and none of them is avoidable
//! without counter-augmented automata, which would cost the bit-set step. The
//! state count is capped instead; `Options.limit` is checked against the
//! projected size in 64-bit arithmetic *before* anything is allocated, and
//! `Diagnostic` names the node that tripped it.
//!
//! # Pruning
//!
//! Fusion and product both manufacture states that cannot participate in a
//! match — fusion leaves the right operand's entry states reachable from
//! nothing, and either can produce a syntactically contradictory guard, as can
//! a `throughout` whose boolean contradicts a guard it is conjoined into. A
//! pass before materialisation keeps only the states reachable from `first`,
//! co-reachable to `last`, and not guarded by an unsatisfiable cube.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Nfa = @import("Nfa.zig");
const Monitor = @import("Monitor.zig");

const Word = Nfa.Word;
const Guard = Nfa.Guard;
const Literal = Nfa.Literal;
const BitIterator = Nfa.BitIterator;

pub const Options = struct {
    /// Ceiling on the state count of any fragment, intermediate or final.
    limit: u32 = 4096,
    /// Ceiling on a monitor's `depth` — how many obligation attempts it may
    /// have to track at once.
    depth_limit: u32 = 256,
};

/// Why a lowering failed, for a front-end to report against source. Written on
/// every failure and never otherwise, so it needs no absent value.
pub const Diagnostic = struct {
    /// The node whose lowering failed.
    node: Ir.Node.Index,
    /// The size that tripped a limit: states for `StateLimitExceeded`, pending
    /// attempts for `DepthLimitExceeded`. Zero when no limit was involved.
    size: u64 = 0,
};

pub const Error = error{
    /// A fragment would have exceeded `Options.limit` states.
    StateLimitExceeded,
    /// Goto or non-consecutive repetition over an operand that is not a
    /// `seq_boolean`. The `Ir` constructors assert this too, but only in a safe
    /// build, so the lowering rejects it unconditionally.
    UnsupportedRepeatOperand,
    /// An obligation with no bounded longest match that the counted discipline
    /// cannot represent either. One shape, in two spellings.
    ///
    /// `always p` and the left operand of an `until` are *universal over
    /// cycles*: they say `p` holds at every cycle of a window. That is a
    /// statement about overlapping attempts of `p`, one per cycle, and a counted
    /// attempt is a single state — it cannot carry a set of sub-attempts that
    /// grows with the trace. When `p` matches in exactly one cycle each
    /// sub-attempt resolves inside its own cycle and the set never grows, which
    /// is the case that is lowered; anything longer is refused here, naming the
    /// operand.
    ///
    /// TODO: lift it with a second subset construction. Determinise `p`
    /// totally, then build the subset automaton over *that*, in which a dead
    /// sub-attempt is a state and so is visible in the subset; "some attempt of
    /// `p` has died" is then "the subset contains a dead state", which one token
    /// can carry. Cost is `2^|determinise(p)|`, capped like every other blow-up
    /// here.
    ///
    /// Existential operators have no such restriction: `s_eventually p` and the
    /// right operand of an `until` ask for *some* attempt of `p` to succeed, and
    /// the ordinary construction already tracks those however long they are.
    ObligationNotBounded,
    /// A trigger strictly below an unbounded operator: `always (a |=> b)`,
    /// `s_eventually (a |=> b)`, `(a |=> b) until c`.
    ///
    /// Tracking mode is a property of the whole monitor, and this shape wants
    /// both at once. The implication's own obligation is bounded, so it is
    /// ring-tracked; the operator above it is not, so it is counted. For a
    /// *bounded* range the shift lands on the antecedent instead — which is the
    /// only correct place for it, see `lowerPropShift` — and an unbounded range
    /// has no filler chain to grow there.
    ///
    /// A trigger *above* an unbounded operator is fine and is the point of the
    /// exercise: `req |=> s_eventually ack` lowers. A trigger is existential,
    /// one state vector for all of its attempts, so it needs no bound and
    /// composes with either discipline.
    ObligationModeMixed,
    /// A property operator was applied to an operand that raises obligations of
    /// its own — the classic case being an implication as a consequent, but a
    /// conjunction, a followed-by or a disjunctive shift over one is the same
    /// problem. A monitor has one trigger, tracked existentially, and there is
    /// nowhere to put a second.
    ///
    /// An abort is not in that list. It does not need a second trigger: it
    /// covers the one there is, uniformly over every attempt of it. See
    /// `UnsupportedAbortScope` for the one shape of that which is refused.
    UnsupportedConsequent,
    /// `prop_and`, `prop_or`, `prop_implies` or `prop_iff` over operands whose
    /// strengths do not meet that operator's requirement. The pair is pending
    /// until both sides resolve, and one `Monitor.Strength` per monitor cannot
    /// say what the end of a trace makes of a pending pair whose two sides
    /// would answer differently.
    ///
    /// For `prop_and` and `prop_or` the requirement is that the two agree. For
    /// `prop_implies` it is that they *differ*, which is the same rule seen
    /// through the reduction: `p implies q` is `not p or q`, `not` inverts
    /// strength, so the disjunction agrees exactly when the operands did not.
    /// `prop_iff` can never satisfy it — the two directions carry `q`'s and
    /// `p`'s strengths respectively, and each direction already forced those to
    /// be opposite, so the outer conjunction always sees a mixed pair.
    ///
    /// None of this is a conservative approximation. Take `p` and `q` both
    /// weak and `p implies q`. On a trace where `q` has definitively failed and
    /// `p` is still pending, the end of the trace makes `p` hold, so `q` was
    /// required and the implication fails. On a trace where both are pending it
    /// holds. Same monitor, same automaton, two live-state configurations,
    /// opposite dispositions — and one bit to carry them in.
    ///
    /// TODO: lift it with a per-state end-of-trace disposition instead of one
    /// `Monitor.Strength` per monitor: a mask of obligation states whose
    /// pending attempt holds, OR-combined over the live vector, with the
    /// conjunctive fold done at product-construction time where both operands'
    /// labels are still in hand. That deletes this error outright and lets
    /// `prop_iff` lower. It changes what `sva_monitor_strength` means, so it is
    /// a deliberate ABI decision rather than a local fix.
    MixedStrength,
    /// `prop_not` over an operand that contributes a trigger of its own — the
    /// complement of an implication, `not (s |-> p)`.
    ///
    /// A trigger is the *union* of every attempt of the antecedent, which is
    /// sound only because all the union is ever asked is whether some match
    /// ends this cycle. Complementing needs the opposite: `not (s |-> p)` is
    /// "some match of `s` is not followed by `p`", which is a statement about
    /// one attempt at a time, and the union has thrown exactly that away.
    ///
    /// TODO: lower it as `s #-# not p`. LRM 16.12.9 already defines followed-by
    /// as `not (seq |-> not prop)`, so this is that identity read right to left,
    /// and `lowerFollowedBy` is the construction it lands on — the antecedent
    /// joins the obligation instead of becoming a trigger, which is precisely
    /// what makes the per-attempt reading available again. What stops it today
    /// is that the rewrite has to happen on the `Ir` before `lowerProperty`
    /// descends, and nothing here rewrites the tree.
    UnsupportedComplement,
    /// An abort whose scope cannot be carried, in any of three ways.
    ///
    /// It sits below an operator that renumbers its operand's states, so the
    /// mask naming the abort's subtree cannot be carried through it. `prop_or`
    /// and a shift relocate an operand's states as a block and are fine; a
    /// product (`prop_and`, `prop_always`) and a fusion (`prop_followed_by_*`)
    /// are not. An accept under a product would not be a mask operation in any
    /// case: it has to record that one side of the pair is satisfied, which is a
    /// state remap rather than a set of states to clear.
    ///
    /// Or it is a `.branch`-scoped abort over an obligation that will be
    /// counter-tracked. Killing part of an attempt is a mask subtracted from its
    /// state vector, and a counted attempt has no vector: its single state
    /// stands for a whole subset of the operand's, and there is no state
    /// standing for that subset minus the mask. `.attempt` scope is carried
    /// through determinisation and is fine — see `Monitor.Abort.Scope`.
    ///
    /// Or it sits under a `prop_not`, whatever its scope. The mask moves through
    /// the subsets perfectly well; what does not survive is *arming*. An abort
    /// is armed from the unfiltered one-step image of where an attempt is,
    /// precisely so that it covers the cycle its subtree's last thread dies in —
    /// and a complement gives a `never` guard to the one state that image would
    /// have been read off. See `Builder.complement`.
    UnsupportedAbortScope,
    /// An implication whose antecedent has no bounded longest match:
    /// `(a ##[1:$] b) |-> c`, `(b[=2]) |-> c`. The diagnostic names the
    /// antecedent.
    ///
    /// An obligation is owed per antecedent *match*, so what the cycle a match
    /// ends in has to report is how many attempts ended one there. Telling them
    /// apart is what `Monitor.trigger_window` does, and an unbounded antecedent
    /// has unboundedly many attempts alive at once for it to tell apart. This is
    /// refused rather than answered with the number of matching *cycles*, which
    /// is what a trigger reporting a bit answers and is wrong by however many
    /// attempts ended together: `merge`'s `a ##[1:$] b` row measures eight in one
    /// cycle.
    ///
    /// Both tracking modes refuse it. The obligation's discipline has no bearing
    /// on it — this is a property of the antecedent, and `.counted` counts
    /// obligations, not the matches that seed them.
    ///
    /// TODO: counting one soundly needs the trigger determinised, where an
    /// attempt is a single state and one `u64` per state is exact. That is
    /// `determinise` over the other automaton, and `Monitor.Mode` would grow a
    /// third discipline to read it.
    AntecedentNotBounded,
    /// One operand of an `intersect` or `and` assigns a local variable slot
    /// that the sibling operand reads from the incoming context. LRM F.5.5
    /// requires both operands to be evaluated from the SAME input context, so
    /// the read sibling cannot see the write; one valuation per state cannot
    /// express both views at once, and the two operands of a product are one
    /// thread here.
    LocalCrossRead,
    /// A local variable assignment under an operator that determinises its
    /// operand — `seq_first_match` or `prop_not`. A determinised state
    /// stands for a SET of operand threads, and with an assignment beneath it
    /// those threads hold different valuations, which is exactly what one
    /// valuation per state cannot carry. Reading a local under either is fine:
    /// with no assignment there, every operand thread carries the valuation
    /// that flowed in.
    LocalUnderDeterminisation,
    /// A zero-delay fusion whose right-hand component reads a local variable
    /// slot the left-hand component assigns in the same cycle. Fusion conjoins
    /// the two guards into one cube, and this shape needs the cube split around
    /// the assignment: the left half must be evaluated before the assignment
    /// and the right half after. One valuation per state cannot express both.
    LocalFusedRead,
    /// Local variables under a universal operator with no upper bound — an
    /// unbounded `prop_always` range, or either operand of a `prop_until*`.
    /// Those leave one attempt per cycle pending for ever, which the counted
    /// discipline exists to collapse into a counter per state; a valuation
    /// makes the collapse unsound: two attempts in one state with different
    /// valuations have different futures. An unbounded *sequence* obligation
    /// is a different shape and stays supported.
    LocalUnboundedAttempts,
    /// The IR declares local variable slots and this lowering does not carry them
    /// yet: no projection onto states, no flow sets, no group discipline. Every
    /// local table would come out empty, which a backend reads as "no local
    /// variables" and simulates as though the match items were absent -- so the
    /// directive is refused rather than lowered into a wrong answer.
    ///
    /// TODO: LRM 16.10. Goes away with the group step.
    LocalIncomplete,
    /// The obligation is bounded but deeper than `Options.depth_limit`, or the
    /// antecedent is and has more attempts in flight than that.
    DepthLimitExceeded,
} || Allocator.Error;

/// A sequence's automaton, with the alphabet its guards are written over.
pub const Automaton = struct {
    alphabet: Nfa.Alphabet,
    nfa: Nfa,

    pub fn deinit(automaton: *Automaton, gpa: Allocator) void {
        automaton.alphabet.deinit(gpa);
        automaton.nfa.deinit(gpa);
    }
};

/// Lower the sequence rooted at `sequence`. The result owns its storage.
pub fn sequence(
    gpa: Allocator,
    ir: *const Ir,
    node: Ir.Node.Index,
    options: Options,
    diagnostic: ?*Diagnostic,
) Error!Automaton {
    std.debug.assert(ir.tagOf(node).layer() == .sequence);

    var builder: Builder = .{ .gpa = gpa, .ir = ir, .options = options, .diagnostic = diagnostic };
    defer builder.deinit();
    try builder.reserveCubes();

    var fragment = try builder.lower(node);
    defer fragment.deinit(gpa);

    var automata: [1]Nfa = undefined;
    const alphabet = try builder.materialize(&.{&fragment}, &.{}, &.{}, false, &.{}, &automata);
    return .{ .alphabet = alphabet, .nfa = automata[0] };
}

/// Lower a directive to a `Monitor`.
///
/// The property layer is thin, and stays thin: an implication contributes a
/// trigger automaton, everything above the sequences folds into the obligation,
/// and the aborts come out as conditions over slices of it. All of it is built
/// against one alphabet, so a term the antecedent and the consequent share is
/// evaluated once per cycle rather than once each.
pub fn assertion(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    options: Options,
    diagnostic: ?*Diagnostic,
) Error!Monitor {
    const directive = ir.assertions.get(@intFromEnum(index));

    // The shapes one valuation per thread cannot represent, refused before
    // anything is built so a diagnostic names a node the front-end can still map
    // back to source (LRM 16.10, F.5.4, F.5.5).
    if (ir.localCount() > 0) {
        var flow = try Flow.compute(gpa, ir);
        defer flow.deinit(gpa);
        try checkLocalHazards(ir, &flow, directive.prop, diagnostic);
    }

    var builder: Builder = .{ .gpa = gpa, .ir = ir, .options = options, .diagnostic = diagnostic };
    defer builder.deinit();
    try builder.reserveCubes();

    // Which discipline tracks the obligation, read off the tree before anything
    // is built. Two things want it this early: a ring has to be sized before it
    // is allocated, and a shift with an open range has no filler chain to build
    // in the first place — so the walk that decides the bound is also the walk
    // that decides what gets built, which is why `propBound` and
    // `lowerProperty` are written as mirrors.
    const bound = propBound(ir, directive.prop);
    if (bound.mixed) |node| {
        if (diagnostic) |into| into.* = .{ .node = node };
        return error.ObligationModeMixed;
    }
    // A local variable makes a thread a (position, valuation) pair, and neither a
    // state vector nor a counter per state can tell two valuations apart, so a
    // monitor with slots is tracked as a list of valuation GROUPS instead --
    // whatever bound its obligation has. `depth` stays the attempt bound where
    // there is one and 0 where there is not, which is what the header promises.
    const mode: Monitor.Mode = if (ir.localCount() > 0)
        .threaded
    else if (bound.length == null)
        .counted
    else
        .ring;
    // Only a ring has a depth; counted mode reports zero and means it.
    const depth = if (bound.length) |longest|
        try builder.checkDepth(bound.culprit, longest)
    else
        0;

    var plan = try builder.lowerProperty(directive.prop);
    defer plan.deinit(gpa);

    // The reset condition is a root in its own right: a backend evaluates it
    // alongside the guards. It is interned last, so it lands past every root a
    // guard mentions, and pinned so materialisation keeps it there even though
    // no guard mentions it. An assertion that never resets has a constant false
    // here, which folds to nothing in any backend. Every abort condition is a
    // root on the same footing.
    const pinned = try gpa.alloc(u32, 1 + plan.aborts.items.len);
    defer gpa.free(pinned);
    pinned[0] = try builder.rootOf(directive.reset);
    for (plan.aborts.items, pinned[1..]) |abort, *into| into.* = abort.cond;

    var fragments: [2]*const Fragment = undefined;
    var count: usize = 1;
    fragments[0] = &plan.obligation;
    if (plan.trigger) |*fragment| {
        fragments[1] = fragment;
        count = 2;
    }

    // Masks over the obligation's states that have to survive everything which
    // renumbers them — pruning, and in counted mode the determinisation too:
    // one per abort, plus the mask of states an attempt has failed in.
    const abort_count = plan.aborts.items.len;
    const masks = try gpa.alloc([]Word, abort_count + @intFromBool(mode == .counted));
    defer gpa.free(masks);
    for (plan.aborts.items, masks[0..abort_count]) |abort, *into| into.* = abort.mask;

    // The determinised obligation's dead states, owned here because no `Plan`
    // field fits it and it outlives the fragment it was computed from.
    var dead: []Word = &.{};
    defer gpa.free(dead);

    if (mode == .counted) {
        try builder.determiniseObligation(directive.prop, &plan, masks, &dead);
        fragments[0] = &plan.obligation;
    }

    var automata: [2]Nfa = undefined;
    var alphabet = try builder.materialize(
        fragments[0..count],
        pinned,
        masks,
        mode == .counted,
        plan.pending orelse &.{},
        automata[0..count],
    );
    errdefer alphabet.deinit(gpa);
    errdefer for (automata[0..count]) |*nfa| nfa.deinit(gpa);

    // How many antecedent attempts the trigger has to tell apart, off the
    // finished automaton rather than off the tree.
    //
    // The automaton is the exact answer and the tree is not: a shift over an
    // implication lands on the antecedent, `first_match(a[*1:$])` is `a` with a
    // longest match of one, and `maxLength` sees neither. Reading it here also
    // reads it after `prune`, which is what licenses the whole scheme — every
    // surviving state is co-reachable to an exit, so a thread that outlives the
    // longest match does not exist.
    var trigger_window: u32 = 0;
    if (count == 2) {
        // Set with the fragment, so a trigger without one is a lowering bug.
        const antecedent = plan.trigger_node.?;
        const reach = try triggerReach(gpa, &automata[1]);
        const longest = reach.longest orelse
            return builder.rejectProperty(antecedent, error.AntecedentNotBounded);
        // A boolean answers when no two matches can end in one cycle, which is
        // when they all span the same number of cycles. An abort above the
        // implication wants the attempts still matching counted as well, so it
        // wants ids whenever a match spans more than one cycle at all.
        const halfway = longest > 1 and coversTrigger(plan.aborts.items);
        if (reach.varied or halfway) {
            trigger_window = try builder.checkDepth(antecedent, longest);
        }
    }

    const aborts = try gpa.alloc(Monitor.Abort, plan.aborts.items.len);
    errdefer gpa.free(aborts);
    const words = automata[0].words;
    const abort_masks = try gpa.alloc(Word, @as(usize, words) * aborts.len);
    for (plan.aborts.items, aborts, pinned[1..], 0..) |abort, *into, root, slot| {
        into.* = .{
            .cond = root,
            .effect = abort.effect,
            .scope = abort.scope,
            .covers_trigger = abort.covers_trigger,
            .sampled = abort.sampled,
        };
        // `materialize` rewrote the mask in place and the surviving states never
        // outnumber the originals, so the final mask is its leading words.
        @memcpy(abort_masks[slot * words ..][0..words], abort.mask[0..words]);
    }

    // `Nfa.dead` is allocated zeroed by `emit`, so a ring monitor needs nothing
    // here. In counted mode the mask came through `materialize` with the abort
    // masks and is read back at the narrower width for the same reason.
    if (mode == .counted) @memcpy(automata[0].dead, dead[0..words]);

    // The local variable tables the backend reads, owned by the monitor: the slot
    // table verbatim from the IR, the dense assignment-site table the states'
    // programs index, and the per-root read mask that splits root evaluation.
    const locals = try gpa.dupe(Ir.Local, ir.locals.items);
    errdefer gpa.free(locals);
    const assign_slot = try gpa.dupe(u32, builder.assign_slot.items);
    errdefer gpa.free(assign_slot);
    const assign_handle = try gpa.dupe(u32, builder.assign_handle.items);
    errdefer gpa.free(assign_handle);
    // Over the alphabet's roots, not the builder's: materialisation compacts and
    // renumbers them, and this mask is indexed the way a backend indexes them.
    const root_local = try rootLocalMask(&builder, gpa, alphabet.roots);
    errdefer gpa.free(root_local);

    return .{
        .id = directive.id,
        .kind = directive.kind,
        .clock = directive.clock,
        .alphabet = alphabet,
        .trigger = if (count == 2) automata[1] else null,
        .overlap = plan.overlap,
        .obligation = automata[0],
        .mode = mode,
        .depth = depth,
        .trigger_window = trigger_window,
        .strength = plan.strength,
        .reset = pinned[0],
        .aborts = aborts,
        .abort_masks = abort_masks,
        .locals = locals,
        .local_words = if (ir.localCount() == 0) 0 else ir.localWords(),
        .assign_slot = assign_slot,
        .assign_handle = assign_handle,
        .root_local = root_local,
    };
}

/// Which roots read a local variable slot, one bit each. A backend evaluates those
/// per valuation group and the rest once per cycle, so this is the whole of what
/// local variables cost a monitor whose sequences only READ them.
fn rootLocalMask(builder: *Builder, gpa: Allocator, roots: []const Ir.Node.Index) Allocator.Error![]Word {
    const words = Nfa.wordsFor(@intCast(roots.len));
    const mask = try gpa.alloc(Word, words);
    @memset(mask, 0);
    if (builder.ir.localCount() == 0) return mask;

    const slot_words = builder.ir.localWords();
    const reads = try gpa.alloc(Word, slot_words);
    defer gpa.free(reads);
    for (roots, 0..) |root, index| {
        @memset(reads, 0);
        builder.boolReads(root, reads);
        for (reads) |word| {
            if (word != 0) {
                Nfa.bitSet(mask, @intCast(index));
                break;
            }
        }
    }
    return mask;
}

/// Whether any of `aborts` was written above the implication, and so covers the
/// antecedent as well as the obligation.
fn coversTrigger(aborts: []const Builder.Plan.Abort) bool {
    for (aborts) |abort| if (abort.covers_trigger) return true;
    return false;
}

/// How long a trigger's matches are, over the finished automaton: the spread
/// that decides `Monitor.trigger_window`.
const Reach = struct {
    /// Cycles the longest match spans, or null when there is no longest — the
    /// automaton has a cycle, so matches grow without bound and so does the
    /// number of attempts alive at once. Refused; see `Error.AntecedentNotBounded`.
    longest: ?u32,
    /// Whether two matches can differ in length. When they cannot, an attempt
    /// seeded in a different cycle ends in a different cycle, so at most one
    /// match ends per cycle and one bit counts them.
    varied: bool,
};

/// `Reach` of a *pruned* automaton, which every trigger is: `prune` keeps only
/// states that are reachable from `first` and co-reachable to `last`.
///
/// A breadth-first walk in lockstep with the cycles. `frontier` after `k` rounds
/// is the states a thread can be in `k` cycles into an attempt, so the rounds it
/// takes to empty is the longest a thread can live — and because every state is
/// co-reachable, a maximal path ends in an exit and that is the longest match
/// too. Guards are not consulted: this is the structural bound, which is what
/// bounds the ids whatever the trace does.
///
/// An acyclic automaton empties inside `state_count` rounds, a path visiting no
/// state twice. Still going after that is exactly a cycle, and a cycle in a
/// co-reachable automaton reaches an exit from every one of its states, so the
/// match lengths are unbounded rather than merely the paths.
fn triggerReach(gpa: Allocator, nfa: *const Nfa) Allocator.Error!Reach {
    const frontier = try maskAlloc(gpa, nfa.words);
    defer gpa.free(frontier);
    const next = try maskAlloc(gpa, nfa.words);
    defer gpa.free(next);

    @memcpy(frontier, nfa.first);
    var shortest: u32 = 0;
    var longest: u32 = 0;
    var cycles: u32 = 1;
    while (Nfa.bitAny(frontier)) : (cycles += 1) {
        if (cycles > nfa.state_count) return .{ .longest = null, .varied = true };
        if (Nfa.bitOverlaps(frontier, nfa.last)) {
            if (shortest == 0) shortest = cycles;
            longest = cycles;
        }
        @memset(next, 0);
        var live: Nfa.BitIterator = .init(frontier);
        while (live.next()) |state| Nfa.bitUnion(next, nfa.followOf(state));
        @memcpy(frontier, next);
    }
    // An antecedent that matches nothing at all leaves both at zero, which is
    // one length rather than two and wants no ids: it never fires.
    return .{ .longest = longest, .varied = shortest != longest };
}

/// The longest match of a sequence in cycles, or null when it has none.
///
/// Saturating: a nesting whose bound overflows is reported as astronomically
/// deep rather than as unbounded, so it is rejected by `depth_limit` with the
/// size that caused it rather than by the wrong error.
fn maxLength(ir: *const Ir, node: Ir.Node.Index) ?u64 {
    const data = ir.dataOf(node);
    switch (ir.tagOf(node)) {
        .seq_boolean => return 1,
        // `and` ends at the later of its two ends, so the longer side's bound
        // is the pair's, and a side with no bound leaves the pair with none —
        // which is exactly how a disjunction behaves too.
        .seq_or, .seq_and => {
            const left = maxLength(ir, @enumFromInt(data.lhs)) orelse return null;
            const right = maxLength(ir, @enumFromInt(data.rhs)) orelse return null;
            return @max(left, right);
        },
        .seq_intersect => {
            // Both sides span the same cycles, so either bound bounds the pair
            // and one unbounded side is harmless.
            const left = maxLength(ir, @enumFromInt(data.lhs));
            const right = maxLength(ir, @enumFromInt(data.rhs));
            if (left) |bound| return if (right) |other| @min(bound, other) else bound;
            return right;
        },
        // The boolean rides along on the operand's cycles without adding any,
        // so it cannot lengthen a match and is not consulted.
        .seq_throughout => return maxLength(ir, @enumFromInt(data.rhs)),
        // Every match `first_match` keeps is one of its operand's, so the
        // operand's bound bounds it too. An operand without one leaves it
        // without one, even where the shortest match is in fact bounded —
        // `first_match(a[*1:$])` is just `a`. Seeing that needs the longest
        // path over the automaton and this reads the tree, so the answer is a
        // rejected obligation rather than a wrong `depth`.
        .seq_first_match => return maxLength(ir, @enumFromInt(data.lhs)),
        // LRM 16.10's assignments happen at the end of a match of the operand
        // and change no cycle of it, so the bound is the operand's.
        .seq_match => return maxLength(ir, @enumFromInt(data.lhs)),
        .seq_delay => {
            const delay = ir.delayOf(node);
            if (delay.hi == Ir.unbounded) return null;
            const right = maxLength(ir, delay.right) orelse return null;
            const left = maxLength(ir, delay.left) orelse return null;
            // A delay of d starts the right match d cycles after the left match
            // ends, and d == 0 shares that cycle.
            return left +| delay.hi +| right -| 1;
        },
        .seq_repeat_consec => {
            const repeat = ir.repeatOf(node);
            if (repeat.hi == Ir.unbounded) return null;
            const operand = maxLength(ir, repeat.seq) orelse return null;
            return @as(u64, repeat.hi) *| operand;
        },
        // However few occurrences are asked for, the gaps between them are
        // unbounded.
        .seq_repeat_goto, .seq_repeat_nonconsec => return null,
        else => unreachable, // not a sequence-layer tag
    }
}

/// Which discipline tracks a property's obligation, and what it costs, read off
/// the tree.
///
/// `Monitor.depth` is the obligation's longest match, and whether it has one at
/// all decides between a ring of state vectors and a counter per state. Both
/// answers are wanted before a fragment exists — a ring so it can be sized, a
/// shift with an open range because there is no filler chain to build in front
/// of it. That is why this mirrors `Builder.lowerProperty` step for step, and
/// why the two have to agree on the one question that is not local: whether a
/// shift lengthens the obligation or the antecedent. An arm added to one wants
/// the matching arm here.
const Bound = struct {
    /// The obligation's longest match in cycles, or null when it has none — in
    /// which case it is counter-tracked rather than refused.
    length: ?u64,
    /// The node a diagnostic names: the one that lost the bound when `length` is
    /// null, and the one charged for the depth when it has one.
    culprit: Ir.Node.Index,
    /// Whether an implication above this point contributes a trigger.
    ///
    /// A trigger is tracked existentially, as one state vector shared by every
    /// attempt, so lengthening it costs the ring nothing — which is why a shift
    /// over an implication lands there and leaves `length` alone.
    triggered: bool = false,
    /// The innermost unbounded operator with a trigger strictly *below* it, if
    /// any: the shape that wants both disciplines at once. See
    /// `Error.ObligationModeMixed`.
    mixed: ?Ir.Node.Index = null,
};

/// What an unbounded operator reports as the mixed-mode culprit: the innermost
/// one already found below it, or itself when a trigger is what it is sitting
/// over.
fn mixedAt(node: Ir.Node.Index, inner: ?Ir.Node.Index, triggered: bool) ?Ir.Node.Index {
    if (inner) |found| return found;
    return if (triggered) node else null;
}

fn propBound(ir: *const Ir, node: Ir.Node.Index) Bound {
    const data = ir.dataOf(node);
    switch (ir.tagOf(node)) {
        .prop_seq_weak, .prop_seq_strong => {
            const seq: Ir.Node.Index = @enumFromInt(data.lhs);
            return .{ .length = maxLength(ir, seq), .culprit = seq };
        },
        // The antecedent is the trigger and never joins the obligation, so its
        // own length is irrelevant here however long it is.
        .prop_implies_overlap, .prop_implies_nonoverlap => {
            var bound = propBound(ir, @enumFromInt(data.rhs));
            bound.triggered = true;
            return bound;
        },
        .prop_delay, .prop_always => {
            const range = ir.propRangeOf(node);
            var bound = propBound(ir, range.prop);
            // An open range is the unbounded `always p` / `s_eventually p`: no
            // longest match, so the obligation is counted rather than ringed.
            // A trigger below cannot come along — the shift would have had to
            // land on it, and an open range has no chain to put there.
            if (range.hi == Ir.unbounded) return .{
                .length = null,
                .culprit = node,
                .triggered = bound.triggered,
                .mixed = mixedAt(node, bound.mixed, bound.triggered),
            };
            const length = bound.length orelse return bound;
            if (bound.triggered) return bound;
            bound.length = length +| range.hi;
            return bound;
        },
        // Both operands join the obligation, and it has no bound either way: the
        // left one may have to hold for the whole trace, and the right one may
        // never arrive. A trigger under either is the mixed shape for the same
        // reason an open shift over one is.
        .prop_until, .prop_until_with, .prop_s_until, .prop_s_until_with => {
            const left = propBound(ir, @enumFromInt(data.lhs));
            const right = propBound(ir, @enumFromInt(data.rhs));
            const triggered = left.triggered or right.triggered;
            return .{
                .length = null,
                .culprit = node,
                .triggered = triggered,
                .mixed = mixedAt(node, left.mixed orelse right.mixed, triggered),
            };
        },
        // A complement resolves in exactly the cycle its operand resolves, so
        // it borrows the operand's bound unchanged — and, notably, does *not*
        // force the counted discipline.
        //
        // That is worth spelling out because the opposite is the natural guess:
        // a complement is built by determinising, and a determinised obligation
        // is what `.counted` runs. But the two are not the same object.
        // `.counted` exists because an obligation with no longest match cannot
        // be held in a ring of `depth` vectors, and it reads failure out of
        // `Nfa.dead` because its determinisation is *total*. `lowerPropNot`
        // hands back a partial fragment: the states where the operand matched
        // are gone, so the vector empties exactly there and the ordinary
        // emptiness reading is restored. A ring holds it like any other
        // fragment.
        //
        // The bound itself: after `maxLength(p)` cycles every thread of a
        // pruned `p` has matched or died, so `p` has resolved and so has
        // `not p`. The operand's bound bounds the complement exactly.
        .prop_not => return propBound(ir, @enumFromInt(data.lhs)),
        // Each side keeps its own schedule and the pair is pending until both
        // have resolved, so the longer side's bound is the pair's — the same
        // reading as `seq_and`, which is what this lowers onto.
        //
        // `prop_implies` and `prop_iff` reduce onto those two, and a complement
        // costs no cycles, so the same fold covers all four.
        .prop_and, .prop_or, .prop_implies, .prop_iff => {
            const left = propBound(ir, @enumFromInt(data.lhs));
            const right = propBound(ir, @enumFromInt(data.rhs));
            const triggered = left.triggered or right.triggered;
            const mixed = left.mixed orelse right.mixed;
            const first = left.length orelse return .{
                .length = null,
                .culprit = left.culprit,
                .triggered = triggered,
                .mixed = mixed,
            };
            const second = right.length orelse return .{
                .length = null,
                .culprit = right.culprit,
                .triggered = triggered,
                .mixed = mixed,
            };
            return .{
                .length = @max(first, second),
                .culprit = node,
                .triggered = triggered,
                .mixed = mixed,
            };
        },
        // A concatenation: the antecedent joins the obligation here rather than
        // becoming a trigger, so its length counts. Overlapping shares a cycle.
        .prop_followed_by_overlap, .prop_followed_by_nonoverlap => {
            const seq: Ir.Node.Index = @enumFromInt(data.lhs);
            var bound = propBound(ir, @enumFromInt(data.rhs));
            const length = bound.length orelse return bound;
            const antecedent = maxLength(ir, seq) orelse return .{
                .length = null,
                .culprit = seq,
                .triggered = bound.triggered,
                .mixed = bound.mixed,
            };
            const gap: u64 = if (ir.tagOf(node) == .prop_followed_by_overlap) 0 else 1;
            bound.length = antecedent +| gap +| length -| 1;
            return bound;
        },
        // An abort only ever resolves an attempt sooner than it would have
        // resolved itself, so it cannot lengthen the obligation.
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
        => return propBound(ir, @enumFromInt(data.rhs)),
        else => unreachable, // not a property-layer tag
    }
}
// -- Fragments ---------------------------------------------------------------

/// A homogeneous ε-free automaton under construction. Same mask representation
/// as the finished `Nfa`, so materialising one is a renumber and a copy.
const Fragment = struct {
    state_count: u32,
    words: u32,
    guard: []Guard.Index,
    follow: []Word,
    first: []Word,
    last: []Word,
    /// Per state, the LRM 16.10 assignment program that runs on arriving there,
    /// indexing `Builder.sites`. `Nfa.Program.none` for every state of a
    /// local-free lowering, which is what keeps this one array's cost the whole
    /// cost of the feature here: the pool it points into is never allocated.
    ///
    /// Positions are renumbered by every combinator, so this travels with them —
    /// `blit` carries it, and the constructions that build a state out of a PAIR
    /// of positions (a zero-delay fusion, an intersect or and product) have to
    /// concatenate the two programs themselves, in that order.
    program: []Nfa.Program,

    /// `state_count` states, all guarded by `always`, with no edges.
    fn init(gpa: Allocator, state_count: u32) Allocator.Error!Fragment {
        const words = Nfa.wordsFor(state_count);
        const guard = try gpa.alloc(Guard.Index, state_count);
        errdefer gpa.free(guard);
        const follow = try gpa.alloc(Word, @as(usize, state_count) * words);
        errdefer gpa.free(follow);
        const first = try gpa.alloc(Word, words);
        errdefer gpa.free(first);
        const last = try gpa.alloc(Word, words);
        errdefer gpa.free(last);
        const program = try gpa.alloc(Nfa.Program, state_count);

        @memset(guard, .always);
        @memset(follow, 0);
        @memset(first, 0);
        @memset(last, 0);
        @memset(program, Nfa.Program.none);
        return .{
            .state_count = state_count,
            .words = words,
            .guard = guard,
            .follow = follow,
            .first = first,
            .last = last,
            .program = program,
        };
    }

    fn deinit(fragment: *Fragment, gpa: Allocator) void {
        gpa.free(fragment.guard);
        gpa.free(fragment.follow);
        gpa.free(fragment.first);
        gpa.free(fragment.last);
        gpa.free(fragment.program);
    }

    fn row(fragment: *Fragment, state: u32) []Word {
        return fragment.follow[@as(usize, state) * fragment.words ..][0..fragment.words];
    }

    fn rowConst(fragment: *const Fragment, state: u32) []const Word {
        return fragment.follow[@as(usize, state) * fragment.words ..][0..fragment.words];
    }

    /// Copy `source`'s states, internal edges and assignment programs in starting
    /// at `base`. Entry and exit sets are the caller's business: a combinator
    /// rarely wants them verbatim.
    ///
    /// A program travels as an index pair into a pool that is never renumbered, so
    /// relocating a state is a plain copy — which is why every combinator that
    /// only relocates states needs nothing else for LRM 16.10.
    fn blit(fragment: *Fragment, source: *const Fragment, base: u32) void {
        for (0..source.state_count) |index| {
            const state: u32 = @intCast(index);
            fragment.guard[base + state] = source.guard[state];
            fragment.program[base + state] = source.program[state];
            blitMask(fragment.row(base + state), source.rowConst(state), base);
        }
    }

    /// The program of a state built out of two positions at once: `left`'s sites
    /// then `right`'s. LRM 16.10 runs a match item list at the end of the match it
    /// is attached to, and a fused or product state ends both, so the left
    /// operand's assignments precede the right's — the same order the source text
    /// has them in.
    fn fuseProgram(
        fragment: *Fragment,
        sites: *std.ArrayListUnmanaged(u32),
        gpa: Allocator,
        state: u32,
        left: Nfa.Program,
        right: Nfa.Program,
    ) Allocator.Error!void {
        if (left.len == 0 and right.len == 0) return;
        if (right.len == 0) {
            fragment.program[state] = left;
            return;
        }
        if (left.len == 0) {
            fragment.program[state] = right;
            return;
        }
        // Reserved up front: the copy reads out of the same list it appends to, and
        // an append that reallocates would leave the source slice dangling.
        try sites.ensureUnusedCapacity(gpa, left.len + right.len);
        const start: u32 = @intCast(sites.items.len);
        for (0..left.len) |offset| sites.appendAssumeCapacity(sites.items[left.start + offset]);
        for (0..right.len) |offset| sites.appendAssumeCapacity(sites.items[right.start + offset]);
        fragment.program[state] = .{ .start = start, .len = left.len + right.len };
    }
};

/// Whether every match of `fragment` spans exactly one cycle: every state is an
/// entry and an exit at once, so a match needs no predecessor and waits on no
/// successor. Meant for a *pruned* fragment, where a state that fails this is a
/// state some match really does pass through.
///
/// Vacuously true of a fragment with no states, which matches nothing at all;
/// `holdsEveryCycle` says why that is the useful reading.
fn singleCycle(fragment: *const Fragment) bool {
    for (0..fragment.state_count) |index| {
        const state: u32 = @intCast(index);
        if (!Nfa.bitIsSet(fragment.first, state)) return false;
        if (!Nfa.bitIsSet(fragment.last, state)) return false;
    }
    return true;
}

/// Union `source`'s bits into `destination`, shifted up by `base`.
fn blitMask(destination: []Word, source: []const Word, base: u32) void {
    var bits: BitIterator = .init(source);
    while (bits.next()) |index| Nfa.bitSet(destination, base + index);
}

fn maskAlloc(gpa: Allocator, words: u32) Allocator.Error![]Word {
    const mask = try gpa.alloc(Word, words);
    @memset(mask, 0);
    return mask;
}

// -- Builder -----------------------------------------------------------------

const Builder = struct {
    gpa: Allocator,
    ir: *const Ir,
    options: Options,
    diagnostic: ?*Diagnostic,

    /// Boolean nodes the guards are written over. See `rootOf` for how one is
    /// keyed.
    roots: std.ArrayListUnmanaged(Ir.Node.Index) = .empty,
    root_of: std.AutoHashMapUnmanaged(u64, u31) = .empty,
    /// Backing pool for interned cubes.
    literals: std.ArrayListUnmanaged(Literal) = .empty,
    /// Backing pool the fragments' `program` slices index: LRM 16.10 assignment
    /// site numbers, dense over the monitor's site table and appended in the order
    /// the `seq_match` nodes are lowered. Never renumbered, so relocating a state
    /// is a copy of the index pair. Empty for a local-free assertion.
    sites: std.ArrayListUnmanaged(u32) = .empty,
    /// Per assignment site, the slot it writes and the opaque handle the backend
    /// evaluates for its value. Parallel to `sites`' contents by index.
    assign_slot: std.ArrayListUnmanaged(u32) = .empty,
    assign_handle: std.ArrayListUnmanaged(u32) = .empty,
    /// Interned cubes; slots 0 and 1 are `always` and `never`.
    table: std.ArrayListUnmanaged(Guard) = .empty,
    /// Cube under construction.
    scratch: std.ArrayListUnmanaged(Literal) = .empty,
    /// Content index over `table`: content hash to the head of a chain of cubes
    /// sharing that hash. See `intern`.
    cube_bucket: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// Parallel to `table`: the next cube sharing a bucket, or `end_of_chain`.
    cube_chain: std.ArrayListUnmanaged(u32) = .empty,

    const end_of_chain: u32 = std.math.maxInt(u32);

    fn deinit(builder: *Builder) void {
        builder.roots.deinit(builder.gpa);
        builder.root_of.deinit(builder.gpa);
        builder.literals.deinit(builder.gpa);
        builder.sites.deinit(builder.gpa);
        builder.assign_slot.deinit(builder.gpa);
        builder.assign_handle.deinit(builder.gpa);
        builder.table.deinit(builder.gpa);
        builder.scratch.deinit(builder.gpa);
        builder.cube_bucket.deinit(builder.gpa);
        builder.cube_chain.deinit(builder.gpa);
    }

    /// Seat the two reserved cubes, in order, so their indices match
    /// `Guard.Index.always` and `.never`. Neither is ever found by content, but
    /// `cube_chain` stays index-aligned with `table`.
    fn reserveCubes(builder: *Builder) Allocator.Error!void {
        for (0..2) |_| {
            try builder.table.append(builder.gpa, .{ .start = 0, .len = 0 });
            try builder.cube_chain.append(builder.gpa, end_of_chain);
        }
    }

    fn fail(builder: *Builder, node: Ir.Node.Index, size: u64) Error {
        if (builder.diagnostic) |into| into.* = .{ .node = node, .size = size };
        return error.StateLimitExceeded;
    }

    /// Project a state count in 64-bit arithmetic and check it against the cap
    /// before a fragment that size is allocated.
    fn checkLimit(builder: *Builder, node: Ir.Node.Index, states: u64) Error!u32 {
        if (states > builder.options.limit) return builder.fail(node, states);
        return @intCast(states);
    }

    fn checkDepth(builder: *Builder, node: Ir.Node.Index, depth: u64) Error!u32 {
        if (depth > builder.options.depth_limit) {
            if (builder.diagnostic) |into| into.* = .{ .node = node, .size = depth };
            return error.DepthLimitExceeded;
        }
        return @intCast(depth);
    }

    // -- Guards --------------------------------------------------------------

    /// Intern a boolean node as a root, returning its index.
    ///
    /// The key is the *term*, not the node. Nothing in the `Ir` hash-conses, so
    /// a front-end building bottom-up emits a fresh `atom` node every time it
    /// mentions a signal, and two such nodes are the same term. Keying on the
    /// handle keeps a backend from evaluating one atom twice per cycle, and —
    /// the part that changes behaviour — lets a cube recognise `a` and `!a` as
    /// contradictory when they arrive as separate nodes, which is the usual
    /// case for a fusion or a product.
    ///
    /// Every other root is compared by identity. Structural equality of
    /// arbitrary boolean trees is the `Ir`'s business, not the lowering's; the
    /// cost of missing it is a duplicated evaluation and a surviving dead
    /// state, never a wrong automaton.
    fn rootOf(builder: *Builder, node: Ir.Node.Index) Allocator.Error!u31 {
        const key: u64 = switch (builder.ir.tagOf(node)) {
            .atom => @intFromEnum(builder.ir.atomOf(node)),
            else => (@as(u64, 1) << 32) | @intFromEnum(node),
        };
        const entry = try builder.root_of.getOrPut(builder.gpa, key);
        if (!entry.found_existing) {
            entry.value_ptr.* = @intCast(builder.roots.items.len);
            try builder.roots.append(builder.gpa, node);
        }
        return entry.value_ptr.*;
    }

    /// Flatten `node` (or its negation) into `scratch` as a conjunction of
    /// literals. Conjunctions split and negations push down — a negated
    /// disjunction is a conjunction by De Morgan — so a guard is as fine a cube
    /// as syntax allows, which is what makes contradictions detectable.
    /// Returns false if the cube is `false` outright.
    fn cube(builder: *Builder, node: Ir.Node.Index, negated: bool) Allocator.Error!bool {
        const ir = builder.ir;
        const data = ir.dataOf(node);
        const tag = ir.tagOf(node);

        if (tag == .bool_const) return (data.lhs != 0) != negated;
        if (tag == .bnot) return builder.cube(@enumFromInt(data.lhs), !negated);
        if ((tag == .band and !negated) or (tag == .bor and negated)) {
            if (!try builder.cube(@enumFromInt(data.lhs), negated)) return false;
            return builder.cube(@enumFromInt(data.rhs), negated);
        }

        try builder.scratch.append(builder.gpa, .{
            .negated = negated,
            .root = try builder.rootOf(node),
        });
        return true;
    }

    fn literalLess(_: void, left: Literal, right: Literal) bool {
        if (left.root != right.root) return left.root < right.root;
        return @intFromBool(left.negated) < @intFromBool(right.negated);
    }

    fn hashLiterals(cube_literals: []const Literal) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(cube_literals));
    }

    fn literalsOf(builder: *const Builder, index: u32) []const Literal {
        const entry = builder.table.items[index];
        return builder.literals.items[entry.start..][0..entry.len];
    }

    /// Canonicalise `scratch` and intern it. A cube holding both polarities of
    /// a root is unsatisfiable, which is the whole reason the flattening above
    /// bothers to split conjunctions.
    ///
    /// Interning is content-addressed through `cube_bucket`, whose values head a
    /// chain threaded through `cube_chain`. Hashing the candidate rather than
    /// copying it in means a hit costs no allocation, and the chain resolves a
    /// collision exactly, without the map ever holding a key that points into
    /// the pool it would be invalidated by.
    fn intern(builder: *Builder) Allocator.Error!Guard.Index {
        const sorted = builder.scratch.items;
        std.mem.sort(Literal, sorted, {}, literalLess);

        var kept: usize = 0;
        for (sorted) |literal| {
            if (kept > 0 and sorted[kept - 1].root == literal.root) {
                if (sorted[kept - 1].negated != literal.negated) return .never;
                continue;
            }
            sorted[kept] = literal;
            kept += 1;
        }
        const candidate = sorted[0..kept];
        if (kept == 0) return .always;

        const bucket = try builder.cube_bucket.getOrPut(builder.gpa, hashLiterals(candidate));
        if (!bucket.found_existing) bucket.value_ptr.* = end_of_chain;

        var index = bucket.value_ptr.*;
        while (index != end_of_chain) : (index = builder.cube_chain.items[index]) {
            const have = builder.literalsOf(index);
            if (have.len != kept) continue;
            if (std.mem.eql(u8, std.mem.sliceAsBytes(have), std.mem.sliceAsBytes(candidate))) {
                return @enumFromInt(index);
            }
        }

        const fresh: u32 = @intCast(builder.table.items.len);
        const start: u32 = @intCast(builder.literals.items.len);
        try builder.literals.appendSlice(builder.gpa, candidate);
        try builder.table.append(builder.gpa, .{ .start = start, .len = @intCast(kept) });
        try builder.cube_chain.append(builder.gpa, bucket.value_ptr.*);
        bucket.value_ptr.* = fresh;
        return @enumFromInt(fresh);
    }

    /// The cube of a boolean node, or of its negation.
    fn guardOf(builder: *Builder, node: Ir.Node.Index, negated: bool) Allocator.Error!Guard.Index {
        builder.scratch.clearRetainingCapacity();
        if (!try builder.cube(node, negated)) return .never;
        return builder.intern();
    }

    /// The cube of `left ∧ right`.
    fn conjoin(builder: *Builder, left: Guard.Index, right: Guard.Index) Allocator.Error!Guard.Index {
        if (left == .never or right == .never) return .never;
        if (left == .always) return right;
        if (right == .always or left == right) return left;

        builder.scratch.clearRetainingCapacity();
        for ([_]Guard.Index{ left, right }) |guard| {
            try builder.scratch.appendSlice(
                builder.gpa,
                builder.literalsOf(@intFromEnum(guard)),
            );
        }
        return builder.intern();
    }

    /// The cube of `left ∧ ¬literal`.
    ///
    /// The complement of a whole cube is a disjunction and so not a guard;
    /// falsifying one literal of it is the piece of that disjunction which
    /// still is one.
    fn conjoinNegated(
        builder: *Builder,
        left: Guard.Index,
        literal: Literal,
    ) Allocator.Error!Guard.Index {
        if (left == .never) return .never;

        builder.scratch.clearRetainingCapacity();
        if (left != .always) {
            try builder.scratch.appendSlice(
                builder.gpa,
                builder.literalsOf(@intFromEnum(left)),
            );
        }
        try builder.scratch.append(builder.gpa, .{
            .negated = !literal.negated,
            .root = literal.root,
        });
        return builder.intern();
    }

    /// Whether `cell` entails `guard`. Both are canonical cubes, so this is the
    /// literals of the second being a subset of the literals of the first, read
    /// off as a merge over two sorted runs.
    fn entails(builder: *const Builder, cell: Guard.Index, guard: Guard.Index) bool {
        if (cell == .never) return true; // false entails anything
        if (guard == .never) return false;
        if (guard == .always or cell == guard) return true;
        if (cell == .always) return false;

        const want = builder.literalsOf(@intFromEnum(guard));
        const have = builder.literalsOf(@intFromEnum(cell));
        var at: usize = 0;
        for (want) |literal| {
            while (at < have.len and literalLess({}, have[at], literal)) at += 1;
            if (at == have.len) return false;
            if (@as(u32, @bitCast(have[at])) != @as(u32, @bitCast(literal))) return false;
            at += 1;
        }
        return true;
    }

    // -- Property lowering ---------------------------------------------------

    /// What a property contributes to a monitor, under construction.
    ///
    /// The property layer does not get automaton machinery of its own. Every
    /// operator here is a *verdict* combinator, and a verdict combinator over
    /// sequence properties collapses onto the sequence layer, because a
    /// sequence property holds exactly when its sequence has a match:
    ///
    ///   - `p and q` holds when both hold, on independent schedules — which is
    ///     `seq_and`, whose match ends at the later of the two ends.
    ///   - `p or q` is `seq_or`.
    ///   - `nexttime[n] p` and `eventually [m:n] p` are a filler chain in front
    ///     of `p`, entered at one offset in the range: `shift`.
    ///   - `always [m:n] p` is the conjunction of `p` shifted to each offset.
    ///   - `s #-# p` is `s` concatenated with `p` — see `lowerFollowedBy`.
    ///
    /// So a `Plan` is what a monitor already holds, and lowering a property is
    /// folding the tree into it.
    const Plan = struct {
        /// The obligation, one attempt of which the ring tracks per seeding.
        obligation: Fragment,
        /// The antecedent of the implication this property is, if any.
        trigger: ?Fragment = null,
        /// The sequence node `trigger` was lowered from, for a diagnostic to
        /// name when the antecedent is what a refusal is about. Set with
        /// `trigger` and meaningless without it.
        trigger_node: ?Ir.Node.Index = null,
        /// Whether the obligation starts on the trigger's last cycle.
        overlap: bool = false,
        strength: Monitor.Strength,
        /// Aborts scoped to part of `obligation`, their masks indexed by its
        /// states and rewritten whenever those states move.
        aborts: std.ArrayListUnmanaged(Abort) = .empty,
        /// Obligation states that are *pending by construction*: they lead to no
        /// exit, and that is not evidence the attempt has failed.
        ///
        /// Null when the property contains no complement, which is every shape
        /// before phase 4 and most shapes after it.
        ///
        /// The rest of the lowering equates "this state can reach no exit" with
        /// "no continuation of this attempt can match", and for a sequence
        /// obligation those really are the same sentence — which is why `prune`
        /// deletes such states and lets the vector go empty there. A complement
        /// breaks the equivalence. Inside `not p`, a state that cannot reach an
        /// exit is a state from which `p` is *certain to match eventually*, and
        /// `not p` does not fail until `p` actually does. Deleting it would
        /// report the failure at the cycle the outcome became inevitable rather
        /// than the cycle it arrived — which is `not pending` resolving to
        /// something other than pending, and it would cost `not not p` its
        /// agreement with `p`.
        ///
        /// So this mask is the third verdict made structural: the states where
        /// the automaton has nothing left to say and the answer is still
        /// "pending". `prune` keeps them, and it is the only thing that reads
        /// this. Nothing survives into the `Monitor` — once pruning is done they
        /// are ordinary states.
        pending: ?[]Word = null,

        const Abort = struct {
            cond: u32,
            effect: Monitor.Abort.Effect,
            scope: Monitor.Abort.Scope,
            /// The `prop_*_on` node this came from, so a refusal reached later —
            /// once the tracking mode is known — can still name it.
            node: Ir.Node.Index,
            covers_trigger: bool,
            sampled: bool,
            /// `obligation.words` long.
            mask: []Word,
        };

        fn deinit(plan: *Plan, gpa: Allocator) void {
            plan.obligation.deinit(gpa);
            if (plan.trigger) |*trigger| trigger.deinit(gpa);
            for (plan.aborts.items) |abort| gpa.free(abort.mask);
            plan.aborts.deinit(gpa);
            if (plan.pending) |mask| gpa.free(mask);
        }

        /// Replace the obligation, having moved its states to `base` inside
        /// `replacement`, and carry every mask over those states across.
        fn rebase(
            plan: *Plan,
            gpa: Allocator,
            replacement: Fragment,
            base: u32,
        ) Allocator.Error!void {
            for (plan.aborts.items) |*abort| {
                const moved = try maskAlloc(gpa, replacement.words);
                blitMask(moved, abort.mask, base);
                gpa.free(abort.mask);
                abort.mask = moved;
            }
            if (plan.pending) |mask| {
                const moved = try maskAlloc(gpa, replacement.words);
                blitMask(moved, mask, base);
                gpa.free(mask);
                plan.pending = moved;
            }
            plan.obligation.deinit(gpa);
            plan.obligation = replacement;
        }

        /// Union `other`'s pending states, shifted up by `base`, into this
        /// plan's — for a combinator that keeps both operands' states as blocks.
        fn absorbPending(
            plan: *Plan,
            gpa: Allocator,
            other: *const Plan,
            base: u32,
        ) Allocator.Error!void {
            const source = other.pending orelse return;
            const into = plan.pending orelse mask: {
                const fresh = try maskAlloc(gpa, plan.obligation.words);
                plan.pending = fresh;
                break :mask fresh;
            };
            blitMask(into, source, base);
        }
    };

    fn rejectProperty(builder: *Builder, node: Ir.Node.Index, err: Error) Error {
        if (builder.diagnostic) |into| into.* = .{ .node = node };
        return err;
    }

    fn lowerProperty(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const ir = builder.ir;
        const gpa = builder.gpa;
        const data = ir.dataOf(node);
        switch (ir.tagOf(node)) {
            .prop_seq_weak, .prop_seq_strong => {
                const obligation = try builder.lower(@enumFromInt(data.lhs));
                return .{
                    .obligation = obligation,
                    .strength = if (ir.tagOf(node) == .prop_seq_weak) .weak else .strong,
                };
            },
            .prop_implies_overlap, .prop_implies_nonoverlap => {
                var plan = try builder.lowerProperty(@enumFromInt(data.rhs));
                errdefer plan.deinit(gpa);
                if (plan.trigger != null) {
                    return builder.rejectProperty(@enumFromInt(data.rhs), error.UnsupportedConsequent);
                }
                plan.trigger = try builder.lower(@enumFromInt(data.lhs));
                plan.trigger_node = @enumFromInt(data.lhs);
                plan.overlap = ir.tagOf(node) == .prop_implies_overlap;
                return plan;
            },
            .prop_delay, .prop_always => return builder.lowerPropShift(node),
            .prop_and, .prop_or, .prop_implies, .prop_iff => {
                return builder.lowerPropJunction(node);
            },
            .prop_not => return builder.lowerPropNot(node),
            .prop_until,
            .prop_until_with,
            .prop_s_until,
            .prop_s_until_with,
            => return builder.lowerUntil(node),
            .prop_followed_by_overlap, .prop_followed_by_nonoverlap => {
                return builder.lowerFollowedBy(node);
            },
            .prop_accept_on,
            .prop_reject_on,
            .prop_sync_accept_on,
            .prop_sync_reject_on,
            => return builder.lowerAbort(node),
            else => unreachable, // not a property-layer tag
        }
    }

    /// `nexttime[n] p`, `eventually [m:n] p` and `always [m:n] p`.
    ///
    /// Over an implication the shift lands on the *antecedent* rather than the
    /// obligation, and that is not an optimisation but the only correct place
    /// for it. `nexttime[n] (s |-> p)` obliges `p` for exactly those matches of
    /// `s` that start `n` cycles into the attempt; a trigger is seeded every
    /// cycle, so growing an `n`-cycle filler prefix selects exactly those and
    /// costs the ring nothing.
    ///
    /// A *range* only survives that move for `always`: a conjunction of
    /// universals is a universal over the union of their triggers, which one
    /// trigger expresses. `eventually [m:n] (s |-> p)` is a *disjunction* of
    /// implications, each vacuously true on its own, and no single trigger says
    /// that.
    fn lowerPropShift(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        const range = builder.ir.propRangeOf(node);
        const conjunctive = builder.ir.tagOf(node) == .prop_always;

        var plan = try builder.lowerProperty(range.prop);
        errdefer plan.deinit(gpa);

        // A trigger below an open range is `propBound`'s `ObligationModeMixed`,
        // refused before anything was built, so what is left here either has no
        // trigger or has a bounded range to shift it by.
        if (range.hi == Ir.unbounded) {
            std.debug.assert(plan.trigger == null);
            try builder.lowerOpenShift(node, &plan, range.lo, conjunctive);
            return plan;
        }

        if (plan.trigger) |*trigger| {
            if (range.lo != range.hi and !conjunctive) {
                return builder.rejectProperty(node, error.UnsupportedConsequent);
            }
            const shifted = try builder.shift(node, trigger, range.lo, range.hi);
            trigger.deinit(gpa);
            plan.trigger = shifted;
            return plan;
        }

        if (!conjunctive) {
            const shifted = try builder.shift(node, &plan.obligation, range.lo, range.hi);
            // `shift` puts the operand's states above the filler chain, so the
            // filler count is the base the abort masks move to.
            try plan.rebase(gpa, shifted, range.hi);
            return plan;
        }

        // A conjunction over the range. Every conjunct is the same property at a
        // different offset, so a product is what "all of them match" means and
        // the state count is the price of the operator rather than of spelling
        // it this way — see the tests below for the measurement.
        if (plan.aborts.items.len != 0) {
            return builder.rejectProperty(node, error.UnsupportedAbortScope);
        }
        if (plan.pending != null) {
            return builder.rejectProperty(node, error.UnsupportedComplement);
        }
        var conjunction: ?Fragment = null;
        errdefer if (conjunction) |*fragment| fragment.deinit(gpa);
        var offset = range.lo;
        while (offset <= range.hi) : (offset += 1) {
            var conjunct = try builder.shift(node, &plan.obligation, offset, offset);
            defer conjunct.deinit(gpa);
            if (conjunction) |*accumulated| {
                const combined = try builder.lowerAnd(node, accumulated, &conjunct);
                accumulated.deinit(gpa);
                conjunction = combined;
            } else {
                conjunction = try builder.copyFragment(&conjunct);
            }
        }
        try plan.rebase(gpa, conjunction.?, 0);
        conjunction = null;
        return plan;
    }

    /// `s_eventually p` — an open range read disjunctively — and `always p`, the
    /// same range read conjunctively. Both come out of `lowerPropShift` once its
    /// filler chain has nothing finite to be.
    ///
    /// # `s_eventually`
    ///
    /// LRM 16.12.2: the property holds at *some* cycle at or after the offset.
    /// One attempt of `p` per candidate cycle, and only one of them has to
    /// succeed — so a filler chain whose last state loops on itself and feeds
    /// `p`'s entries says it: the attempt waits in the loop for as long as it
    /// likes, and each cycle it could commit in is a different path through the
    /// one fragment. Nothing here is universal, so nothing constrains `p`'s
    /// length.
    ///
    /// It never fails inside the trace, and that is not a gap: while the loop is
    /// live some later cycle could still satisfy `p`, so there is nothing to
    /// report. Which is exactly why the LRM's unbounded eventually is *strong* —
    /// the verdict is the end of the trace finding the attempt still pending.
    ///
    /// # `always`
    ///
    /// The same range universally: `p` holds at *every* cycle from the offset on.
    /// Two consequences, both the opposite of the above.
    ///
    /// The obligation has **no exits**. "For ever" has no finite witness, so the
    /// fragment can never complete a match — it can only die, when `p` is
    /// violated. That is the honest automaton for a safety property rather than a
    /// degenerate one: the attempt pends for as long as the property holds, which
    /// under the weak reading the end of the trace turns into a pass, and a
    /// counted monitor reads the death out of `Nfa.dead` rather than out of an
    /// exit there is none of.
    ///
    /// And `p` must match in a single cycle — `holdsEveryCycle`'s restriction,
    /// for its reason.
    fn lowerOpenShift(
        builder: *Builder,
        node: Ir.Node.Index,
        plan: *Plan,
        lo: u32,
        conjunctive: bool,
    ) Error!void {
        const gpa = builder.gpa;

        if (!conjunctive) {
            const shifted = try builder.shiftOpen(node, &plan.obligation, lo);
            // As `shift`: the operand's states land above the filler chain, so
            // the chain's length is the base an abort mask below moves to.
            try plan.rebase(gpa, shifted, @max(lo, 1));
            return;
        }

        // A conjunction over unboundedly many offsets, so there is no product to
        // build the way the bounded arm does; the single-cycle operand is what
        // collapses all of them into one loop. Rewiring the operand renumbers
        // nothing but rewrites everything, so an abort mask over it cannot come
        // along — the bounded conjunctive arm refuses one for the same reason.
        if (plan.aborts.items.len != 0) {
            return builder.rejectProperty(node, error.UnsupportedAbortScope);
        }
        if (plan.pending != null) {
            return builder.rejectProperty(node, error.UnsupportedComplement);
        }
        var forever = try builder.holdsEveryCycle(builder.ir.propRangeOf(node).prop, &plan.obligation);
        defer forever.deinit(gpa);
        // The run's exits are what `until` concatenates onto; `always` has
        // nothing to concatenate and nothing to complete, so they go.
        @memset(forever.last, 0);
        const shifted = try builder.shift(node, &forever, lo, lo);
        try plan.rebase(gpa, shifted, lo);
    }

    /// `operand`, entered at *any* offset from `lo` on: `shift` with the filler
    /// chain's last state looping on itself instead of running out.
    ///
    /// `max(lo, 1)` fillers. Offset 0 enters `operand` from `first` directly, as
    /// in `shift`, and the loop still has to live somewhere when `lo` is 0.
    fn shiftOpen(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        lo: u32,
    ) Error!Fragment {
        const fillers = @max(lo, 1);
        const total = try builder.checkLimit(node, @as(u64, fillers) + operand.state_count);
        var fragment = try Fragment.init(builder.gpa, total);
        errdefer fragment.deinit(builder.gpa);

        for (0..fillers - 1) |index| {
            Nfa.bitSet(fragment.row(@intCast(index)), @intCast(index + 1));
        }
        Nfa.bitSet(fragment.row(fillers - 1), fillers - 1);
        fragment.blit(operand, fillers);

        Nfa.bitSet(fragment.first, 0);
        if (lo == 0) blitMask(fragment.first, operand.first, fillers);
        blitMask(fragment.row(fillers - 1), operand.first, fillers);
        blitMask(fragment.last, operand.last, fillers);
        return fragment;
    }

    /// `operand[*1:$]`: the run of cycles in every one of which `operand` holds.
    ///
    /// `operand` has to match in a single cycle, which after pruning is the
    /// structural claim that every surviving state is an entry and an exit at
    /// once — each state alone is a whole match, so "`operand` holds this cycle"
    /// is "some state's guard holds this cycle". Rewiring every state to the
    /// entries then asks that of every cycle of the run.
    ///
    /// A longer operand needs one attempt of it per cycle, all overlapping, and a
    /// single state cannot carry a set of them that grows with the trace.
    /// `blame` is the node a refusal names; see `Error.ObligationNotBounded` for
    /// what lifting the restriction would take.
    ///
    /// An operand that prunes to nothing is kept rather than refused: it can
    /// never hold, so the run is the empty language, which `determinise` turns
    /// into the bare trap and every attempt fails at once. That is the right
    /// answer for `always (a and !a)`.
    fn holdsEveryCycle(
        builder: *Builder,
        blame: Ir.Node.Index,
        operand: *const Fragment,
    ) Error!Fragment {
        const gpa = builder.gpa;
        // Pruned first, so that an unreachable state or a self-contradictory
        // guard cannot fail the check below for a reason that has nothing to do
        // with how many cycles the operand spans.
        var fragment = try builder.compact(operand, &.{}, false, &.{});
        errdefer fragment.deinit(gpa);
        if (!singleCycle(&fragment)) {
            return builder.rejectProperty(blame, error.ObligationNotBounded);
        }

        for (0..fragment.state_count) |index| {
            const row = fragment.row(@intCast(index));
            @memcpy(row, fragment.first);
        }
        return fragment;
    }

    /// The four `until` forms.
    ///
    /// # The derivation
    ///
    /// LRM 16.12.10: `p until q` holds when `q` holds at some cycle `k` at or
    /// after the attempt's first, and `p` holds at every cycle in `[t, k)`. Read
    /// as a set of matches rather than as a verdict that is a *sequence*, and one
    /// the combinators already spell:
    ///
    ///   - `k == t` asks nothing of `p`, so `q` alone is a match.
    ///   - `k > t` asks `p` of `[t, k)` — a non-empty run — and `q` from `k`.
    ///     `holdsEveryCycle` is that run, and a delay of one concatenates `q`.
    ///
    /// so `p until q ≡ q or (p[*1:$] ##1 q)`.
    ///
    /// The property holds exactly when that set is non-empty. Let `k0` be the
    /// first cycle `q` holds in: if `p` held over `[t, k0)` then `k0` is a match,
    /// and conversely a match at any `k` needs `p` over `[t, k)`, which contains
    /// `[t, k0)` because `k0 <= k`. Same claim — and a monitor resolving an
    /// attempt at its *first* match resolves it at `k0`, which is the cycle the
    /// definition names.
    ///
    /// It fails in the first cycle nothing is live, and that is the first cycle no
    /// continuation can match from: `p` has failed, so no further attempt of `q`
    /// can start, and every attempt of `q` already started has died. Both readings
    /// give `max` of those two cycles; the differential test is what checks it
    /// rather than this comment.
    ///
    /// `until_with` includes cycle `k` — `p` must hold there too. LRM 16.12.10
    /// gives `p until_with q ≡ p until (q and p)`, and "`p` at `k`, and `q` from
    /// `k`" is a *fusion*: one cycle that is simultaneously the run's last and
    /// `q`'s first, guards conjoined, which is a delay of zero. So
    ///
    ///     p until_with q ≡ p[*1:$] ##0 q
    ///
    /// with no disjunct for `k == t`, because a one-repetition run already is
    /// that case.
    ///
    /// The `s_` forms are the same construction. Strength is not a property of
    /// the match set at all — it is what becomes of an attempt the trace ends
    /// with still pending, and `q` never arriving is precisely that.
    ///
    /// # What is restricted
    ///
    /// `p`, for `holdsEveryCycle`'s reason. `q` is not: it is existential, one
    /// attempt per candidate `k` with only one having to succeed, which the
    /// concatenation expresses however many cycles `q` spans.
    fn lowerUntil(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        const ir = builder.ir;
        const data = ir.dataOf(node);
        const tag = ir.tagOf(node);
        const inclusive = tag == .prop_until_with or tag == .prop_s_until_with;

        var left = try builder.lowerProperty(@enumFromInt(data.lhs));
        errdefer left.deinit(gpa);
        var right = try builder.lowerProperty(@enumFromInt(data.rhs));
        errdefer right.deinit(gpa);

        // A monitor has one trigger and neither operand may bring a second.
        // `propBound` refuses this shape as `ObligationModeMixed` before anything
        // is built; these stay because the two walks are hand-written mirrors, and
        // a divergence should refuse rather than mislower.
        if (left.trigger != null) {
            return builder.rejectProperty(@enumFromInt(data.lhs), error.UnsupportedConsequent);
        }
        if (right.trigger != null) {
            return builder.rejectProperty(@enumFromInt(data.rhs), error.UnsupportedConsequent);
        }
        // A fusion and a disjoint union both renumber, and the run rewrites the
        // left operand's edges outright, so no abort mask survives from either
        // side.
        if (left.aborts.items.len != 0 or right.aborts.items.len != 0) {
            return builder.rejectProperty(node, error.UnsupportedAbortScope);
        }
        if (left.pending != null or right.pending != null) {
            return builder.rejectProperty(node, error.UnsupportedComplement);
        }

        var run = try builder.holdsEveryCycle(@enumFromInt(data.lhs), &left.obligation);
        defer run.deinit(gpa);

        const combined = if (inclusive)
            try builder.lowerDelay(node, &run, 0, 0, &right.obligation)
        else tail: {
            var concatenated = try builder.lowerDelay(node, &run, 1, 1, &right.obligation);
            defer concatenated.deinit(gpa);
            break :tail try builder.lowerOr(node, &right.obligation, &concatenated);
        };
        try left.rebase(gpa, combined, 0);
        // The tag carries the disposition, not the operands. See `Ir.propUntil`.
        left.strength = switch (tag) {
            .prop_until, .prop_until_with => .weak,
            else => .strong,
        };
        right.deinit(gpa);
        return left;
    }

    /// `p and q`, `p or q`, and the two reductions onto them: `p implies q` is
    /// `not p or q`, and `p iff q` is the conjunction of the two directions.
    ///
    /// The reductions are lowered here rather than rewritten into the `Ir`
    /// because a refusal has to name the operator as written. Building the
    /// desugared tree instead would put `prop_or` on the diagnostic and leave a
    /// front-end to guess which of its own operators produced it.
    fn lowerPropJunction(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        const data = builder.ir.dataOf(node);
        const tag = builder.ir.tagOf(node);

        var left = try builder.lowerProperty(@enumFromInt(data.lhs));
        errdefer left.deinit(gpa);
        var right = try builder.lowerProperty(@enumFromInt(data.rhs));
        errdefer right.deinit(gpa);

        if (left.trigger != null) return builder.rejectProperty(@enumFromInt(data.lhs), error.UnsupportedConsequent);
        if (right.trigger != null) return builder.rejectProperty(@enumFromInt(data.rhs), error.UnsupportedConsequent);

        switch (tag) {
            .prop_and, .prop_or => {},
            // `not p or q`. The strength requirement is checked *before* the
            // complement is built, because the complement is the exponential
            // step and a refusal should not pay for it. Inverted, it reads:
            // the disjunction below wants its operands to agree, `not` turns
            // `p`'s strength over, so `implies` wants its operands to differ.
            .prop_implies => {
                if (left.strength == right.strength) {
                    return builder.rejectProperty(node, error.MixedStrength);
                }
                try builder.complement(node, &left);
            },
            // `(p implies q) and (q implies p)`, which never lowers.
            //
            // Each direction already requires its two operands to carry
            // opposite strengths, and each direction comes out carrying its
            // right-hand operand's — so the left conjunct is `q`'s strength and
            // the right conjunct is `p`'s, which the first requirement has just
            // forced to be the opposite of `q`'s. The conjunction below wants
            // them equal. No assignment satisfies both, so this is arithmetic
            // rather than a case that happens not to be handled, and there is
            // nothing to gain by building the two complements first to discover
            // it. Both operands *are* lowered above, so a malformed operand is
            // still reported before this.
            //
            // `Error.MixedStrength` carries the lift that would make it work.
            .prop_iff => return builder.rejectProperty(node, error.MixedStrength),
            else => unreachable, // not a junction tag
        }

        if (left.strength != right.strength) return builder.rejectProperty(node, error.MixedStrength);

        if (tag == .prop_and) {
            // A product renumbers both sides, so neither may carry a mask over
            // its states: not an abort's, and not a complement's pending set.
            if (left.aborts.items.len != 0 or right.aborts.items.len != 0) {
                return builder.rejectProperty(node, error.UnsupportedAbortScope);
            }
            if (left.pending != null or right.pending != null) {
                return builder.rejectProperty(node, error.UnsupportedComplement);
            }
            const combined = try builder.lowerAnd(node, &left.obligation, &right.obligation);
            try left.rebase(gpa, combined, 0);
            right.deinit(gpa);
            return left;
        }

        // A disjoint union: the left operand keeps its state numbering and the
        // right one is offset by it, so both sides' masks survive.
        const base = left.obligation.state_count;
        const combined = try builder.lowerOr(node, &left.obligation, &right.obligation);
        try left.rebase(gpa, combined, 0);
        try left.absorbPending(gpa, &right, base);

        try left.aborts.ensureUnusedCapacity(gpa, right.aborts.items.len);
        for (right.aborts.items) |abort| {
            const moved = try maskAlloc(gpa, left.obligation.words);
            blitMask(moved, abort.mask, base);
            left.aborts.appendAssumeCapacity(.{
                .cond = abort.cond,
                .effect = abort.effect,
                .scope = abort.scope,
                .node = abort.node,
                // Always false here: both operands were refused above if they
                // carried a trigger, so no abort reaching this point covers one.
                .covers_trigger = abort.covers_trigger,
                .sampled = abort.sampled,
                .mask = moved,
            });
        }
        right.deinit(gpa);

        // A reject below a disjunction only kills its own disjunct: the other
        // one may still hold, so the attempt is not resolved. An accept is the
        // other way round — one disjunct holding makes the disjunction hold —
        // and keeps resolving the whole attempt.
        for (left.aborts.items) |*abort| {
            if (abort.effect == .reject) abort.scope = .branch;
        }
        return left;
    }

    /// `not p`: the operand's plan, complemented in place.
    fn lowerPropNot(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        var plan = try builder.lowerProperty(@enumFromInt(builder.ir.dataOf(node).lhs));
        errdefer plan.deinit(gpa);
        try builder.complement(node, &plan);
        return plan;
    }

    /// Turn `plan` into the plan for its own negation. `node` is what a
    /// diagnostic names — the `prop_not`, or the `prop_implies` whose reduction
    /// asked for this.
    ///
    /// # The automaton
    ///
    /// Determinise the obligation *totally* and exchange two masks. That is the
    /// whole construction, and the totality is what makes it one:
    ///
    ///   - A total determinised state is a subset of the operand's states, and
    ///     the subset is empty exactly where every thread of the operand has
    ///     died — which is where the operand *fails*, and so where `not p`
    ///     holds. Those become the complement's exits.
    ///   - A state whose subset meets the operand's exits is where the operand
    ///     completes a match, and so where `not p` fails. A failure in this
    ///     library is an empty state vector, so those states are given a `never`
    ///     guard: nothing enters them, the vector empties in exactly that cycle,
    ///     and the ordinary reading reports the failure with no new machinery.
    ///
    /// Deleting them cannot leave a stray sibling holding the attempt open. Two
    /// determinised states enabled by one valuation differ only in their cell —
    /// they share a subset, so they are accepting together or neither is, and a
    /// valuation that would have entered a deleted state has nowhere else to go.
    ///
    /// The result is an ordinary fragment. It composes with `lowerOr` and the
    /// shifts like any other, and it does *not* select the counted discipline:
    /// see `propBound`, which borrows the operand's bound unchanged.
    ///
    /// # The part that is not the automaton
    ///
    /// Strength inverts (LRM 16.12.2). A pending attempt stays pending under
    /// negation — that is the third row of the lattice — so all that turns over
    /// is what the end of the trace makes of it.
    ///
    /// `Plan.pending` is set to every state, which is the third row again, made
    /// structural. Inside a complement, "this state can reach no exit" means
    /// "the operand is certain to match from here", not "the attempt has
    /// failed" — the failure arrives when the operand's match does. Pruning such
    /// a state would report it at the cycle the outcome became inevitable
    /// instead, which is `not pending` resolving to something other than
    /// pending, and would cost `not not p` its agreement with `p`.
    ///
    /// # Why an abort underneath is refused
    ///
    /// Not because the mask cannot be moved — `determinise` moves it. Because
    /// the *arming* question stops being answerable, and it stops being
    /// answerable on the one cycle that matters.
    ///
    /// An abort is armed when its subtree could hold a thread this cycle, which
    /// `Monitor.State` reads as the one-step reachable set of the pre-advance
    /// vector — deliberately the reachable set and not the live one, so that it
    /// covers the cycle the subtree's last thread *dies* in. That cycle is the
    /// whole point of an accept: `accept_on(c) p` with `c` holding as `p`'s last
    /// thread dies discharges the attempt instead of failing it.
    ///
    /// A determinised successor is already filtered by the valuation, so the
    /// unfiltered reachable set is not recoverable from it — and the complement
    /// then deletes exactly the states that would have answered. The state
    /// standing for "the operand's threads survived" is the state whose subset
    /// meets the operand's exits, and that is the one given a `never` guard two
    /// paragraphs up. What is left in the reachable set is the trap, whose
    /// subset is empty and meets no abort's mask, so the abort reads as
    /// unarmed in precisely the cycle it was supposed to fire.
    ///
    /// TODO: lift it by carrying a second mask per abort — the determinised
    /// states whose subset's *unfiltered* one-step image meets the abort's
    /// subtree — and arming from that instead of from `⋃ follow`. That is a
    /// `Monitor.Abort` field and a rule change in `State.step`, so it is an ABI
    /// decision rather than a local one.
    fn complement(builder: *Builder, node: Ir.Node.Index, plan: *Plan) Error!void {
        const gpa = builder.gpa;

        // A trigger is the union of every attempt of the antecedent, which is
        // sound only for the existential question a trigger is ever asked.
        // Complementing needs a per-attempt answer the union has discarded.
        if (plan.trigger != null) {
            return builder.rejectProperty(node, error.UnsupportedComplement);
        }
        if (plan.aborts.items.len != 0) {
            return builder.rejectProperty(plan.aborts.items[0].node, error.UnsupportedAbortScope);
        }

        // No abort survives the check above, so no mask has to be carried
        // through either the pruning or the subsets.
        std.debug.assert(plan.aborts.items.len == 0);

        // Pruned first, for `determiniseObligation`'s reason: a state that leads
        // to no exit has to be gone before the subsets are built, or it sits
        // inside a non-empty subset that is nonetheless doomed and the operand's
        // failure — this complement's *match* — is reported late or never. An
        // exit-free operand is `always p`, where co-reachability is vacuous and
        // pruning on it would delete everything.
        const safety = !Nfa.bitAny(plan.obligation.last);
        const clean = try builder.compact(
            &plan.obligation,
            &.{},
            safety,
            plan.pending orelse &.{},
        );
        plan.obligation.deinit(gpa);
        plan.obligation = clean;

        // An inner complement's pending set has done its work above and does not
        // survive the subsets: the states it named are folded into subsets now,
        // and the outer complement recomputes the question for itself below.
        if (plan.pending) |stale| {
            gpa.free(stale);
            plan.pending = null;
        }

        var total = try builder.determinise(node, &plan.obligation, &.{});
        errdefer total.deinit(gpa);

        // The swap. `last` has to be read before it is written over.
        const matched = try gpa.dupe(Word, total.fragment.last);
        defer gpa.free(matched);
        @memcpy(total.fragment.last, total.dead);
        var accepting: BitIterator = .init(matched);
        while (accepting.next()) |state| total.fragment.guard[state] = .never;

        const pending = try maskAlloc(gpa, total.fragment.words);
        errdefer gpa.free(pending);
        for (0..total.fragment.state_count) |state| Nfa.bitSet(pending, @intCast(state));

        plan.obligation.deinit(gpa);
        plan.obligation = total.fragment;
        gpa.free(total.dead);
        plan.pending = pending;
        plan.strength = plan.strength.not();
    }

    /// `s #-# p` and `s #=# p`.
    ///
    /// LRM 16.12.9 defines followed-by as `not (seq |-> not prop)`, and reading
    /// that as an automaton is what makes it cheap. Negating the implication
    /// turns its universal over antecedent matches into an existential *and*
    /// turns vacuity from a pass into a failure, and an existential over
    /// "`s` matches, then `p` holds from there" is exactly a match of the two
    /// concatenated. So this is not the implication construction with a flipped
    /// verdict — it never becomes a trigger at all, and the antecedent joins the
    /// obligation instead.
    fn lowerFollowedBy(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        const data = builder.ir.dataOf(node);
        const gap: u32 = if (builder.ir.tagOf(node) == .prop_followed_by_overlap) 0 else 1;

        var plan = try builder.lowerProperty(@enumFromInt(data.rhs));
        errdefer plan.deinit(gpa);
        if (plan.trigger != null) {
            return builder.rejectProperty(@enumFromInt(data.rhs), error.UnsupportedConsequent);
        }
        // A fusion manufactures a state per exit/entry pair, so the consequent's
        // states do not survive as a block and an abort mask over them cannot be
        // carried through.
        if (plan.aborts.items.len != 0) {
            return builder.rejectProperty(node, error.UnsupportedAbortScope);
        }
        if (plan.pending != null) {
            return builder.rejectProperty(node, error.UnsupportedComplement);
        }

        var antecedent = try builder.lower(@enumFromInt(data.lhs));
        defer antecedent.deinit(gpa);
        const combined = try builder.lowerDelay(node, &antecedent, gap, gap, &plan.obligation);
        try plan.rebase(gpa, combined, 0);
        return plan;
    }

    /// `accept_on`, `reject_on`, and their `sync_` forms.
    fn lowerAbort(builder: *Builder, node: Ir.Node.Index) Error!Plan {
        const gpa = builder.gpa;
        const tag = builder.ir.tagOf(node);
        const data = builder.ir.dataOf(node);

        var plan = try builder.lowerProperty(@enumFromInt(data.rhs));
        errdefer plan.deinit(gpa);

        // An abort above an implication covers the antecedent as well as the
        // consequent, and pushing it down into the consequent is a different
        // property: below the implication it only arms once the consequent
        // starts, which for `|=>` is a cycle after the attempt began.
        //
        // Covering the trigger is expressible even though a trigger is the union
        // of every attempt's antecedent, because the abort's scope is that whole
        // union. A monitor has exactly one trigger — `lowerProperty`'s
        // implication arm refuses a consequent carrying one, and
        // `lowerPropJunction`, `lowerFollowedBy` and the disjunctive half of
        // `lowerPropShift` all refuse a trigger-carrying operand — so an abort
        // above a trigger is above the monitor's only trigger, and every attempt
        // of it is covered — uniformly, though not collectively: `Monitor.State`
        // resolves each of them separately, and seeds nothing from any.
        //
        // A multi-cycle antecedent is covered too, now that the trigger counts
        // its attempts. An attempt caught halfway through matching owes a verdict
        // of its own, which `Monitor.trigger_window` is what recovers: `step`
        // resolves every attempt with a live antecedent thread, one verdict each,
        // and clears their threads so that none completes an antecedent later and
        // imposes an obligation on an attempt already resolved. Both halves of
        // what this used to refuse for.
        //
        // Nothing is checked here. An unbounded antecedent is refused in
        // `assertion` whether an abort covers it or not — the count is wrong
        // without a bound either way — and by the time a window is sized the
        // trigger is known bounded. What the abort does change is the *size*: a
        // fixed-length antecedent needs no ids to count matches, because they
        // cannot end together, and does need them to count the attempts still
        // matching. `coversTrigger` is how the sizing there learns that.
        const covers_trigger = plan.trigger != null;

        // Everything lowered so far is what the abort guards.
        const mask = try maskAlloc(gpa, plan.obligation.words);
        errdefer gpa.free(mask);
        for (0..plan.obligation.state_count) |state| Nfa.bitSet(mask, @intCast(state));

        try plan.aborts.append(gpa, .{
            .cond = try builder.rootOf(@enumFromInt(data.lhs)),
            .effect = switch (tag) {
                .prop_accept_on, .prop_sync_accept_on => .accept,
                else => .reject,
            },
            .scope = .attempt,
            .node = node,
            .covers_trigger = covers_trigger,
            .sampled = switch (tag) {
                .prop_sync_accept_on, .prop_sync_reject_on => true,
                else => false,
            },
            .mask = mask,
        });
        return plan;
    }

    /// A copy of `source` that the caller owns.
    fn copyFragment(builder: *Builder, source: *const Fragment) Allocator.Error!Fragment {
        var fragment = try Fragment.init(builder.gpa, source.state_count);
        fragment.blit(source, 0);
        @memcpy(fragment.first, source.first);
        @memcpy(fragment.last, source.last);
        return fragment;
    }

    /// `operand`, entered `lo..hi` cycles into the attempt.
    ///
    /// A chain of `hi` unguarded filler states the attempt spends its first
    /// cycles in, with `operand` blitted above them and entered from every
    /// offset in the range. Close to `lowerDelay` against a one-cycle `true`,
    /// with two differences that matter: offset 0 enters `operand` directly
    /// rather than fusing a pair state onto a left match there is none of, and
    /// `operand`'s states stay contiguous at a known base, which is what lets an
    /// abort mask below the shift survive it.
    fn shift(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        lo: u32,
        hi: u32,
    ) Error!Fragment {
        std.debug.assert(lo <= hi);
        std.debug.assert(hi != Ir.unbounded);
        if (hi == 0) return builder.copyFragment(operand);

        const total = try builder.checkLimit(node, @as(u64, hi) + operand.state_count);
        var fragment = try Fragment.init(builder.gpa, total);
        errdefer fragment.deinit(builder.gpa);

        // The filler chain. `Fragment.init` guards every state with `always`,
        // which is what a filler cycle is.
        for (0..hi - 1) |index| {
            Nfa.bitSet(fragment.row(@intCast(index)), @intCast(index + 1));
        }
        fragment.blit(operand, hi);

        Nfa.bitSet(fragment.first, 0);
        if (lo == 0) blitMask(fragment.first, operand.first, hi);
        var offset = @max(lo, 1);
        while (offset <= hi) : (offset += 1) {
            blitMask(fragment.row(offset - 1), operand.first, hi);
        }
        blitMask(fragment.last, operand.last, hi);
        return fragment;
    }

    // -- Lowering ------------------------------------------------------------

    fn lower(builder: *Builder, node: Ir.Node.Index) Error!Fragment {
        const ir = builder.ir;
        const data = ir.dataOf(node);
        switch (ir.tagOf(node)) {
            .seq_boolean => {
                const guard = try builder.guardOf(@enumFromInt(data.lhs), false);
                var fragment = try Fragment.init(builder.gpa, 1);
                fragment.guard[0] = guard;
                Nfa.bitSet(fragment.first, 0);
                Nfa.bitSet(fragment.last, 0);
                return fragment;
            },
            .seq_or => {
                var left = try builder.lower(@enumFromInt(data.lhs));
                defer left.deinit(builder.gpa);
                var right = try builder.lower(@enumFromInt(data.rhs));
                defer right.deinit(builder.gpa);
                return builder.lowerOr(node, &left, &right);
            },
            .seq_and => {
                var left = try builder.lower(@enumFromInt(data.lhs));
                defer left.deinit(builder.gpa);
                var right = try builder.lower(@enumFromInt(data.rhs));
                defer right.deinit(builder.gpa);
                return builder.lowerAnd(node, &left, &right);
            },
            .seq_intersect => {
                var left = try builder.lower(@enumFromInt(data.lhs));
                defer left.deinit(builder.gpa);
                var right = try builder.lower(@enumFromInt(data.rhs));
                defer right.deinit(builder.gpa);
                return builder.lowerIntersect(node, &left, &right);
            },
            .seq_throughout => {
                // The left operand is a boolean, so it contributes a guard
                // rather than a fragment, and the result is the operand's own
                // fragment rewritten in place — there is nothing to combine.
                const exp = try builder.guardOf(@enumFromInt(data.lhs), false);
                var operand = try builder.lower(@enumFromInt(data.rhs));
                errdefer operand.deinit(builder.gpa);
                try builder.lowerThroughout(&operand, exp);
                return operand;
            },
            .seq_first_match => {
                var operand = try builder.lower(@enumFromInt(data.lhs));
                defer operand.deinit(builder.gpa);
                return builder.lowerFirstMatch(node, &operand);
            },
            .seq_delay => {
                const delay = ir.delayOf(node);
                var right = try builder.lower(delay.right);
                defer right.deinit(builder.gpa);
                var left = try builder.lower(delay.left);
                defer left.deinit(builder.gpa);
                return builder.lowerDelay(node, &left, delay.lo, delay.hi, &right);
            },
            .seq_repeat_consec => {
                const repeat = ir.repeatOf(node);
                var operand = try builder.lower(repeat.seq);
                defer operand.deinit(builder.gpa);
                return builder.lowerRepeat(node, &operand, repeat.lo, repeat.hi);
            },
            .seq_repeat_goto, .seq_repeat_nonconsec => {
                const repeat = ir.repeatOf(node);
                if (ir.tagOf(repeat.seq) != .seq_boolean) {
                    if (builder.diagnostic) |into| into.* = .{ .node = repeat.seq };
                    return error.UnsupportedRepeatOperand;
                }
                const operand: Ir.Node.Index = @enumFromInt(ir.dataOf(repeat.seq).lhs);
                return builder.lowerOccurrences(
                    node,
                    try builder.guardOf(operand, false),
                    try builder.guardOf(operand, true),
                    repeat.lo,
                    repeat.hi,
                    ir.tagOf(node) == .seq_repeat_nonconsec,
                );
            },
            .seq_match => {
                // LRM 16.10 runs a match item list "at the end of any nonempty match
                // of the subsequence", and a match of the operand ends in one of its
                // exit positions -- so the program lands on every state in `last` and
                // nowhere else. The automaton's shape does not change: the operand's
                // own fragment comes back with those states annotated.
                var operand = try builder.lower(@enumFromInt(data.lhs));
                errdefer operand.deinit(builder.gpa);
                try builder.attachMatchItems(node, &operand);
                return operand;
            },
            else => unreachable, // not a sequence-layer tag
        }
    }

    /// Attach a `seq_match` node's assignments to every exit position of its
    /// operand, appending them to the site pool.
    ///
    /// A position may already carry a program: match item lists nest
    /// (`((a, x = 1), y = 2)`), and the inner list is attached first, so the outer
    /// one's sites go after it — which is the order LRM 16.10 asks for, innermost
    /// match ending first. Extending is a re-append rather than an in-place grow
    /// because the pool is shared and a program is a contiguous slice of it; the
    /// duplicated prefix costs a few words on a shape that nests match items and
    /// nothing at all on every other.
    fn attachMatchItems(builder: *Builder, node: Ir.Node.Index, fragment: *Fragment) Error!void {
        const gpa = builder.gpa;
        var assigns: std.ArrayListUnmanaged(Ir.Assign) = .empty;
        defer assigns.deinit(gpa);
        try builder.ir.matchOf(node, &assigns, gpa);
        std.debug.assert(assigns.items.len > 0); // `seqMatch` folds an empty list away

        // Site numbers are dense over the whole monitor and minted here, in the
        // order the nodes are lowered, so the backend's site table is this list.
        const first_site: u32 = @intCast(builder.assign_slot.items.len);
        for (assigns.items) |assign| {
            try builder.assign_slot.append(gpa, assign.slot);
            try builder.assign_handle.append(gpa, assign.handle);
        }

        var exits: BitIterator = .init(fragment.last);
        while (exits.next()) |index| {
            const state: u32 = @intCast(index);
            const held = fragment.program[state];
            // Reserve first: appending from the same list would read through a
            // slice the append itself may have reallocated.
            try builder.sites.ensureUnusedCapacity(gpa, held.len + assigns.items.len);
            const start: u32 = @intCast(builder.sites.items.len);
            for (0..held.len) |offset| {
                const site = builder.sites.items[held.start + offset];
                builder.sites.appendAssumeCapacity(site);
            }
            for (0..assigns.items.len) |offset| {
                builder.sites.appendAssumeCapacity(first_site + @as(u32, @intCast(offset)));
            }
            fragment.program[state] = .{
                .start = start,
                .len = held.len + @as(u32, @intCast(assigns.items.len)),
            };
        }
    }

    fn lowerOr(
        builder: *Builder,
        node: Ir.Node.Index,
        left: *const Fragment,
        right: *const Fragment,
    ) Error!Fragment {
        const total = try builder.checkLimit(node, @as(u64, left.state_count) + right.state_count);
        var fragment = try Fragment.init(builder.gpa, total);
        fragment.blit(left, 0);
        fragment.blit(right, left.state_count);
        blitMask(fragment.first, left.first, 0);
        blitMask(fragment.first, right.first, left.state_count);
        blitMask(fragment.last, left.last, 0);
        blitMask(fragment.last, right.last, left.state_count);
        return fragment;
    }

    /// `exp throughout A`: `exp` holds in every cycle the match spans.
    ///
    /// A fragment is homogeneous, so a guard belongs to its state rather than
    /// to the edges entering it, and a state is occupied exactly in the cycles
    /// its guard holds. Requiring `exp` in every cycle of a match is therefore
    /// a conjunction into every state's guard, with `first`, `follow` and
    /// `last` untouched: no state is added, and the operand's fragment is
    /// rewritten where it stands rather than copied.
    ///
    /// A state whose conjunction is unsatisfiable is simply one no cycle can
    /// enter, which is a case pruning already handles — including the whole
    /// fragment collapsing, for a `throughout` that contradicts its operand
    /// everywhere.
    fn lowerThroughout(builder: *Builder, fragment: *Fragment, exp: Guard.Index) Allocator.Error!void {
        for (fragment.guard) |*guard| guard.* = try builder.conjoin(guard.*, exp);
    }

    /// One cycle that is simultaneously an exit of the left operand and an entry
    /// of the right.
    const Fusion = struct { left_exit: u32, right_entry: u32, guard: Guard.Index };

    /// `left ##[lo:hi] right`, covering both the overlap at delay 0 and the
    /// filler chain from delay 1 up, in one fragment.
    fn lowerDelay(
        builder: *Builder,
        node: Ir.Node.Index,
        left: *const Fragment,
        lo: u32,
        hi: u32,
        right: *const Fragment,
    ) Error!Fragment {
        const gpa = builder.gpa;
        const open = hi == Ir.unbounded;
        const left_count = left.state_count;

        // Delay 0 against a left operand fuses: one cycle is simultaneously the
        // last of the left match and the first of the right, so it needs a
        // state guarded by both.
        var fusions: std.ArrayListUnmanaged(Fusion) = .empty;
        defer fusions.deinit(gpa);
        if (lo == 0) {
            var exits: BitIterator = .init(left.last);
            while (exits.next()) |exit| {
                var entries: BitIterator = .init(right.first);
                while (entries.next()) |entry| {
                    const guard = try builder.conjoin(left.guard[exit], right.guard[entry]);
                    if (guard == .never) continue;
                    try fusions.append(gpa, .{
                        .left_exit = exit,
                        .right_entry = entry,
                        .guard = guard,
                    });
                }
            }
        }

        // Fillers cover delays of 1 and up: a delay of k leaves k - 1 cycles
        // between the two matches.
        const chain = open or hi >= 1;
        const chain_min = @max(lo, 1);
        const fillers: u32 = if (!chain)
            0
        else if (open)
            @max(chain_min - 1, 1)
        else
            hi - 1;

        const total = try builder.checkLimit(
            node,
            @as(u64, left_count) + right.state_count + fusions.items.len + fillers,
        );
        var fragment = try Fragment.init(gpa, total);
        errdefer fragment.deinit(gpa);
        fragment.blit(left, 0);
        fragment.blit(right, left_count);

        const fusion_base = left_count + right.state_count;
        const filler_base = fusion_base + @as(u32, @intCast(fusions.items.len));

        const right_entries = try maskAlloc(gpa, fragment.words);
        defer gpa.free(right_entries);
        blitMask(right_entries, right.first, left_count);

        for (fusions.items, 0..) |fusion, offset| {
            const state = fusion_base + @as(u32, @intCast(offset));
            fragment.guard[state] = fusion.guard;
            // A fused state stands in for the right operand's entry state, so
            // it leaves along that state's edges.
            blitMask(fragment.row(state), right.rowConst(fusion.right_entry), left_count);
            if (Nfa.bitIsSet(left.first, fusion.left_exit)) Nfa.bitSet(fragment.first, state);
            if (Nfa.bitIsSet(right.last, fusion.right_entry)) Nfa.bitSet(fragment.last, state);
            // The fused cycle ends the left operand's match and begins the right's,
            // so it owes both positions' assignments, left first (LRM 16.10 and
            // F.5.5's `##0` clause, which flows the left's output context into the
            // right). A read on the right of a slot the left writes here would need
            // the guard cube split around the assignment instead, which is what
            // Error.LocalFusedRead refuses.
            try fragment.fuseProgram(
                &builder.sites,
                gpa,
                state,
                left.program[fusion.left_exit],
                right.program[fusion.right_entry],
            );
        }
        for (0..left_count) |index| {
            const state: u32 = @intCast(index);
            for (fusions.items, 0..) |fusion, offset| {
                if (Nfa.bitIsSet(left.rowConst(state), fusion.left_exit)) {
                    Nfa.bitSet(fragment.row(state), fusion_base + @as(u32, @intCast(offset)));
                }
            }
        }

        var filler: u32 = 0;
        while (filler + 1 < fillers) : (filler += 1) {
            Nfa.bitSet(fragment.row(filler_base + filler), filler_base + filler + 1);
        }
        if (open and fillers > 0) {
            Nfa.bitSet(fragment.row(filler_base + fillers - 1), filler_base + fillers - 1);
        }

        blitMask(fragment.first, left.first, 0);
        // Left states keep their indices, so its exit mask needs no shift.
        var exits: BitIterator = .init(left.last);
        while (exits.next()) |exit| {
            if (fillers > 0) Nfa.bitSet(fragment.row(exit), filler_base);
            if (chain and chain_min == 1) Nfa.bitUnion(fragment.row(exit), right_entries);
        }

        if (fillers > 0) {
            if (open) {
                Nfa.bitUnion(fragment.row(filler_base + fillers - 1), right_entries);
            } else {
                var reached: u32 = @max(chain_min - 1, 1);
                while (reached <= fillers) : (reached += 1) {
                    Nfa.bitUnion(fragment.row(filler_base + reached - 1), right_entries);
                }
            }
        }

        blitMask(fragment.last, right.last, left_count);
        return fragment;
    }

    /// `A[*lo:hi]`, unrolled. An open upper bound loops the last copy back on
    /// itself rather than unrolling forever.
    fn lowerRepeat(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        lo: u32,
        hi: u32,
    ) Error!Fragment {
        const gpa = builder.gpa;
        const open = hi == Ir.unbounded;
        const copies: u32 = if (open) lo else hi;
        const stride = operand.state_count;

        const total = try builder.checkLimit(node, @as(u64, copies) * stride);
        var fragment = try Fragment.init(gpa, total);
        errdefer fragment.deinit(gpa);
        if (total == 0) return fragment; // the operand can never match

        for (0..copies) |copy| fragment.blit(operand, @intCast(copy * stride));

        const entries = try maskAlloc(gpa, fragment.words);
        defer gpa.free(entries);

        var copy: u32 = 0;
        while (copy + 1 < copies) : (copy += 1) {
            const base = copy * stride;
            @memset(entries, 0);
            blitMask(entries, operand.first, base + stride);
            var exits: BitIterator = .init(operand.last);
            while (exits.next()) |exit| Nfa.bitUnion(fragment.row(base + exit), entries);
        }

        const last_base = (copies - 1) * stride;
        if (open) {
            @memset(entries, 0);
            blitMask(entries, operand.first, last_base);
            var exits: BitIterator = .init(operand.last);
            while (exits.next()) |exit| Nfa.bitUnion(fragment.row(last_base + exit), entries);
        }

        blitMask(fragment.first, operand.first, 0);
        if (open) {
            blitMask(fragment.last, operand.last, last_base);
        } else {
            var accepted = lo;
            while (accepted <= hi) : (accepted += 1) {
                blitMask(fragment.last, operand.last, (accepted - 1) * stride);
            }
        }
        return fragment;
    }

    /// `b[->lo:hi]` and `b[=lo:hi]`. Occurrence `i` is a state guarded by `b`,
    /// preceded by a self-looping gap state guarded by `¬b`. Non-consecutive
    /// repetition adds a trailing gap the match may sit in.
    ///
    /// An open upper bound is handled by tacking a loop on the end rather than
    /// unrolling: goto gets one more occurrence/gap pair that cycles, and
    /// non-consecutive repetition — which after its `lo`th occurrence accepts
    /// *every* later cycle — gets an unguarded sink.
    fn lowerOccurrences(
        builder: *Builder,
        node: Ir.Node.Index,
        present: Guard.Index,
        absent: Guard.Index,
        lo: u32,
        hi: u32,
        trailing: bool,
    ) Error!Fragment {
        const open = hi == Ir.unbounded;
        const occurrences: u32 = if (open) lo else hi;
        const extra: u32 = if (open) (if (trailing) 1 else 2) else @intFromBool(trailing);

        const total = try builder.checkLimit(node, @as(u64, occurrences) * 2 + extra);
        var fragment = try Fragment.init(builder.gpa, total);
        errdefer fragment.deinit(builder.gpa);

        // Occurrences at [0, occurrences), their gaps next, then the tail.
        const gap_base = occurrences;
        const tail_base = 2 * occurrences;
        for (0..occurrences) |index| {
            const occurrence: u32 = @intCast(index);
            fragment.guard[occurrence] = present;
            fragment.guard[gap_base + occurrence] = absent;
            Nfa.bitSet(fragment.row(gap_base + occurrence), gap_base + occurrence);
            Nfa.bitSet(fragment.row(gap_base + occurrence), occurrence);
            if (occurrence + 1 < occurrences) {
                Nfa.bitSet(fragment.row(occurrence), gap_base + occurrence + 1);
                Nfa.bitSet(fragment.row(occurrence), occurrence + 1);
            }
        }
        Nfa.bitSet(fragment.first, 0);
        Nfa.bitSet(fragment.first, gap_base);
        for (lo - 1..occurrences) |index| Nfa.bitSet(fragment.last, @intCast(index));

        if (open and !trailing) {
            fragment.guard[tail_base] = present;
            fragment.guard[tail_base + 1] = absent;
            for ([_]u32{ occurrences - 1, tail_base, tail_base + 1 }) |source| {
                Nfa.bitSet(fragment.row(source), tail_base);
                Nfa.bitSet(fragment.row(source), tail_base + 1);
            }
            Nfa.bitSet(fragment.last, tail_base);
        } else if (open) {
            // Past the `lo`th occurrence every cycle ends a match, whatever it
            // holds: a later occurrence just raises the count.
            fragment.guard[tail_base] = .always;
            Nfa.bitSet(fragment.row(occurrences - 1), tail_base);
            Nfa.bitSet(fragment.row(tail_base), tail_base);
            Nfa.bitSet(fragment.last, tail_base);
        } else if (trailing) {
            fragment.guard[tail_base] = absent;
            Nfa.bitSet(fragment.row(tail_base), tail_base);
            for (lo - 1..occurrences) |index| {
                Nfa.bitSet(fragment.row(@intCast(index)), tail_base);
            }
            Nfa.bitSet(fragment.last, tail_base);
        }
        return fragment;
    }

    /// A reachable state of the product, and the pair it stands for.
    const Pair = struct { left: u32, right: u32, guard: Guard.Index };

    /// Where a cycle can take a product state: a pair before its guard is
    /// known.
    const Successor = struct { left: u32, right: u32 };

    /// Stands in for an operand of an `and` that has already ended, in place of
    /// a state of it. Not a state index: no operand has this many states, since
    /// `Options.limit` is a `u32` count that has already been checked.
    ///
    /// Only `and` produces one. `intersect` ends both sides in the same cycle,
    /// so neither is ever carried past the other.
    const finished: u32 = std.math.maxInt(u32);

    /// Marks a product pair the guard conjunction ruled out, so it is looked up
    /// once rather than reconsidered from every predecessor.
    const rejected: u32 = std.math.maxInt(u32);

    fn pairKey(left: u32, right: u32) u64 {
        return (@as(u64, left) << 32) | right;
    }

    /// `A intersect B`: the product restricted to reachable pairs, with the two
    /// guards conjoined. Length matching is exactly what a synchronised product
    /// is, so there is nothing else to say.
    fn lowerIntersect(
        builder: *Builder,
        node: Ir.Node.Index,
        left: *const Fragment,
        right: *const Fragment,
    ) Error!Fragment {
        const gpa = builder.gpa;

        var index: std.AutoHashMapUnmanaged(u64, u32) = .empty;
        defer index.deinit(gpa);
        var pairs: std.ArrayListUnmanaged(Pair) = .empty;
        defer pairs.deinit(gpa);

        var left_entries: BitIterator = .init(left.first);
        while (left_entries.next()) |left_state| {
            var right_entries: BitIterator = .init(right.first);
            while (right_entries.next()) |right_state| {
                _ = try builder.discover(node, &index, &pairs, left, right, left_state, right_state);
            }
        }
        const seed_count = pairs.items.len;

        var frontier: usize = 0;
        while (frontier < pairs.items.len) : (frontier += 1) {
            const pair = pairs.items[frontier];
            var left_next: BitIterator = .init(left.rowConst(pair.left));
            while (left_next.next()) |left_state| {
                var right_next: BitIterator = .init(right.rowConst(pair.right));
                while (right_next.next()) |right_state| {
                    _ = try builder.discover(node, &index, &pairs, left, right, left_state, right_state);
                }
            }
        }

        const total = try builder.checkLimit(node, pairs.items.len);
        var fragment = try Fragment.init(gpa, total);
        errdefer fragment.deinit(gpa);

        for (pairs.items, 0..) |pair, offset| {
            const state: u32 = @intCast(offset);
            fragment.guard[state] = pair.guard;
            // A product state stands for one position of each operand at once, so
            // it owes both their assignments, left first (LRM 16.10; F.5.5 gives the
            // two operands the same input context, and Error.LocalCrossRead refuses
            // the one shape where a shared valuation would be visible across the
            // fork). `finished` is not a position and carries no program.
            try fragment.fuseProgram(
                &builder.sites,
                gpa,
                state,
                if (pair.left == finished) Nfa.Program.none else left.program[pair.left],
                if (pair.right == finished) Nfa.Program.none else right.program[pair.right],
            );
            var left_next: BitIterator = .init(left.rowConst(pair.left));
            while (left_next.next()) |left_state| {
                var right_next: BitIterator = .init(right.rowConst(pair.right));
                while (right_next.next()) |right_state| {
                    const target = index.get(pairKey(left_state, right_state)) orelse continue;
                    if (target != rejected) Nfa.bitSet(fragment.row(state), target);
                }
            }
            if (offset < seed_count) Nfa.bitSet(fragment.first, state);
            if (Nfa.bitIsSet(left.last, pair.left) and Nfa.bitIsSet(right.last, pair.right)) {
                Nfa.bitSet(fragment.last, state);
            }
        }
        return fragment;
    }

    fn discover(
        builder: *Builder,
        node: Ir.Node.Index,
        index: *std.AutoHashMapUnmanaged(u64, u32),
        pairs: *std.ArrayListUnmanaged(Pair),
        left: *const Fragment,
        right: *const Fragment,
        left_state: u32,
        right_state: u32,
    ) Error!?u32 {
        const entry = try index.getOrPut(builder.gpa, pairKey(left_state, right_state));
        if (entry.found_existing) {
            return if (entry.value_ptr.* == rejected) null else entry.value_ptr.*;
        }

        const guard = try builder.productGuard(left, right, left_state, right_state);
        if (guard == .never) {
            // An unsatisfiable product state can never be entered. Record the
            // rejection rather than a state, so the pair is not re-derived from
            // every predecessor that reaches it.
            entry.value_ptr.* = rejected;
            return null;
        }
        if (pairs.items.len >= builder.options.limit) {
            return builder.fail(node, @as(u64, builder.options.limit) + 1);
        }

        entry.value_ptr.* = @intCast(pairs.items.len);
        try pairs.append(builder.gpa, .{
            .left = left_state,
            .right = right_state,
            .guard = guard,
        });
        return entry.value_ptr.*;
    }

    /// The guard on a product state: the conjunction of the two, or — for an
    /// operand of an `and` that has already ended — just the live one's, since
    /// a side that has finished matching constrains no further cycle.
    fn productGuard(
        builder: *Builder,
        left: *const Fragment,
        right: *const Fragment,
        left_state: u32,
        right_state: u32,
    ) Allocator.Error!Guard.Index {
        if (left_state == finished) return right.guard[right_state];
        if (right_state == finished) return left.guard[left_state];
        return builder.conjoin(left.guard[left_state], right.guard[right_state]);
    }

    /// Where a cycle can take an `and` product state.
    ///
    /// While both sides are live they advance together. A side sitting on one
    /// of its own exits may instead have *ended there*, which leaves the other
    /// to run on alone — so an exit contributes that continuation as well as
    /// the paired one, and the two readings of the trace are both explored
    /// rather than chosen between here.
    fn andSuccessors(
        gpa: Allocator,
        left: *const Fragment,
        right: *const Fragment,
        pair: Pair,
        out: *std.ArrayListUnmanaged(Successor),
    ) Allocator.Error!void {
        out.clearRetainingCapacity();

        if (pair.left == finished) {
            var next: BitIterator = .init(right.rowConst(pair.right));
            while (next.next()) |state| try out.append(gpa, .{ .left = finished, .right = state });
            return;
        }
        if (pair.right == finished) {
            var next: BitIterator = .init(left.rowConst(pair.left));
            while (next.next()) |state| try out.append(gpa, .{ .left = state, .right = finished });
            return;
        }

        var left_next: BitIterator = .init(left.rowConst(pair.left));
        while (left_next.next()) |left_state| {
            var right_next: BitIterator = .init(right.rowConst(pair.right));
            while (right_next.next()) |right_state| {
                try out.append(gpa, .{ .left = left_state, .right = right_state });
            }
        }
        if (Nfa.bitIsSet(right.last, pair.right)) {
            var next: BitIterator = .init(left.rowConst(pair.left));
            while (next.next()) |state| try out.append(gpa, .{ .left = state, .right = finished });
        }
        if (Nfa.bitIsSet(left.last, pair.left)) {
            var next: BitIterator = .init(right.rowConst(pair.right));
            while (next.next()) |state| try out.append(gpa, .{ .left = finished, .right = state });
        }
    }

    /// `A and B`: both match from the same cycle, and the match ends at the
    /// later of the two ends.
    ///
    /// The pair states `intersect` builds say both sides are live in the same
    /// cycle, which is the whole of `intersect` because its two ends coincide.
    /// `and` also has to say that one side has *already ended* while the other
    /// runs on, and a position automaton cannot idle it: every state consumes a
    /// cycle under its own guard, so a side still carried is a side still
    /// constraining the trace. It is dropped from the state instead. A state is
    /// a pair from `A × B`, or one operand's state alongside `finished` — a
    /// copy of each operand to run on alone in, `|A| · |B| + |A| + |B|` states
    /// before reachability takes its cut.
    ///
    /// Homogeneity survives, which is what keeps this a fragment: a pair state
    /// is entered under the conjunction of the two guards and a half-finished
    /// one under the live side's guard alone, so either way the guard is a
    /// function of the state and not of the edge. So does ε-freedom — finishing
    /// is not a transition but a property of the state a transition leaves.
    fn lowerAnd(
        builder: *Builder,
        node: Ir.Node.Index,
        left: *const Fragment,
        right: *const Fragment,
    ) Error!Fragment {
        const gpa = builder.gpa;

        var index: std.AutoHashMapUnmanaged(u64, u32) = .empty;
        defer index.deinit(gpa);
        var pairs: std.ArrayListUnmanaged(Pair) = .empty;
        defer pairs.deinit(gpa);
        var successors: std.ArrayListUnmanaged(Successor) = .empty;
        defer successors.deinit(gpa);

        // Both sides start in the attempt's first cycle, and neither can have
        // finished before it has matched one, so every entry is a live pair.
        var left_entries: BitIterator = .init(left.first);
        while (left_entries.next()) |left_state| {
            var right_entries: BitIterator = .init(right.first);
            while (right_entries.next()) |right_state| {
                _ = try builder.discover(node, &index, &pairs, left, right, left_state, right_state);
            }
        }
        const seed_count = pairs.items.len;

        var frontier: usize = 0;
        while (frontier < pairs.items.len) : (frontier += 1) {
            try andSuccessors(gpa, left, right, pairs.items[frontier], &successors);
            for (successors.items) |target| {
                _ = try builder.discover(node, &index, &pairs, left, right, target.left, target.right);
            }
        }

        const total = try builder.checkLimit(node, pairs.items.len);
        var fragment = try Fragment.init(gpa, total);
        errdefer fragment.deinit(gpa);

        for (pairs.items, 0..) |pair, offset| {
            const state: u32 = @intCast(offset);
            fragment.guard[state] = pair.guard;
            // A product state stands for one position of each operand at once, so
            // it owes both their assignments, left first (LRM 16.10; F.5.5 gives the
            // two operands the same input context, and Error.LocalCrossRead refuses
            // the one shape where a shared valuation would be visible across the
            // fork). `finished` is not a position and carries no program.
            try fragment.fuseProgram(
                &builder.sites,
                gpa,
                state,
                if (pair.left == finished) Nfa.Program.none else left.program[pair.left],
                if (pair.right == finished) Nfa.Program.none else right.program[pair.right],
            );
            try andSuccessors(gpa, left, right, pair, &successors);
            for (successors.items) |target| {
                const next = index.get(pairKey(target.left, target.right)) orelse continue;
                if (next != rejected) Nfa.bitSet(fragment.row(state), next);
            }
            if (offset < seed_count) Nfa.bitSet(fragment.first, state);
            // The match ends where the later side does, so this cycle ends it
            // if the live sides all end here — a side already `finished` ended
            // in an earlier one.
            const left_ends = pair.left == finished or Nfa.bitIsSet(left.last, pair.left);
            const right_ends = pair.right == finished or Nfa.bitIsSet(right.last, pair.right);
            if (left_ends and right_ends) Nfa.bitSet(fragment.last, state);
        }
        return fragment;
    }

    /// A determinised state of a `first_match`: where the operand can be this
    /// cycle, and the cube the cycle was entered under.
    const Reached = struct { subset: u32, guard: Guard.Index };

    /// Subsets of an operand's states, interned by content so a determinised
    /// state can name one by a dense index.
    ///
    /// Content-addressed the way the cube table is, and for the same reason: a
    /// map keyed by a slice of `pool` would be invalidated the moment `pool`
    /// grew, so a hash heads a chain threaded through `chain` and a candidate
    /// is compared against the chain exactly.
    const Subsets = struct {
        /// Words per mask — over the operand's states, not the result's.
        words: u32,
        /// Flat `count · words` matrix of the interned masks.
        pool: std.ArrayListUnmanaged(Word) = .empty,
        bucket: std.AutoHashMapUnmanaged(u64, u32) = .empty,
        chain: std.ArrayListUnmanaged(u32) = .empty,

        fn deinit(subsets: *Subsets, gpa: Allocator) void {
            subsets.pool.deinit(gpa);
            subsets.bucket.deinit(gpa);
            subsets.chain.deinit(gpa);
        }

        fn maskOf(subsets: *const Subsets, subset: u32) []const Word {
            return subsets.pool.items[@as(usize, subset) * subsets.words ..][0..subsets.words];
        }

        fn intern(subsets: *Subsets, gpa: Allocator, mask: []const Word) Allocator.Error!u32 {
            const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(mask));
            const bucket = try subsets.bucket.getOrPut(gpa, hash);
            if (!bucket.found_existing) bucket.value_ptr.* = end_of_chain;

            var subset = bucket.value_ptr.*;
            while (subset != end_of_chain) : (subset = subsets.chain.items[subset]) {
                if (std.mem.eql(Word, subsets.maskOf(subset), mask)) return subset;
            }

            const fresh: u32 = @intCast(subsets.chain.items.len);
            try subsets.pool.appendSlice(gpa, mask);
            try subsets.chain.append(gpa, bucket.value_ptr.*);
            bucket.value_ptr.* = fresh;
            return fresh;
        }
    };

    /// Everything one `first_match` lowering carries: the determinised states
    /// found so far, their successors, and the scratch the cell enumeration
    /// runs on.
    const Determinisation = struct {
        /// Interned subsets of the operand's states.
        subsets: Subsets,
        /// The states found so far, and a content index over them keyed by
        /// `pairKey(subset, cube)`.
        states: std.ArrayListUnmanaged(Reached) = .empty,
        index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
        /// Successors, flattened: state `s` owns `edges[start[s]..start[s + 1]]`.
        /// Recorded as they are found rather than recomputed in a second pass
        /// the way the products do it, because a cell costs cube interning
        /// where a product's successor costs a bit iteration.
        edges: std.ArrayListUnmanaged(u32) = .empty,
        start: std.ArrayListUnmanaged(u32) = .empty,
        /// Where the operand can go this cycle before its guards are consulted,
        /// and what one cell of the valuation space leaves of that.
        candidates: []Word,
        subset: []Word,
        /// The distinct guards over `candidates`, and the literals of the one
        /// being split on — copied out, because interning a cube can move the
        /// pool they would otherwise be read from.
        guards: std.ArrayListUnmanaged(Guard.Index) = .empty,
        literals: std.ArrayListUnmanaged(Literal) = .empty,
        /// The cells cut so far, and the buffer the next split fills.
        cells: std.ArrayListUnmanaged(Guard.Index) = .empty,
        split: std.ArrayListUnmanaged(Guard.Index) = .empty,
        /// Membership over whichever of those is being filled.
        seen: std.AutoHashMapUnmanaged(u32, void) = .empty,

        fn init(gpa: Allocator, words: u32) Allocator.Error!Determinisation {
            const candidates = try maskAlloc(gpa, words);
            errdefer gpa.free(candidates);
            return .{
                .subsets = .{ .words = words },
                .candidates = candidates,
                .subset = try maskAlloc(gpa, words),
            };
        }

        fn deinit(into: *Determinisation, gpa: Allocator) void {
            into.subsets.deinit(gpa);
            into.states.deinit(gpa);
            into.index.deinit(gpa);
            into.edges.deinit(gpa);
            into.start.deinit(gpa);
            gpa.free(into.candidates);
            gpa.free(into.subset);
            into.guards.deinit(gpa);
            into.literals.deinit(gpa);
            into.cells.deinit(gpa);
            into.split.deinit(gpa);
            into.seen.deinit(gpa);
        }

        /// Keep a cell, unless it is unsatisfiable or already in `split`.
        fn record(into: *Determinisation, gpa: Allocator, cell: Guard.Index) Allocator.Error!void {
            if (cell == .never) return;
            if ((try into.seen.getOrPut(gpa, @intFromEnum(cell))).found_existing) return;
            try into.split.append(gpa, cell);
        }
    };

    /// Cut the valuation space into the cells the guards on `candidates`
    /// distinguish, leaving them in `into.cells`.
    ///
    /// `candidates` is where the operand can be next cycle before its guards
    /// are consulted. Which of them it is actually in is decided by those
    /// guards, so two valuations agreeing on every one of them leave the
    /// operand in the same place — the cells of that agreement are exactly the
    /// distinct successors.
    ///
    /// A cell is `⋀ g ∧ ⋀ ¬h` over the guards that hold and those that do not,
    /// and the complement of a cube is a disjunction rather than a cube. So a
    /// cell is enumerated as one branch per guard holding and one per *literal*
    /// of it failing: those branches still cover the space, and each one is a
    /// cube, which is what a state's guard has to be. Two of them may overlap,
    /// and that costs a duplicate state and nothing else — a valuation in both
    /// is in one cell, so both branches leave the operand in the same place and
    /// behave identically from there on.
    ///
    /// Nothing carries a subset down the enumeration, because a cell's cube
    /// already fixes the truth of every candidate guard: a guard holds in a
    /// cell exactly when the cell entails it. `reach` reads the subset back off
    /// the cube that way.
    fn firstMatchCells(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        candidates: []const Word,
        into: *Determinisation,
    ) Error!void {
        const gpa = builder.gpa;

        // The distinct guards, rather than one split per state carrying one:
        // a filler chain is a run of states all guarded by `always`, and it
        // should cost nothing.
        into.guards.clearRetainingCapacity();
        into.seen.clearRetainingCapacity();
        var candidate: BitIterator = .init(candidates);
        while (candidate.next()) |state| {
            const guard = operand.guard[state];
            // `always` is in every cell and `never` in none, so neither of them
            // tells two valuations apart.
            if (guard == .always or guard == .never) continue;
            if ((try into.seen.getOrPut(gpa, @intFromEnum(guard))).found_existing) continue;
            try into.guards.append(gpa, guard);
        }

        into.cells.clearRetainingCapacity();
        try into.cells.append(gpa, .always);
        for (into.guards.items) |guard| {
            into.literals.clearRetainingCapacity();
            try into.literals.appendSlice(gpa, builder.literalsOf(@intFromEnum(guard)));

            into.split.clearRetainingCapacity();
            into.seen.clearRetainingCapacity();
            for (into.cells.items) |cell| {
                try into.record(gpa, try builder.conjoin(cell, guard));
                for (into.literals.items) |literal| {
                    try into.record(gpa, try builder.conjoinNegated(cell, literal));
                }
            }
            // Every surviving cell becomes at most one state, so the projected
            // size is checked here as well as against the state list.
            if (into.split.items.len > builder.options.limit) {
                return builder.fail(node, into.split.items.len);
            }
            std.mem.swap(std.ArrayListUnmanaged(Guard.Index), &into.cells, &into.split);
        }
    }

    /// The determinised state `into.subset` and `cell` name, discovered if it
    /// is new. The pairing is what keeps the result a fragment — see
    /// `lowerFirstMatch`.
    fn subsetState(
        builder: *Builder,
        node: Ir.Node.Index,
        into: *Determinisation,
        cell: Guard.Index,
    ) Error!u32 {
        const gpa = builder.gpa;

        const subset = try into.subsets.intern(gpa, into.subset);
        const entry = try into.index.getOrPut(gpa, pairKey(subset, @intFromEnum(cell)));
        if (entry.found_existing) return entry.value_ptr.*;
        if (into.states.items.len >= builder.options.limit) {
            return builder.fail(node, @as(u64, builder.options.limit) + 1);
        }

        entry.value_ptr.* = @intCast(into.states.items.len);
        try into.states.append(gpa, .{ .subset = subset, .guard = cell });
        return entry.value_ptr.*;
    }

    /// Where one cell leaves the operand, as a determinised state.
    ///
    /// Null for a cell that leaves it nowhere to be: that is an attempt dying,
    /// which a position automaton spells as having no state to enter rather
    /// than as a state of its own. `determinise` is the one caller that wants
    /// the state, and reaches for `subsetState` itself — `into.subset` is left
    /// empty here, which is the subset it wants named.
    fn reach(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        into: *Determinisation,
        candidates: []const Word,
        cell: Guard.Index,
    ) Error!?u32 {
        @memset(into.subset, 0);
        var candidate: BitIterator = .init(candidates);
        while (candidate.next()) |state| {
            if (builder.entails(cell, operand.guard[state])) Nfa.bitSet(into.subset, state);
        }
        if (!Nfa.bitAny(into.subset)) return null;
        return try builder.subsetState(node, into, cell);
    }

    /// `first_match(A)`: the shortest match of `A` from each start cycle, with
    /// every longer match of the same attempt suppressed.
    ///
    /// Every other combinator here is a language operation on matches taken one
    /// at a time, which is why a union or a product of fragments is one. This
    /// one is not. A match survives only if no *shorter* match of the operand
    /// covers the same cycles, which is the set difference `L \ L·Σ⁺` — a
    /// complement, and complementing an ε-free automaton is determinising it.
    /// Whether a shorter match ended is a property of *where the operand could
    /// be*, and no single position knows that; one thread can be short of an
    /// exit while another sits on one.
    ///
    /// So this determinises. A state is a subset of the operand's states paired
    /// with the cube the cycle was entered under, and the pairing is what keeps
    /// the result a fragment: a subset on its own would have to be entered
    /// under `⋀ g ∧ ⋀ ¬h`, which is not a cube and so not a guard, so a subset
    /// is split across the cells its predecessor's guards cut the valuation
    /// space into — see `firstMatchCells`. Homogeneity survives, because a
    /// state's guard is the cell it stands for; so does ε-freedom, because a
    /// subset consumes exactly one cycle like any other state.
    ///
    /// Suppression is then one line. A subset holding an exit of the operand
    /// has just matched, so it is an exit here and has *no successors at all*:
    /// every longer match of that attempt dies with the states that would have
    /// carried it. Attempts do not interfere, even sharing one state vector —
    /// two attempts meeting in the same state agree on everything the operand
    /// could still do, so they end together and both end on their own first
    /// match.
    ///
    /// `nullable` stays identically false: the shortest match of a language
    /// with no empty match is not empty. And every match kept is one of the
    /// operand's, so the longest is no longer than the operand's longest and
    /// no `depth` gets worse — see `maxLength`.
    ///
    /// The cost is the determinisation's: `|subsets| · |cells|` states in the
    /// worst case, capped by `Options.limit` like every other blow-up here.
    /// Reachability takes most of it back, because a subset is only ever
    /// explored while no match has ended yet, which is a short window.
    fn lowerFirstMatch(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
    ) Error!Fragment {
        const gpa = builder.gpa;
        if (operand.state_count == 0) return Fragment.init(gpa, 0);

        var into: Determinisation = try .init(gpa, operand.words);
        defer into.deinit(gpa);

        // The attempt's first cycle: the operand is wherever its entry states
        // and this cycle's valuation leave it.
        try builder.firstMatchCells(node, operand, operand.first, &into);
        for (into.cells.items) |cell| {
            _ = try builder.reach(node, operand, &into, operand.first, cell);
        }
        const seed_count = into.states.items.len;

        var frontier: usize = 0;
        while (frontier < into.states.items.len) : (frontier += 1) {
            try into.start.append(gpa, @intCast(into.edges.items.len));
            const subset = into.states.items[frontier].subset;
            // A match of the operand ends here, so the attempt is over and this
            // state leads nowhere. This is the whole of `first_match`.
            if (Nfa.bitOverlaps(into.subsets.maskOf(subset), operand.last)) continue;

            @memset(into.candidates, 0);
            var live: BitIterator = .init(into.subsets.maskOf(subset));
            while (live.next()) |state| Nfa.bitUnion(into.candidates, operand.rowConst(state));

            try builder.firstMatchCells(node, operand, into.candidates, &into);
            for (into.cells.items) |cell| {
                const target = try builder.reach(node, operand, &into, into.candidates, cell) orelse
                    continue;
                try into.edges.append(gpa, target);
            }
        }
        try into.start.append(gpa, @intCast(into.edges.items.len));

        const total = try builder.checkLimit(node, into.states.items.len);
        var fragment = try Fragment.init(gpa, total);
        errdefer fragment.deinit(gpa);

        for (into.states.items, 0..) |state, offset| {
            const index: u32 = @intCast(offset);
            fragment.guard[index] = state.guard;
            for (into.edges.items[into.start.items[offset]..into.start.items[offset + 1]]) |target| {
                Nfa.bitSet(fragment.row(index), target);
            }
            if (offset < seed_count) Nfa.bitSet(fragment.first, index);
            if (Nfa.bitOverlaps(into.subsets.maskOf(state.subset), operand.last)) {
                Nfa.bitSet(fragment.last, index);
            }
        }
        return fragment;
    }

    /// The *total* determinisation of `operand`: the same subset construction
    /// `lowerFirstMatch` runs, with the attempts that died kept as states and
    /// no truncation at the ones that matched.
    ///
    /// `first_match` only ever has to know whether a match has ended *yet*, so
    /// a dying attempt can be spelled the way a position automaton always
    /// spells one — as having no state to enter. Complementation cannot. `not`,
    /// and every unbounded operator that has to decide an obligation has
    /// finally failed, read the *absence* of a match as a verdict, and an
    /// absence carried as a missing edge is indistinguishable from an attempt
    /// that was never seeded: the state vector holds nothing either way. So the
    /// death has to be a state, and every valuation has to lead somewhere.
    ///
    /// Two changes to `lowerFirstMatch` do it, and nothing else differs:
    ///
    ///   - **The empty subset is a subset.** `reach` declines to name a state
    ///     for it and this names one anyway, so a cell that leaves the operand
    ///     nowhere to be lands in `(∅, cell)` rather than dropping its edge.
    ///     Seed cells included: an attempt can die on the cycle it was seeded
    ///     in. The empty subset has no candidates, so `firstMatchCells` cuts
    ///     one cell over it — `always` — and every `(∅, cell)` steps to
    ///     `(∅, always)`, which then steps to itself. The trap a caller wants
    ///     is that state: unguarded, self-looping, entered within one cycle of
    ///     any death, and not an exit, because it stands for no match being
    ///     possible any more.
    ///
    ///     A *single* shared trap would be wrong, and instructively so. It
    ///     would have to be guarded by `always` to absorb every continuation,
    ///     and a state in the seed row guarded by `always` is entered on every
    ///     seeding — so a live attempt would carry the death of the attempt
    ///     beside it and the state vector would say both at once. Letting the
    ///     dead states inherit their cell's cube is what keeps them apart.
    ///
    ///   - **An accepting subset is explored like any other.**
    ///     `lowerFirstMatch` stops there because a match ending ends the
    ///     attempt; here the attempt runs on, and the later matches
    ///     `first_match` suppresses are exactly the ones an unbounded operator
    ///     still has to see.
    ///
    /// The result is total: `firstMatchCells` cuts the valuation space into
    /// cells that cover it, and every cell now names a state. It is also
    /// deterministic in the only sense a fragment can be — two cells may
    /// overlap, but a cell decides the truth of every candidate guard, so two
    /// cells sharing a valuation agree on all of them and carry the same
    /// subset. Co-enabled siblings therefore have the same exits and the same
    /// successors, which makes them bisimilar; `first_match` needed that only
    /// for its exits, and a complement reads the whole state.
    ///
    /// `dead` is what a caller reads the death out of, and it is the states
    /// carrying the **empty subset** rather than the wider "no path to an exit"
    /// that co-reachability computes. Two reasons, and the second is the
    /// convincing one.
    ///
    /// With the operand pruned first — which `determiniseObligation` does — the
    /// two sets coincide. A path from an operand state to an exit lifts to a path
    /// here, because the cells cover the valuation space, so a non-empty subset
    /// always has an exit ahead of it.
    ///
    /// And the wider set is *wrong* for a fragment with no exits at all. `always
    /// p` is one: a safety obligation has no finite witness, so co-reachability
    /// marks every state doomed and an attempt would fail the cycle it started.
    /// Its only failure is that no continuation is possible any more, which is
    /// the empty subset exactly. One rule therefore covers a matching obligation
    /// and a safety one, which two rules would not.
    ///
    /// `masks` names state masks over the operand — abort scopes — and each is
    /// freed and replaced by one over the result: a determinised state is in the
    /// translated mask when its subset meets the original. That is the right
    /// translation for the only thing a mask is asked, whether the guarded
    /// subtree could hold a thread of this attempt, because a subset is precisely
    /// the threads it holds.
    ///
    /// An operand with no states falls out of the general path as the bare
    /// trap, which is the total automaton for the empty language.
    ///
    /// The cost is the determinisation's, `|subsets| · |cells|` states, and
    /// dropping the truncation is what makes that bound bite: `first_match`
    /// takes most of it back by only ever exploring a subset while no match has
    /// ended, and this explores the whole reachable product. `Options.limit`
    /// caps it like every other blow-up here.
    fn determinise(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        masks: [][]Word,
    ) Error!Total {
        const gpa = builder.gpa;

        var into: Determinisation = try .init(gpa, operand.words);
        defer into.deinit(gpa);

        try builder.firstMatchCells(node, operand, operand.first, &into);
        for (into.cells.items) |cell| {
            _ = try builder.reachTotal(node, operand, &into, operand.first, cell);
        }
        const seed_count = into.states.items.len;

        var frontier: usize = 0;
        while (frontier < into.states.items.len) : (frontier += 1) {
            try into.start.append(gpa, @intCast(into.edges.items.len));
            const subset = into.states.items[frontier].subset;

            @memset(into.candidates, 0);
            var live: BitIterator = .init(into.subsets.maskOf(subset));
            while (live.next()) |state| Nfa.bitUnion(into.candidates, operand.rowConst(state));

            try builder.firstMatchCells(node, operand, into.candidates, &into);
            for (into.cells.items) |cell| {
                const target = try builder.reachTotal(node, operand, &into, into.candidates, cell);
                try into.edges.append(gpa, target);
            }
        }
        try into.start.append(gpa, @intCast(into.edges.items.len));

        const count = try builder.checkLimit(node, into.states.items.len);
        var fragment = try Fragment.init(gpa, count);
        errdefer fragment.deinit(gpa);
        const dead = try maskAlloc(gpa, fragment.words);
        errdefer gpa.free(dead);

        for (into.states.items, 0..) |state, offset| {
            const index: u32 = @intCast(offset);
            fragment.guard[index] = state.guard;
            for (into.edges.items[into.start.items[offset]..into.start.items[offset + 1]]) |target| {
                Nfa.bitSet(fragment.row(index), target);
            }
            if (offset < seed_count) Nfa.bitSet(fragment.first, index);
            const subset = into.subsets.maskOf(state.subset);
            if (Nfa.bitOverlaps(subset, operand.last)) Nfa.bitSet(fragment.last, index);
            if (!Nfa.bitAny(subset)) Nfa.bitSet(dead, index);
        }

        for (masks) |*mask| {
            const moved = try maskAlloc(gpa, fragment.words);
            errdefer gpa.free(moved);
            for (into.states.items, 0..) |state, offset| {
                if (Nfa.bitOverlaps(into.subsets.maskOf(state.subset), mask.*)) {
                    Nfa.bitSet(moved, @intCast(offset));
                }
            }
            gpa.free(mask.*);
            mask.* = moved;
        }
        return .{ .fragment = fragment, .dead = dead };
    }

    /// A total determinisation: the fragment, and the states an attempt in it
    /// has failed in.
    const Total = struct {
        fragment: Fragment,
        /// `fragment.words` long. See `determinise` for what is in it and why
        /// that set rather than a wider one.
        dead: []Word,

        fn deinit(total: *Total, gpa: Allocator) void {
            total.fragment.deinit(gpa);
            gpa.free(total.dead);
        }
    };

    /// `reach`, with the empty subset named rather than declined. See
    /// `determinise`, which is the whole reason the two differ.
    fn reachTotal(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        into: *Determinisation,
        candidates: []const Word,
        cell: Guard.Index,
    ) Error!u32 {
        if (try builder.reach(node, operand, into, candidates, cell)) |state| return state;
        @memset(into.subset, 0);
        return builder.subsetState(node, into, cell);
    }

    // -- Materialisation -----------------------------------------------------

    /// Walk `mask` to a fixpoint over `edges`, never entering a state outside
    /// `enterable`.
    fn closure(
        builder: *Builder,
        mask: []Word,
        edges: []const Word,
        words: u32,
        enterable: []const Word,
    ) Allocator.Error!void {
        var stack: std.ArrayListUnmanaged(u32) = .empty;
        defer stack.deinit(builder.gpa);

        var seen: BitIterator = .init(mask);
        while (seen.next()) |state| try stack.append(builder.gpa, state);

        while (stack.pop()) |state| {
            var outgoing: BitIterator = .init(edges[@as(usize, state) * words ..][0..words]);
            while (outgoing.next()) |target| {
                if (!Nfa.bitIsSet(enterable, target) or Nfa.bitIsSet(mask, target)) continue;
                Nfa.bitSet(mask, target);
                try stack.append(builder.gpa, target);
            }
        }
    }

    /// What pruning kept of one fragment.
    const Surviving = struct {
        /// Per source state, its new index or `dropped`.
        map: []u32,
        state_count: u32,

        fn deinit(surviving: *Surviving, gpa: Allocator) void {
            gpa.free(surviving.map);
        }
    };

    const dropped: u32 = std.math.maxInt(u32);

    /// Restrict a fragment to the states that can take part in a match: entered
    /// under a satisfiable guard, reachable from `first`, and co-reachable to
    /// `last`.
    ///
    /// `keep_doomed` drops the last of those three wholesale. A counted
    /// obligation needs it: the states that lead to no exit are what its
    /// failures are *read out of*, so removing them would remove the verdict.
    /// For an exit-free obligation — `always p` — it is every state, and
    /// pruning would leave nothing at all.
    ///
    /// `exempt` drops it for named states only, and is `Plan.pending`: the
    /// states a complement leaves without an exit on purpose. Empty for a
    /// property with no `prop_not` in it, which is the common case and costs
    /// the `bitAny` below and nothing else.
    fn prune(
        builder: *Builder,
        fragment: *const Fragment,
        keep_doomed: bool,
        exempt: []const Word,
    ) Allocator.Error!Surviving {
        const gpa = builder.gpa;
        const count = fragment.state_count;
        const words = fragment.words;

        const enterable = try maskAlloc(gpa, words);
        defer gpa.free(enterable);
        for (0..count) |index| {
            if (fragment.guard[index] != .never) Nfa.bitSet(enterable, @intCast(index));
        }

        const reachable = try maskAlloc(gpa, words);
        defer gpa.free(reachable);
        for (reachable, fragment.first, enterable) |*word, entry, ok| word.* = entry & ok;
        try builder.closure(reachable, fragment.follow, words, enterable);

        const incoming = try gpa.alloc(Word, @as(usize, count) * words);
        defer gpa.free(incoming);
        @memset(incoming, 0);
        for (0..count) |index| {
            const state: u32 = @intCast(index);
            var outgoing: BitIterator = .init(fragment.rowConst(state));
            while (outgoing.next()) |target| {
                Nfa.bitSet(incoming[@as(usize, target) * words ..][0..words], state);
            }
        }

        const live = try maskAlloc(gpa, words);
        defer gpa.free(live);
        if (keep_doomed) {
            @memcpy(live, reachable);
        } else {
            for (live, fragment.last, enterable) |*word, exit, ok| word.* = exit & ok;
            // Seeding the backward walk with the exempt states keeps them and
            // everything that can only reach an exit *through* them, which is
            // the whole point: a complement's doomed region is entered from
            // states that are themselves doomed, so exempting the leaves
            // without exempting the paths to them would prune the paths.
            if (Nfa.bitAny(exempt)) {
                for (live, exempt, enterable) |*word, spared, ok| word.* |= spared & ok;
            }
            try builder.closure(live, incoming, words, enterable);
            Nfa.bitIntersect(live, reachable);
        }

        const map = try gpa.alloc(u32, count);
        errdefer gpa.free(map);
        var state_count: u32 = 0;
        for (0..count) |index| {
            if (Nfa.bitIsSet(live, @intCast(index))) {
                map[index] = state_count;
                state_count += 1;
            } else map[index] = dropped;
        }
        return .{ .map = map, .state_count = state_count };
    }

    /// `fragment` restricted to what `prune` keeps, with every mask over it
    /// rewritten to match. Each mask is freed and replaced at the new width.
    ///
    /// `materialize` applies the same map on the way out to an `Nfa`; this
    /// applies it to a fragment still under construction, which two things want.
    /// `holdsEveryCycle` reads its operand's *shape* rather than combining it,
    /// and a state no cycle can enter would answer for a reason that has nothing
    /// to do with the shape. And `determiniseObligation` needs a state that leads
    /// to no exit gone before it determinises, or such a state ends up inside a
    /// non-empty subset that is nonetheless doomed — one that reports its failure
    /// a cycle late, or never at all when it loops on itself.
    ///
    /// `exempt` is `prune`'s, read in the pre-compaction numbering. A caller
    /// that needs it rewritten too passes the same slice in `masks`, which is
    /// safe in that order: pruning has finished reading it before the rewrite
    /// frees it.
    fn compact(
        builder: *Builder,
        fragment: *const Fragment,
        masks: [][]Word,
        keep_doomed: bool,
        exempt: []const Word,
    ) Allocator.Error!Fragment {
        const gpa = builder.gpa;
        var kept = try builder.prune(fragment, keep_doomed, exempt);
        defer kept.deinit(gpa);

        var into = try Fragment.init(gpa, kept.state_count);
        errdefer into.deinit(gpa);
        for (0..fragment.state_count) |index| {
            const state = kept.map[index];
            if (state == dropped) continue;
            into.guard[state] = fragment.guard[index];
            // A program is an index pair into a pool pruning never touches, so it
            // relocates with its state and needs no rewrite. A dropped state's
            // program goes with it: no cycle can enter it, so its assignments were
            // unreachable (LRM 16.10 attaches them to a match that cannot happen).
            into.program[state] = fragment.program[index];
            var outgoing: BitIterator = .init(fragment.rowConst(@intCast(index)));
            while (outgoing.next()) |target| {
                if (kept.map[target] != dropped) Nfa.bitSet(into.row(state), kept.map[target]);
            }
            if (Nfa.bitIsSet(fragment.first, @intCast(index))) Nfa.bitSet(into.first, state);
            if (Nfa.bitIsSet(fragment.last, @intCast(index))) Nfa.bitSet(into.last, state);
        }

        for (masks) |*mask| {
            const moved = try maskAlloc(gpa, into.words);
            errdefer gpa.free(moved);
            var bits: BitIterator = .init(mask.*);
            while (bits.next()) |state| {
                if (kept.map[state] != dropped) Nfa.bitSet(moved, kept.map[state]);
            }
            gpa.free(mask.*);
            mask.* = moved;
        }
        return into;
    }

    /// Swap `plan`'s obligation for the determinised one the counted discipline
    /// runs, and hand back the mask its failures are read out of.
    ///
    /// `masks` is every mask over the obligation's states: one per abort, and one
    /// trailing slot this writes `dead` into. Each is rewritten twice over, once
    /// by the pruning and once by the determinisation, and the plan's copies are
    /// pointed at the results as it goes.
    fn determiniseObligation(
        builder: *Builder,
        node: Ir.Node.Index,
        plan: *Plan,
        masks: [][]Word,
        dead: *[]Word,
    ) Error!void {
        const gpa = builder.gpa;
        const scopes = masks[0 .. masks.len - 1];

        for (plan.aborts.items) |abort| {
            // A `.branch` abort has no counted reading at all; the derivation is
            // on `Monitor.Abort.Scope`. `.attempt` comes through intact.
            if (abort.scope == .branch) {
                return builder.rejectProperty(abort.node, error.UnsupportedAbortScope);
            }
        }

        // A fragment with no exits is `always p`: a safety obligation, whose only
        // failure is that no continuation is possible at all. Co-reachability is
        // vacuous there and pruning on it would delete every state, so it keeps
        // them. For everything else pruning is exactly what makes "the subset is
        // empty" and "no continuation can match" the same set, which is what lets
        // one `dead` mask serve both kinds of obligation.
        const safety = !Nfa.bitAny(plan.obligation.last);
        // A complement below this point contributed states that lead to no exit
        // on purpose, and pruning must spare them for `Plan.pending`'s reason.
        // They come through determinisation as ordinary states of a subset, so
        // this is the last place the distinction is needed.
        const clean = try builder.compact(
            &plan.obligation,
            scopes,
            safety,
            plan.pending orelse &.{},
        );
        plan.obligation.deinit(gpa);
        plan.obligation = clean;
        for (plan.aborts.items, scopes) |*abort, mask| abort.mask = mask;

        const total = try builder.determinise(node, &plan.obligation, scopes);
        plan.obligation.deinit(gpa);
        plan.obligation = total.fragment;
        for (plan.aborts.items, scopes) |*abort, mask| abort.mask = mask;
        masks[masks.len - 1] = total.dead;
        dead.* = total.dead;
    }

    /// Turn one or more fragments into automata over a single alphabet.
    ///
    /// The guard table and the root set are compacted across *all* of them at
    /// once, which is the point: a monitor's two automata have to name the same
    /// term by the same index, or a backend evaluates it twice. `pinned` names
    /// root indices to keep even if no surviving guard mentions them, and is
    /// rewritten in place with their new indices.
    ///
    /// `masks` names state masks over `fragments[0]` — the abort scopes, and a
    /// counted obligation's `dead` — and is likewise rewritten in place. Pruning
    /// only ever removes states and keeps the survivors in order, so the
    /// rewritten mask is never wider than the one it replaces and the caller
    /// reads it back at the narrower width.
    ///
    /// `keep_doomed` and `exempt` apply to `fragments[0]` alone, which is the
    /// only one that is ever an obligation. See `prune`.
    fn materialize(
        builder: *Builder,
        fragments: []const *const Fragment,
        pinned: []u32,
        masks: []const []Word,
        keep_doomed: bool,
        exempt: []const Word,
        out: []Nfa,
    ) Error!Nfa.Alphabet {
        const gpa = builder.gpa;
        std.debug.assert(fragments.len == out.len);
        std.debug.assert(fragments.len <= 2);
        std.debug.assert(masks.len == 0 or fragments.len >= 1);

        var surviving: [2]Surviving = undefined;
        var pruned: usize = 0;
        defer for (surviving[0..pruned]) |*entry| entry.deinit(gpa);
        while (pruned < fragments.len) : (pruned += 1) {
            surviving[pruned] = try builder.prune(
                fragments[pruned],
                keep_doomed and pruned == 0,
                if (pruned == 0) exempt else &.{},
            );
        }

        if (masks.len != 0) {
            const scratch = try gpa.alloc(Word, fragments[0].words);
            defer gpa.free(scratch);
            for (masks) |mask| {
                @memset(scratch, 0);
                var bits: BitIterator = .init(mask);
                while (bits.next()) |state| {
                    const mapped = surviving[0].map[state];
                    if (mapped != dropped) Nfa.bitSet(scratch, mapped);
                }
                @memcpy(mask, scratch);
            }
        }

        // Guards, keeping the two reserved slots, over every fragment.
        const guard_map = try gpa.alloc(u32, builder.table.items.len);
        defer gpa.free(guard_map);
        @memset(guard_map, dropped);
        guard_map[0] = 0;
        guard_map[1] = 1;
        var guard_count: u32 = 2;
        for (fragments, surviving[0..fragments.len]) |fragment, kept| {
            for (0..fragment.state_count) |index| {
                if (kept.map[index] == dropped) continue;
                const guard = @intFromEnum(fragment.guard[index]);
                if (guard_map[guard] == dropped) {
                    guard_map[guard] = guard_count;
                    guard_count += 1;
                }
            }
        }

        // Roots: whatever is pinned, then whatever a surviving guard mentions.
        const root_map = try gpa.alloc(u32, builder.roots.items.len);
        defer gpa.free(root_map);
        @memset(root_map, dropped);
        var root_count: u32 = 0;
        for (pinned) |root| {
            if (root_map[root] == dropped) {
                root_map[root] = root_count;
                root_count += 1;
            }
        }
        for (builder.table.items, 0..) |entry, guard| {
            if (guard_map[guard] == dropped) continue;
            for (builder.literals.items[entry.start..][0..entry.len]) |literal| {
                if (root_map[literal.root] == dropped) {
                    root_map[literal.root] = root_count;
                    root_count += 1;
                }
            }
        }
        for (pinned) |*root| root.* = root_map[root.*];

        const roots = try gpa.alloc(Ir.Node.Index, root_count);
        errdefer gpa.free(roots);
        for (builder.roots.items, 0..) |root, index| {
            if (root_map[index] != dropped) roots[root_map[index]] = root;
        }

        const guards = try gpa.alloc(Guard, guard_count);
        errdefer gpa.free(guards);
        guards[0] = .{ .start = 0, .len = 0 };
        guards[1] = .{ .start = 0, .len = 0 };

        var literals: std.ArrayListUnmanaged(Literal) = .empty;
        errdefer literals.deinit(gpa);
        for (builder.table.items, 0..) |entry, guard| {
            const mapped = guard_map[guard];
            if (mapped == dropped or mapped < 2) continue;
            const start: u32 = @intCast(literals.items.len);
            for (builder.literals.items[entry.start..][0..entry.len]) |literal| {
                try literals.append(gpa, .{
                    .negated = literal.negated,
                    .root = @intCast(root_map[literal.root]),
                });
            }
            guards[mapped] = .{ .start = start, .len = entry.len };
        }

        var emitted: usize = 0;
        errdefer for (out[0..emitted]) |*nfa| nfa.deinit(gpa);
        while (emitted < fragments.len) : (emitted += 1) {
            out[emitted] = try builder.emit(
                fragments[emitted],
                &surviving[emitted],
                guard_map,
                guard_count,
            );
        }

        return .{
            .roots = roots,
            .guards = guards,
            .literals = try literals.toOwnedSlice(gpa),
        };
    }

    /// Union the slots a boolean node reads into `into`. A guard is a cube over
    /// roots, a root is a boolean tree, and only its atoms read anything — the mask
    /// is the front-end's, declared through `sva_atom_reads`.
    fn boolReads(builder: *Builder, node: Ir.Node.Index, into: []Word) void {
        const ir = builder.ir;
        const data = ir.dataOf(node);
        switch (ir.tagOf(node)) {
            .bool_const => {},
            .atom => {
                const reads = ir.atomReadsOf(@enumFromInt(data.lhs)) orelse return;
                for (into, reads) |*word, mask| word.* |= mask;
            },
            .bnot => builder.boolReads(@enumFromInt(data.lhs), into),
            .band, .bor => {
                builder.boolReads(@enumFromInt(data.lhs), into);
                builder.boolReads(@enumFromInt(data.rhs), into);
            },
            else => unreachable, // not a boolean-layer tag
        }
    }

    /// The slots a state's own cycle reads: every literal of its guard, plus every
    /// slot its program WRITES.
    ///
    /// The writes belong here because a compound assignment reads the slot it
    /// updates (LRM 16.6 admits exactly those operators on a local variable
    /// lvalue), and the read is inside the handle the front-end evaluates rather
    /// than in any atom, so nothing declares it. Treating a written slot as read at
    /// its own site is the sound reading, and it costs only a merge that liveness
    /// would otherwise have allowed.
    fn stateReads(builder: *Builder, fragment: *const Fragment, state: u32, into: []Word) void {
        const entry = builder.table.items[@intFromEnum(fragment.guard[state])];
        for (builder.literals.items[entry.start..][0..entry.len]) |literal| {
            builder.boolReads(builder.roots.items[literal.root], into);
        }
        const program = fragment.program[state];
        for (0..program.len) |offset| {
            const site = builder.sites.items[program.start + offset];
            Nfa.bitSet(into, builder.assign_slot.items[site]);
        }
    }

    /// F.5.4's flow set, projected onto the automaton: per state, the slots whose
    /// values may still be read from that state onward.
    ///
    /// A group sitting in a state has already had that state's guard hold, so what
    /// is still readable is what the states it can go on to read — a least fixpoint
    /// over `follow`, iterated because the graph has cycles wherever a repetition
    /// does. Restricting a valuation to this before comparing two groups is what
    /// lets groups that differ only in dead slots merge, which is what stops an
    /// accumulating assignment under an unbounded repetition growing a group per
    /// cycle for ever.
    fn liveness(
        builder: *Builder,
        fragment: *const Fragment,
        kept: *const Surviving,
        state_count: u32,
        words: u32,
        follow: []const Word,
        live: []Word,
        local_words: u32,
    ) Allocator.Error!void {
        const gpa = builder.gpa;
        const reads = try gpa.alloc(Word, @as(usize, state_count) * local_words);
        defer gpa.free(reads);
        @memset(reads, 0);
        for (0..fragment.state_count) |index| {
            const state = kept.map[index];
            if (state == dropped) continue;
            builder.stateReads(fragment, @intCast(index), reads[@as(usize, state) * local_words ..][0..local_words]);
        }

        var changed = true;
        while (changed) {
            changed = false;
            for (0..state_count) |index| {
                const state: u32 = @intCast(index);
                const into = live[@as(usize, state) * local_words ..][0..local_words];
                var next: BitIterator = .init(follow[@as(usize, state) * words ..][0..words]);
                while (next.next()) |target| {
                    const target_reads = reads[@as(usize, target) * local_words ..][0..local_words];
                    const target_live = live[@as(usize, target) * local_words ..][0..local_words];
                    for (into, target_reads, target_live) |*word, read, alive| {
                        const before = word.*;
                        word.* |= read | alive;
                        if (word.* != before) changed = true;
                    }
                }
            }
        }
    }

    /// Renumber one pruned fragment into an `Nfa` over the compacted guards.
    fn emit(
        builder: *Builder,
        fragment: *const Fragment,
        kept: *const Surviving,
        guard_map: []const u32,
        guard_count: u32,
    ) Allocator.Error!Nfa {
        const gpa = builder.gpa;
        const state_count = kept.state_count;
        const words = Nfa.wordsFor(state_count);

        const guard_of = try gpa.alloc(Guard.Index, state_count);
        errdefer gpa.free(guard_of);
        const follow = try gpa.alloc(Word, @as(usize, state_count) * words);
        errdefer gpa.free(follow);
        const first = try gpa.alloc(Word, words);
        errdefer gpa.free(first);
        const last = try gpa.alloc(Word, words);
        errdefer gpa.free(last);
        // Zero here for every automaton. Only a counted obligation has anything
        // to put in it, and `assertion` fills it once the mask it carried through
        // pruning has been renumbered along with the abort scopes.
        const dead = try gpa.alloc(Word, words);
        errdefer gpa.free(dead);
        const guard_states = try gpa.alloc(Word, @as(usize, guard_count) * words);
        @memset(follow, 0);
        @memset(first, 0);
        @memset(last, 0);
        @memset(dead, 0);
        @memset(guard_states, 0);

        // LRM 16.10 assignment programs, and the slot liveness that lets two groups
        // holding the same *readable* values merge. Both are empty for a local-free
        // assertion, which is the gate every local-keyed path in Nfa is behind.
        const local_words: u32 = if (builder.ir.localCount() == 0) 0 else builder.ir.localWords();
        const program = if (local_words == 0)
            try gpa.alloc(Nfa.Program, 0)
        else
            try gpa.alloc(Nfa.Program, state_count);
        errdefer gpa.free(program);
        if (program.len != 0) @memset(program, Nfa.Program.none);
        const live = try gpa.alloc(Word, @as(usize, state_count) * local_words);
        errdefer gpa.free(live);
        @memset(live, 0);

        for (0..fragment.state_count) |index| {
            const state = kept.map[index];
            if (state == dropped) continue;
            if (program.len != 0) program[state] = fragment.program[index];
            const guard = guard_map[@intFromEnum(fragment.guard[index])];
            guard_of[state] = @enumFromInt(guard);
            Nfa.bitSet(guard_states[@as(usize, guard) * words ..][0..words], state);

            const row = follow[@as(usize, state) * words ..][0..words];
            var outgoing: BitIterator = .init(fragment.rowConst(@intCast(index)));
            while (outgoing.next()) |target| {
                if (kept.map[target] != dropped) Nfa.bitSet(row, kept.map[target]);
            }
            if (Nfa.bitIsSet(fragment.first, @intCast(index))) Nfa.bitSet(first, state);
            if (Nfa.bitIsSet(fragment.last, @intCast(index))) Nfa.bitSet(last, state);
        }

        if (local_words != 0) {
            try builder.liveness(fragment, kept, state_count, words, follow, live, local_words);
        }

        return .{
            .state_count = state_count,
            .words = words,
            .guard_of = guard_of,
            .follow = follow,
            .first = first,
            .last = last,
            .dead = dead,
            .guard_states = guard_states,
            .program = program,
            // The pool is the builder's and outlives every fragment; each automaton
            // gets its own copy so a monitor owns its tables outright.
            .sites = try gpa.dupe(u32, builder.sites.items),
            .live = live,
            .local_words = local_words,
        };
    }
};
/// Flow analysis per F.5.4: sample, block, flow masks per IR node.
/// Gate on localCount > 0; zero words means analysis was skipped.
const FlowAnalysis = struct {
    sample: []Word,
    block: []Word,
    out: []Word,
    must: []Word,
    words: u32,
    local_count: u32,

    fn nodeIndex(analysis: *const FlowAnalysis, node: Ir.Node.Index) usize {
        return @as(usize, @intFromEnum(node)) * analysis.words;
    }

    fn sampleOf(analysis: *const FlowAnalysis, node: Ir.Node.Index) []Word {
        if (analysis.words == 0) return &.{};
        return analysis.sample[analysis.nodeIndex(node)..][0..analysis.words];
    }

    fn blockOf(analysis: *const FlowAnalysis, node: Ir.Node.Index) []Word {
        if (analysis.words == 0) return &.{};
        return analysis.block[analysis.nodeIndex(node)..][0..analysis.words];
    }

    fn outOf(analysis: *const FlowAnalysis, node: Ir.Node.Index) []Word {
        if (analysis.words == 0) return &.{};
        return analysis.out[analysis.nodeIndex(node)..][0..analysis.words];
    }

    fn mustOf(analysis: *const FlowAnalysis, node: Ir.Node.Index) []Word {
        if (analysis.words == 0) return &.{};
        return analysis.must[analysis.nodeIndex(node)..][0..analysis.words];
    }

    fn deinit(analysis: *FlowAnalysis, gpa: Allocator) void {
        gpa.free(analysis.sample);
        gpa.free(analysis.block);
        gpa.free(analysis.out);
        gpa.free(analysis.must);
    }
};
/// Whether a subtree contains an assignment, i.e. a `seq_match` node.
fn hasAssignmentsBelow(ir: *const Ir, node: Ir.Node.Index) bool {
    if (ir.tagOf(node) == .seq_match) return true;
    var buf: [Ir.max_operands]Ir.Node.Index = undefined;
    for (ir.operandsOf(node, &buf)) |operand| {
        if (hasAssignmentsBelow(ir, operand)) return true;
    }
    return false;
}

/// Union into `into` the slots any `seq_match` below `node` WRITES: F.5.4's
/// `sample`, without the declaration scoping, which this IR has no node for.
/// F.5.4's local variable flow, per IR node.
///
/// The standard defines three set-valued functions over an unclocked sequence --
/// `sample` (the slots it assigns), `block` (the slots it stops from flowing out)
/// and `flow` (what flows out given what flowed in) -- and one remark that makes
/// them cheap here: `flow(X, R) = (X | flow({}, R)) - block(R)`. So `flow({}, R)`
/// and `block(R)` are enough to answer any `flow`, and both are bottom-up. IR
/// operands point strictly backwards, so one forward pass over the nodes fills
/// every table.
///
/// A fourth set is not F.5.4's and exists for this implementation: `read_first`,
/// the slots a sequence may READ before it assigns them itself. A group carries
/// one valuation for both operands of a product, so a read that could observe the
/// sibling's write is the shape that cannot be represented -- and a read of a slot
/// the operand has already assigned itself cannot, whatever the sibling did. That
/// distinction is what keeps `Error.LocalCrossRead` off the legal shapes.
///
/// This IR has no declaration node: slots are declared on the Ir and are in scope
/// for the whole directive, so F.5.4's `( t v; R )` rules -- which subtract the
/// declared variable -- have nothing to apply to and are omitted.
const Flow = struct {
    words: u32,
    sample: []Word,
    block: []Word,
    out: []Word,
    read_first: []Word,

    fn rowOf(table: []Word, words: u32, node: Ir.Node.Index) []Word {
        return table[@as(usize, @intFromEnum(node)) * words ..][0..words];
    }

    fn sampleOf(flow: *const Flow, node: Ir.Node.Index) []const Word {
        return rowOf(flow.sample, flow.words, node);
    }
    fn blockOf(flow: *const Flow, node: Ir.Node.Index) []const Word {
        return rowOf(flow.block, flow.words, node);
    }
    fn outOf(flow: *const Flow, node: Ir.Node.Index) []const Word {
        return rowOf(flow.out, flow.words, node);
    }
    fn readFirstOf(flow: *const Flow, node: Ir.Node.Index) []const Word {
        return rowOf(flow.read_first, flow.words, node);
    }

    fn deinit(flow: *Flow, gpa: Allocator) void {
        gpa.free(flow.sample);
        gpa.free(flow.block);
        gpa.free(flow.out);
        gpa.free(flow.read_first);
    }

    fn compute(gpa: Allocator, ir: *const Ir) Allocator.Error!Flow {
        const words = ir.localWords();
        const count = ir.nodes.len;
        const total = count * @as(usize, words);

        var flow: Flow = .{
            .words = words,
            .sample = try gpa.alloc(Word, total),
            .block = try gpa.alloc(Word, total),
            .out = try gpa.alloc(Word, total),
            .read_first = try gpa.alloc(Word, total),
        };
        errdefer flow.deinit(gpa);
        @memset(flow.sample, 0);
        @memset(flow.block, 0);
        @memset(flow.out, 0);
        @memset(flow.read_first, 0);

        for (0..count) |index| {
            const node: Ir.Node.Index = @enumFromInt(index);
            const data = ir.dataOf(node);
            const sample = rowOf(flow.sample, words, node);
            const block = rowOf(flow.block, words, node);
            const out = rowOf(flow.out, words, node);
            const reads = rowOf(flow.read_first, words, node);

            switch (ir.tagOf(node)) {
                // A boolean assigns nothing and blocks nothing; what it READS is
                // the front-end's declared per-atom mask (F.5.4: sample(b) = {},
                // block(b) = {}, flow(X, b) = X).
                .atom => {
                    if (ir.atomReadsOf(@enumFromInt(data.lhs))) |mask| {
                        for (reads, mask) |*word, bit| word.* |= bit;
                    }
                },
                .bool_const => {},
                .bnot => copyInto(reads, rowOf(flow.read_first, words, @enumFromInt(data.lhs))),
                .band, .bor => {
                    copyInto(reads, rowOf(flow.read_first, words, @enumFromInt(data.lhs)));
                    unionInto(reads, rowOf(flow.read_first, words, @enumFromInt(data.rhs)));
                },
                .seq_boolean => copyInto(reads, rowOf(flow.read_first, words, @enumFromInt(data.lhs))),

                // `( 1, v = e )`: sample = {v}, block = {}, flow(X) = X | {v}. The
                // right-hand side is read BEFORE the write lands, so its own reads
                // are read-first even for the slot being written -- which is what a
                // compound assignment does.
                .seq_match => {
                    const operand: Ir.Node.Index = @enumFromInt(data.lhs);
                    copyInto(sample, flow.sampleOf(operand));
                    copyInto(block, flow.blockOf(operand));
                    copyInto(out, flow.outOf(operand));
                    copyInto(reads, flow.readFirstOf(operand));
                    const start = data.rhs;
                    const items = ir.extra.items[start];
                    for (0..items) |item| {
                        const slot = ir.extra.items[start + 1 + item * 2];
                        Nfa.bitSet(sample, slot);
                        Nfa.bitSet(out, slot);
                    }
                },

                // Concatenation, at either delay: sample and flow compose, and a
                // slot blocked by the left flows again if the right assigns it
                // (F.5.4: block(R1 ##1 R2) = (block(R1) - flow({}, R2)) | block(R2)).
                // A read on the right is read-first only if the left has not
                // already assigned it.
                .seq_delay => {
                    const delay = ir.delayOf(node);
                    const left = delay.left;
                    const right = delay.right;
                    copyInto(sample, flow.sampleOf(left));
                    unionInto(sample, flow.sampleOf(right));
                    copyInto(out, flow.outOf(left));
                    unionInto(out, flow.outOf(right));
                    copyInto(reads, flow.readFirstOf(left));
                    for (reads, flow.readFirstOf(right), flow.sampleOf(left)) |*word, later, done| {
                        word.* |= later & ~done;
                    }
                    for (block, flow.blockOf(left), flow.outOf(right)) |*word, blocked, again| {
                        word.* = blocked & ~again;
                    }
                    unionInto(block, flow.blockOf(right));
                    // flow(X, R) = (X | flow({}, R)) - block(R): the remark keeps
                    // `out` free of anything the composite blocks.
                    maskOut(out, block);
                },

                // Disjunction: a slot flows out only if it flows out of BOTH
                // operands, which is F.5.4's intersection and the reason
                // `((a, x = 1) or (b, x = 2))` flows x while assigning it in one
                // branch each does not.
                .seq_or => {
                    const left: Ir.Node.Index = @enumFromInt(data.lhs);
                    const right: Ir.Node.Index = @enumFromInt(data.rhs);
                    copyInto(sample, flow.sampleOf(left));
                    unionInto(sample, flow.sampleOf(right));
                    copyInto(block, flow.blockOf(left));
                    unionInto(block, flow.blockOf(right));
                    copyInto(reads, flow.readFirstOf(left));
                    unionInto(reads, flow.readFirstOf(right));
                    for (out, flow.outOf(left), flow.outOf(right)) |*word, a, bb| word.* = a & bb;
                    maskOut(out, block);
                },

                // A product blocks what both operands assign, on top of what either
                // blocks (F.5.4's intersect rule). `seq_and` is the same shape: the
                // lowering builds it as a product too.
                .seq_intersect, .seq_and => {
                    const left: Ir.Node.Index = @enumFromInt(data.lhs);
                    const right: Ir.Node.Index = @enumFromInt(data.rhs);
                    copyInto(sample, flow.sampleOf(left));
                    unionInto(sample, flow.sampleOf(right));
                    copyInto(block, flow.blockOf(left));
                    unionInto(block, flow.blockOf(right));
                    for (block, flow.sampleOf(left), flow.sampleOf(right)) |*word, a, bb| word.* |= a & bb;
                    copyInto(reads, flow.readFirstOf(left));
                    unionInto(reads, flow.readFirstOf(right));
                    copyInto(out, flow.outOf(left));
                    unionInto(out, flow.outOf(right));
                    maskOut(out, block);
                },

                // `exp throughout R`: the boolean is a condition on every cycle, so
                // it assigns nothing and its reads join the operand's.
                .seq_throughout => {
                    const operand: Ir.Node.Index = @enumFromInt(data.rhs);
                    copyInto(sample, flow.sampleOf(operand));
                    copyInto(block, flow.blockOf(operand));
                    copyInto(out, flow.outOf(operand));
                    copyInto(reads, flow.readFirstOf(operand));
                    unionInto(reads, flow.readFirstOf(@enumFromInt(data.lhs)));
                },

                // first_match keeps a subset of its operand's matches and changes no
                // valuation: every set is the operand's (F.5.4).
                .seq_first_match => {
                    const operand: Ir.Node.Index = @enumFromInt(data.lhs);
                    copyInto(sample, flow.sampleOf(operand));
                    copyInto(block, flow.blockOf(operand));
                    copyInto(out, flow.outOf(operand));
                    copyInto(reads, flow.readFirstOf(operand));
                },

                // Repetition: F.5.4 gives `R[*1:$]` the operand's sets, and a bounded
                // count is the concatenation of copies, which changes none of them --
                // every iteration assigns and reads the same slots. `lo == 0` cannot
                // occur: this IR has no empty match.
                .seq_repeat_consec, .seq_repeat_goto, .seq_repeat_nonconsec => {
                    const operand = ir.repeatOf(node).seq;
                    copyInto(sample, flow.sampleOf(operand));
                    copyInto(block, flow.blockOf(operand));
                    copyInto(out, flow.outOf(operand));
                    copyInto(reads, flow.readFirstOf(operand));
                },

                // The property layer carries no valuation of its own; what matters
                // for the hazards is what its operands do, so the sets propagate.
                else => {
                    var buf: [Ir.max_operands]Ir.Node.Index = undefined;
                    for (ir.operandsOf(node, &buf)) |operand| {
                        unionInto(sample, flow.sampleOf(operand));
                        unionInto(block, flow.blockOf(operand));
                        unionInto(out, flow.outOf(operand));
                        unionInto(reads, flow.readFirstOf(operand));
                    }
                },
            }
        }
        return flow;
    }
};

fn copyInto(into: []Word, from: []const Word) void {
    for (into, from) |*word, bit| word.* = bit;
}

fn unionInto(into: []Word, from: []const Word) void {
    for (into, from) |*word, bit| word.* |= bit;
}

fn maskOut(into: []Word, from: []const Word) void {
    for (into, from) |*word, bit| word.* &= ~bit;
}

fn anyCommon(left: []const Word, right: []const Word) bool {
    for (left, right) |a, b| {
        if (a & b != 0) return true;
    }
    return false;
}

fn checkLocalHazards(
    ir: *const Ir,
    flow: *const Flow,
    node: Ir.Node.Index,
    diagnostic: ?*Diagnostic,
) Error!void {
    const tag = ir.tagOf(node);
    const data = ir.dataOf(node);

    switch (tag) {
        // A determinised state stands for a SET of operand threads, and with an
        // assignment beneath it those threads hold different valuations -- exactly
        // what one valuation per thread cannot carry. A READ below either is fine:
        // with nothing assigned there, every operand thread carries the valuation
        // that flowed in.
        .seq_first_match, .prop_not => {
            if (hasAssignmentsBelow(ir, node)) {
                if (diagnostic) |into| into.* = .{ .node = node };
                return error.LocalUnderDeterminisation;
            }
        },
        // F.5.5 evaluates both operands of a product from the SAME input context, so
        // a slot one operand assigns is not in the other's context and no legal
        // sequence reads it there (LRM 16.10 rule a). A product thread here holds ONE
        // valuation, which would show the sibling's write.
        //
        // The test is on read_first, not on every read: a read of a slot the operand
        // has already assigned ITSELF cannot observe the sibling's value whatever the
        // sibling did, so `(d ##1 (true, x = data) ##0 (e == x))` beside an operand
        // that also assigns x is fine -- which is F.5.4's own s8 shape. Only a read
        // that could still see the incoming value is refused.
        .seq_intersect, .seq_and => {
            const left: Ir.Node.Index = @enumFromInt(data.lhs);
            const right: Ir.Node.Index = @enumFromInt(data.rhs);
            if (anyCommon(flow.sampleOf(left), flow.readFirstOf(right)) or
                anyCommon(flow.sampleOf(right), flow.readFirstOf(left)))
            {
                if (diagnostic) |into| into.* = .{ .node = node };
                return error.LocalCrossRead;
            }
        },
        // A universal operator with no upper bound keeps one attempt per cycle
        // pending for ever. Counted tracking exists to collapse those into a
        // counter per state, and two attempts in one state with different
        // valuations have different futures, so they cannot share one.
        .prop_always => {
            const range = ir.propRangeOf(node);
            if (range.hi == Ir.unbounded and hasAssignmentsBelow(ir, node)) {
                if (diagnostic) |into| into.* = .{ .node = node };
                return error.LocalUnboundedAttempts;
            }
        },
        .prop_until, .prop_until_with, .prop_s_until, .prop_s_until_with => {
            if (hasAssignmentsBelow(ir, node)) {
                if (diagnostic) |into| into.* = .{ .node = node };
                return error.LocalUnboundedAttempts;
            }
        },
        else => {},
    }

    var buf: [Ir.max_operands]Ir.Node.Index = undefined;
    for (ir.operandsOf(node, &buf)) |operand| {
        try checkLocalHazards(ir, flow, operand, diagnostic);
    }
}
// -- Tests -------------------------------------------------------------------
//
// The load-bearing test is differential: a direct recursive reading of the
// pattern semantics computes the set of cycles a match starting at `start` can
// end on, and every trace over a small alphabet is checked against what the
// automaton reports. Unit tests catch shapes; this catches a wrong `follow`
// edge in the fusion or repetition cases, which nothing else will.

const testing = std.testing;

/// Cycles a match of `node` starting at `start` can end on, as a bit per cycle,
/// read straight off the semantics.
fn bruteForceMatches(ir: *const Ir, node: Ir.Node.Index, trace: []const u64, start: u32) u64 {
    if (start >= trace.len) return 0;
    const length: u32 = @intCast(trace.len);
    const data = ir.dataOf(node);

    switch (ir.tagOf(node)) {
        .seq_boolean => {
            const atoms = [_]Word{trace[start]};
            const holds = Nfa.evaluateBoolean(ir, @enumFromInt(data.lhs), &atoms);
            return if (holds) @as(u64, 1) << @intCast(start) else 0;
        },
        .seq_or => return bruteForceMatches(ir, @enumFromInt(data.lhs), trace, start) |
            bruteForceMatches(ir, @enumFromInt(data.rhs), trace, start),
        .seq_intersect => return bruteForceMatches(ir, @enumFromInt(data.lhs), trace, start) &
            bruteForceMatches(ir, @enumFromInt(data.rhs), trace, start),
        .seq_and => {
            // Both sides match from `start`, and the match ends at the later
            // of the two ends: an end of one side counts if the other can
            // finish at or before it.
            const left = bruteForceMatches(ir, @enumFromInt(data.lhs), trace, start);
            const right = bruteForceMatches(ir, @enumFromInt(data.rhs), trace, start);
            if (left == 0 or right == 0) return 0;
            // A side can finish at or before a cycle exactly when its earliest
            // end is at or before it.
            const after_left = ~((@as(u64, 1) << @intCast(@ctz(left))) - 1);
            const after_right = ~((@as(u64, 1) << @intCast(@ctz(right))) - 1);
            return (left & after_right) | (right & after_left);
        },
        .seq_throughout => {
            // Every cycle the match spans has to satisfy the boolean, so the
            // match cannot reach past the first cycle from `start` that fails
            // it.
            const ends = bruteForceMatches(ir, @enumFromInt(data.rhs), trace, start);
            var spanned: u64 = 0;
            var cycle = start;
            while (cycle < length) : (cycle += 1) {
                const atoms = [_]Word{trace[cycle]};
                if (!Nfa.evaluateBoolean(ir, @enumFromInt(data.lhs), &atoms)) break;
                spanned |= @as(u64, 1) << @intCast(cycle);
            }
            return ends & spanned;
        },
        .seq_first_match => {
            // Every match from one start is a prefix of every longer one, so
            // they are ordered by length and the shortest is the earliest end.
            const ends = bruteForceMatches(ir, @enumFromInt(data.lhs), trace, start);
            if (ends == 0) return 0;
            return @as(u64, 1) << @intCast(@ctz(ends));
        },
        // The assignments are invisible to the LANGUAGE: they change values, not
        // which words match (LRM 16.10). This reference matcher is about the
        // language, so it looks straight through them.
        .seq_match => return bruteForceMatches(ir, @enumFromInt(data.lhs), trace, start),
        .seq_delay => {
            const delay = ir.delayOf(node);
            const hi = @min(delay.hi, length);
            var ends: u64 = 0;
            var left_ends = bruteForceMatches(ir, delay.left, trace, start);
            while (left_ends != 0) {
                const seam: u32 = @ctz(left_ends);
                left_ends &= left_ends - 1;
                var gap = delay.lo;
                while (gap <= hi and seam + gap < length) : (gap += 1) {
                    ends |= bruteForceMatches(ir, delay.right, trace, seam + gap);
                }
            }
            return ends;
        },
        .seq_repeat_consec => {
            const repeat = ir.repeatOf(node);
            const hi = @min(repeat.hi, length);
            var ends: u64 = 0;
            var current = bruteForceMatches(ir, repeat.seq, trace, start);
            var count: u32 = 1;
            while (count <= hi and current != 0) : (count += 1) {
                if (count >= repeat.lo) ends |= current;
                var next: u64 = 0;
                var reached = current;
                while (reached != 0) {
                    const at: u32 = @ctz(reached);
                    reached &= reached - 1;
                    if (at + 1 < length) next |= bruteForceMatches(ir, repeat.seq, trace, at + 1);
                }
                current = next;
            }
            return ends;
        },
        .seq_repeat_goto, .seq_repeat_nonconsec => {
            const repeat = ir.repeatOf(node);
            const operand: Ir.Node.Index = @enumFromInt(ir.dataOf(repeat.seq).lhs);

            var at: [64]u32 = undefined;
            var count: u32 = 0;
            var cycle = start;
            while (cycle < length) : (cycle += 1) {
                const atoms = [_]Word{trace[cycle]};
                if (Nfa.evaluateBoolean(ir, operand, &atoms)) {
                    at[count] = cycle;
                    count += 1;
                }
            }

            const goto = ir.tagOf(node) == .seq_repeat_goto;
            var ends: u64 = 0;
            var taken = repeat.lo;
            while (taken <= @min(repeat.hi, count)) : (taken += 1) {
                const occurrence = at[taken - 1];
                if (goto) {
                    ends |= @as(u64, 1) << @intCast(occurrence);
                } else {
                    // The match may sit anywhere in the run of non-occurrences
                    // after the last one taken.
                    const stop = if (taken < count) at[taken] else length;
                    var end = occurrence;
                    while (end < stop) : (end += 1) ends |= @as(u64, 1) << @intCast(end);
                }
            }
            return ends;
        },
        else => unreachable,
    }
}

/// `inner within outer` read off the LRM rather than off the expansion
/// `Ir.seqWithin` builds: a match of `inner` lies somewhere inside the match,
/// which begins and ends where a match of `outer` does.
fn bruteForceWithin(
    ir: *const Ir,
    inner: Ir.Node.Index,
    outer: Ir.Node.Index,
    trace: []const u64,
    start: u32,
) u64 {
    // Where a match of `inner` beginning no earlier than `start` can end.
    var inside: u64 = 0;
    for (start..trace.len) |at| inside |= bruteForceMatches(ir, inner, trace, @intCast(at));
    if (inside == 0) return 0;

    // One of those has to fit, so the match cannot end before the earliest of
    // them does.
    const containing = ~((@as(u64, 1) << @intCast(@ctz(inside))) - 1);
    return bruteForceMatches(ir, outer, trace, start) & containing;
}

const Runner = struct {
    automaton: *const Automaton,
    ir: *const Ir,
    current: []Word,
    next: []Word,
    satisfied: []Word,
    root_bits: []Word,

    fn init(gpa: Allocator, automaton: *const Automaton, ir: *const Ir) !Runner {
        return .{
            .automaton = automaton,
            .ir = ir,
            .current = try gpa.alloc(Word, automaton.nfa.words),
            .next = try gpa.alloc(Word, automaton.nfa.words),
            .satisfied = try gpa.alloc(Word, automaton.nfa.words),
            .root_bits = try gpa.alloc(Word, automaton.alphabet.words()),
        };
    }

    fn deinit(runner: *Runner, gpa: Allocator) void {
        gpa.free(runner.current);
        gpa.free(runner.next);
        gpa.free(runner.satisfied);
        gpa.free(runner.root_bits);
    }

    fn evaluate(runner: *Runner, cycle: u64) void {
        const atoms = [_]Word{cycle};
        runner.automaton.alphabet.evaluate(runner.ir, &atoms, runner.root_bits);
        runner.automaton.nfa.satisfy(
            &runner.automaton.alphabet,
            runner.root_bits,
            runner.satisfied,
        );
    }

    /// The same match set `bruteForceMatches` computes, read off the automaton.
    fn matches(runner: *Runner, trace: []const u64, start: u32) u64 {
        var ends: u64 = 0;
        @memset(runner.current, 0);
        var cycle = start;
        while (cycle < trace.len) : (cycle += 1) {
            runner.evaluate(trace[cycle]);
            if (cycle == start) {
                runner.automaton.nfa.seed(runner.satisfied, runner.current);
            } else {
                runner.automaton.nfa.advance(runner.current, runner.satisfied, runner.next);
                @memcpy(runner.current, runner.next);
            }
            if (runner.automaton.nfa.matched(runner.current)) ends |= @as(u64, 1) << @intCast(cycle);
            if (!Nfa.bitAny(runner.current)) break;
        }
        return ends;
    }

    /// The same, with an attempt seeded every cycle and all of them sharing
    /// one state vector — what an engine actually runs, and what an operator
    /// defined per attempt has to survive.
    fn seededMatches(runner: *Runner, trace: []const u64) u64 {
        var ends: u64 = 0;
        @memset(runner.current, 0);
        for (trace, 0..) |cycle, index| {
            runner.evaluate(cycle);
            runner.automaton.nfa.advance(runner.current, runner.satisfied, runner.next);
            @memcpy(runner.current, runner.next);
            runner.automaton.nfa.seed(runner.satisfied, runner.current);
            if (runner.automaton.nfa.matched(runner.current)) ends |= @as(u64, 1) << @intCast(index);
        }
        return ends;
    }
};

/// Every trace of `length` cycles over `atom_count` atoms, one per assignment
/// of the atoms in each cycle.
const Traces = struct {
    atom_count: u5,
    length: u32,
    combinations: u64,
    assignment: u64 = 0,
    /// The assignment the current trace was built from, for a failure message.
    bits: u64 = 0,
    buffer: [8]u64 = undefined,

    fn init(atom_count: u5, length: u32) Traces {
        return .{
            .atom_count = atom_count,
            .length = length,
            .combinations = @as(u64, 1) << @intCast(@as(u32, atom_count) * length),
        };
    }

    fn next(traces: *Traces) ?[]const u64 {
        if (traces.assignment == traces.combinations) return null;
        traces.bits = traces.assignment;
        traces.assignment += 1;

        const per_cycle = (@as(u64, 1) << traces.atom_count) - 1;
        for (0..traces.length) |cycle| {
            traces.buffer[cycle] = (traces.bits >> @intCast(cycle * traces.atom_count)) & per_cycle;
        }
        return traces.buffer[0..traces.length];
    }
};

/// Check the automaton against `bruteForceMatches` over every trace of `length`
/// cycles over `atom_count` atoms, from every start cycle.
fn differential(
    gpa: Allocator,
    ir: *const Ir,
    pattern: Ir.Node.Index,
    atom_count: u5,
    length: u32,
) !void {
    var automaton = try sequence(gpa, ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    var runner = try Runner.init(gpa, &automaton, ir);
    defer runner.deinit(gpa);

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |trace| {
        for (0..length) |index| {
            const start: u32 = @intCast(index);
            const want = bruteForceMatches(ir, pattern, trace, start);
            const got = runner.matches(trace, start);
            if (want != got) {
                std.debug.print(
                    "trace bits {b}, start {d}: want ends {b}, got {b}\n",
                    .{ traces.bits, start, want, got },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}

/// Check that two patterns match over exactly the same cycles.
///
/// `differential` ties the automaton to `bruteForceMatches`; this ties one
/// reading of `bruteForceMatches` to another, which is how an operator that
/// also has a spelling in terms of operators verified before it gets a second
/// opinion on what it means.
fn sameLanguage(
    ir: *const Ir,
    left: Ir.Node.Index,
    right: Ir.Node.Index,
    atom_count: u5,
    length: u32,
) !void {
    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |trace| {
        for (0..length) |index| {
            const start: u32 = @intCast(index);
            const want = bruteForceMatches(ir, left, trace, start);
            const got = bruteForceMatches(ir, right, trace, start);
            if (want != got) {
                std.debug.print(
                    "trace bits {b}, start {d}: left ends {b}, right ends {b}\n",
                    .{ traces.bits, start, want, got },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}

/// The cycles a match starting at `start` can end on, read off the automaton.
fn endsOf(
    gpa: Allocator,
    ir: *const Ir,
    pattern: Ir.Node.Index,
    trace: []const u64,
    start: u32,
) !u64 {
    var automaton = try sequence(gpa, ir, pattern, .{}, null);
    defer automaton.deinit(gpa);
    var runner = try Runner.init(gpa, &automaton, ir);
    defer runner.deinit(gpa);
    return runner.matches(trace, start);
}

/// How many states a pattern lowers to, after pruning.
fn stateCount(gpa: Allocator, ir: *const Ir, pattern: Ir.Node.Index) !u32 {
    var automaton = try sequence(gpa, ir, pattern, .{}, null);
    defer automaton.deinit(gpa);
    return automaton.nfa.state_count;
}

/// A one-cycle pattern over the atom named by `handle`.
fn atomCycle(ir: *Ir, gpa: Allocator, handle: u32) !Ir.Node.Index {
    return ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(handle)));
}

/// `##[lo:hi] right` measured from the start of an attempt, which is this
/// operator with a one-cycle `true` to its left and nothing else.
fn leadingDelay(ir: *Ir, gpa: Allocator, lo: u32, hi: u32, right: Ir.Node.Index) !Ir.Node.Index {
    const always = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    return ir.seqDelay(gpa, always, lo, hi, right);
}

test "differential: delay ranges" {
    const gpa = testing.allocator;

    const Case = struct { lo: u32, hi: u32, leading: bool };
    const cases = [_]Case{
        .{ .lo = 0, .hi = 0, .leading = false }, // pure overlap
        .{ .lo = 1, .hi = 1, .leading = false }, // plain concatenation
        .{ .lo = 0, .hi = 2, .leading = false }, // overlap and chain together
        .{ .lo = 2, .hi = 3, .leading = false },
        // A bounded range whose lower bound is past 2, which is the only shape
        // where the chain's hand-off bound is not simply its first filler.
        .{ .lo = 3, .hi = 4, .leading = false },
        .{ .lo = 1, .hi = Ir.unbounded, .leading = false },
        .{ .lo = 3, .hi = Ir.unbounded, .leading = false },
        .{ .lo = 0, .hi = 0, .leading = true },
        .{ .lo = 0, .hi = 2, .leading = true },
        .{ .lo = 1, .hi = 1, .leading = true },
        .{ .lo = 2, .hi = 3, .leading = true },
        .{ .lo = 0, .hi = Ir.unbounded, .leading = true },
        .{ .lo = 2, .hi = Ir.unbounded, .leading = true },
    };

    for (cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const right = try atomCycle(&ir, gpa, 1);
        const pattern = if (case.leading)
            try leadingDelay(&ir, gpa, case.lo, case.hi, right)
        else
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), case.lo, case.hi, right);
        try differential(gpa, &ir, pattern, 2, 5);
    }
}

test "differential: fusion on both sides of the seam" {
    const gpa = testing.allocator;

    {
        // `(a or (b ##1 c)) ##0 a` has two exit states to fuse against, so the
        // construction has to make one fused state per pair rather than one.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try atomCycle(&ir, gpa, 0);
        const bc = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 1, try atomCycle(&ir, gpa, 2));
        const left = try ir.seqOr(gpa, a, bc);
        const pattern = try ir.seqDelay(gpa, left, 0, 0, try atomCycle(&ir, gpa, 0));
        try differential(gpa, &ir, pattern, 3, 4);
    }
    {
        // `(a ##1 b) ##0 (b ##[1:2] c)`: the right operand runs on past the
        // shared cycle, so a fused state has to leave along the edges of the
        // entry state it stands in for.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
        const bc = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 2, try atomCycle(&ir, gpa, 2));
        const pattern = try ir.seqDelay(gpa, ab, 0, 0, bc);
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // A right operand with two entry states, so the pairing is not a
        // one-to-one seam in either direction.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const right = try ir.seqOr(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 2)),
            try atomCycle(&ir, gpa, 1),
        );
        const pattern = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 0, 1, right);
        try differential(gpa, &ir, pattern, 3, 4);
    }
}

test "differential: repetition" {
    const gpa = testing.allocator;

    const Case = struct { tag: Ir.Node.Tag, lo: u32, hi: u32 };
    const cases = [_]Case{
        .{ .tag = .seq_repeat_consec, .lo = 1, .hi = 1 },
        .{ .tag = .seq_repeat_consec, .lo = 2, .hi = 3 },
        .{ .tag = .seq_repeat_consec, .lo = 1, .hi = Ir.unbounded },
        .{ .tag = .seq_repeat_consec, .lo = 2, .hi = Ir.unbounded },
        .{ .tag = .seq_repeat_goto, .lo = 1, .hi = 2 },
        .{ .tag = .seq_repeat_goto, .lo = 2, .hi = 2 },
        .{ .tag = .seq_repeat_goto, .lo = 1, .hi = Ir.unbounded },
        .{ .tag = .seq_repeat_goto, .lo = 2, .hi = Ir.unbounded },
        .{ .tag = .seq_repeat_nonconsec, .lo = 1, .hi = 1 },
        .{ .tag = .seq_repeat_nonconsec, .lo = 1, .hi = 2 },
        .{ .tag = .seq_repeat_nonconsec, .lo = 2, .hi = 3 },
        .{ .tag = .seq_repeat_nonconsec, .lo = 1, .hi = Ir.unbounded },
        .{ .tag = .seq_repeat_nonconsec, .lo = 2, .hi = Ir.unbounded },
    };

    for (cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try atomCycle(&ir, gpa, 0);
        const pattern = try ir.seqRepeat(gpa, case.tag, operand, case.lo, case.hi);
        try differential(gpa, &ir, pattern, 1, 6);
    }

    // Consecutive repetition also takes a multi-cycle operand.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
    const pattern = try ir.seqRepeat(gpa, .seq_repeat_consec, ab, 1, 2);
    try differential(gpa, &ir, pattern, 2, 5);
}

test "differential: disjunction and intersection" {
    const gpa = testing.allocator;

    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
        const pattern = try ir.seqOr(gpa, ab, try atomCycle(&ir, gpa, 2));
        try differential(gpa, &ir, pattern, 3, 4);
    }
    {
        // Both sides span the same cycles, so the product has to agree on the
        // end as well as the start.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const left = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1));
        const right = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 2), 2, 4);
        const pattern = try ir.seqIntersect(gpa, left, right);
        try differential(gpa, &ir, pattern, 3, 5);
    }
}

test "differential: nested and negated guards" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try ir.atom(gpa, @enumFromInt(0));
    const b = try ir.atom(gpa, @enumFromInt(1));

    // `(a && !b) ##0 (a || b)` fuses two structured guards, so the cube has to
    // flatten the conjunction and push the negation down to spot overlaps.
    const left = try ir.seqBoolean(gpa, try ir.bBinary(gpa, .band, a, try ir.bNot(gpa, b)));
    const right = try ir.seqBoolean(gpa, try ir.bBinary(gpa, .bor, a, b));
    const fused = try ir.seqDelay(gpa, left, 0, 0, right);
    const pattern = try ir.seqDelay(gpa, fused, 1, 2, try ir.seqBoolean(gpa, b));

    try differential(gpa, &ir, pattern, 2, 5);
}

test "differential: constant guards" {
    const gpa = testing.allocator;

    // A constant is the only way a guard reaches `never` without a fusion, and
    // it drives the occurrence construction's gap states dead (a `true` operand
    // has no non-occurrences) or its occurrence states dead (a `false` one has
    // no occurrences at all). Both must prune to something that still agrees.
    for ([_]bool{ true, false }) |value| {
        const cases = [_]struct { tag: Ir.Node.Tag, lo: u32, hi: u32 }{
            .{ .tag = .seq_repeat_consec, .lo = 1, .hi = 2 },
            .{ .tag = .seq_repeat_goto, .lo = 1, .hi = 2 },
            .{ .tag = .seq_repeat_goto, .lo = 2, .hi = Ir.unbounded },
            .{ .tag = .seq_repeat_nonconsec, .lo = 1, .hi = 2 },
            .{ .tag = .seq_repeat_nonconsec, .lo = 1, .hi = Ir.unbounded },
        };
        for (cases) |case| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const constant = try ir.seqBoolean(gpa, try ir.boolConst(gpa, value));
            const pattern = try ir.seqRepeat(gpa, case.tag, constant, case.lo, case.hi);
            try differential(gpa, &ir, pattern, 1, 5);
        }

        // And as an ordinary operand, where `false` strands the whole tree.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const constant = try ir.seqBoolean(gpa, try ir.boolConst(gpa, value));
        const pattern = try ir.seqDelay(gpa, constant, 0, 2, try atomCycle(&ir, gpa, 0));
        try differential(gpa, &ir, pattern, 1, 5);
    }
}

test "differential: intersection over a contradictory pair" {
    const gpa = testing.allocator;

    // `((a ##1 b) or (a ##1 c)) intersect (a ##1 !b)`. The product reaches the
    // pair demanding both `b` and `!b`; that pair has to be dropped without
    // dropping the sibling that survives alongside it.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const b = try ir.atom(gpa, @enumFromInt(1));
    const left = try ir.seqOr(
        gpa,
        try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1)),
        try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 2)),
    );
    const right = try ir.seqDelay(
        gpa,
        try atomCycle(&ir, gpa, 0),
        1,
        1,
        try ir.seqBoolean(gpa, try ir.bNot(gpa, b)),
    );
    const pattern = try ir.seqIntersect(gpa, left, right);

    try differential(gpa, &ir, pattern, 3, 4);
}

test "differential: throughout" {
    const gpa = testing.allocator;

    {
        // The boolean has to reach every cycle of the match, including the
        // filler cycles of a delay range and the cycles a repetition unrolls
        // to, not just the ones the operand names a letter for.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const b = try atomCycle(&ir, gpa, 1);
        const operand = try ir.seqDelay(gpa, b, 1, 3, try atomCycle(&ir, gpa, 2));
        const pattern = try ir.seqThroughout(gpa, try ir.atom(gpa, @enumFromInt(0)), operand);
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // An open range loops on its last filler, so the conjunction has to
        // land on a state the match can sit in for any number of cycles.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const b = try atomCycle(&ir, gpa, 1);
        const operand = try ir.seqDelay(gpa, b, 1, Ir.unbounded, try atomCycle(&ir, gpa, 2));
        const pattern = try ir.seqThroughout(gpa, try ir.atom(gpa, @enumFromInt(0)), operand);
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 1), 1, 3);
        const pattern = try ir.seqThroughout(gpa, try ir.atom(gpa, @enumFromInt(0)), operand);
        try differential(gpa, &ir, pattern, 2, 5);
    }
    {
        // `!a throughout (a ##1 b)` contradicts its operand in the first cycle
        // and has no match at all; the fragment prunes to nothing rather than
        // to something that still reports one.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
        const pattern = try ir.seqThroughout(gpa, try ir.bNot(gpa, a), operand);
        try differential(gpa, &ir, pattern, 2, 4);

        var automaton = try sequence(gpa, &ir, pattern, .{}, null);
        defer automaton.deinit(gpa);
        try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);
    }
    {
        // A conjunction the operand already carries is the same cube, so the
        // guard is unchanged rather than duplicated.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 0, 2, try atomCycle(&ir, gpa, 1));
        const pattern = try ir.seqThroughout(gpa, a, operand);
        try differential(gpa, &ir, pattern, 2, 5);
    }
}

test "throughout is an intersection the empty-match ban makes free" {
    const gpa = testing.allocator;

    // The LRM spells it `(exp[*0:$]) intersect seq`. No sequence here matches
    // an empty span, so `[*1:$]` covers every match `[*0:$]` would, and the
    // direct construction has to agree with that product exactly.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try ir.atom(gpa, @enumFromInt(0));
    const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 3, try atomCycle(&ir, gpa, 2));

    const direct = try ir.seqThroughout(gpa, a, operand);
    const spelled = try ir.seqIntersect(
        gpa,
        try ir.seqRepeat(gpa, .seq_repeat_consec, try ir.seqBoolean(gpa, a), 1, Ir.unbounded),
        operand,
    );
    try sameLanguage(&ir, direct, spelled, 3, 5);
    try differential(gpa, &ir, spelled, 3, 5);
}

test "differential: and" {
    const gpa = testing.allocator;

    {
        // Different lengths in both directions: the shorter side finishes and
        // the longer one carries the match to its own end.
        for ([_]bool{ false, true }) |flip| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
            const c = try atomCycle(&ir, gpa, 2);
            const pattern = if (flip)
                try ir.seqAnd(gpa, c, ab)
            else
                try ir.seqAnd(gpa, ab, c);
            try differential(gpa, &ir, pattern, 3, 5);
        }
    }
    {
        // Two ranges that overlap: for one trace either side can be the later
        // one, so both wait copies are live in the same automaton.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const left = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
        const right = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 2));
        const pattern = try ir.seqAnd(gpa, left, right);
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // Repetition puts a state in `last` that also has somewhere to go, so
        // ending there and running on are both readings of the same cycle.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const left = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 0), 1, 3);
        const right = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 1), 2, 3);
        const pattern = try ir.seqAnd(gpa, left, right);
        try differential(gpa, &ir, pattern, 2, 5);
    }
    {
        // An unbounded side: the wait copy it leaves behind loops, which is
        // sound for a sequence even though it costs the shape its `depth`.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const left = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, Ir.unbounded, try atomCycle(&ir, gpa, 1));
        const pattern = try ir.seqAnd(gpa, left, try atomCycle(&ir, gpa, 2));
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // Contradictory sides never both start, so there is no match to end.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const pattern = try ir.seqAnd(
            gpa,
            try ir.seqBoolean(gpa, a),
            try ir.seqDelay(gpa, try ir.seqBoolean(gpa, try ir.bNot(gpa, a)), 1, 1, try atomCycle(&ir, gpa, 1)),
        );
        try differential(gpa, &ir, pattern, 2, 4);

        var automaton = try sequence(gpa, &ir, pattern, .{}, null);
        defer automaton.deinit(gpa);
        try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);
    }
}

test "and ends at the later end where intersect insists on one end" {
    const gpa = testing.allocator;

    // `(a ##1 b) and c` over a trace holding `a` and `c` in cycle 0 and `b` in
    // cycle 1. The right side ended in cycle 0 and the left in cycle 1, so the
    // match ends in cycle 1 — where `intersect`, which is the same product
    // without the waiting, has no match at all.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
    const c = try atomCycle(&ir, gpa, 2);
    const conjunction = try ir.seqAnd(gpa, ab, c);
    const product = try ir.seqIntersect(gpa, ab, c);

    const trace = [_]u64{ 0b101, 0b010 };
    try testing.expectEqual(@as(u64, 0b10), try endsOf(gpa, &ir, conjunction, &trace, 0));
    try testing.expectEqual(@as(u64, 0), try endsOf(gpa, &ir, product, &trace, 0));
}

test "and is an intersection against a side padded to the later end" {
    const gpa = testing.allocator;

    // `A and B` is `(A intersect (B ##[0:$] true)) or ((A ##[0:$] true)
    // intersect B)`: whichever side ends later fixes the end, and the other is
    // padded out to meet it. That spelling costs two products where the direct
    // construction costs one, but it is written in operators that were
    // verified before this one, so it is worth an opinion on what `and` means.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const pad = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    const left = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
    const right = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 2), 1, 2);

    const direct = try ir.seqAnd(gpa, left, right);
    const spelled = try ir.seqOr(
        gpa,
        try ir.seqIntersect(gpa, left, try ir.seqDelay(gpa, right, 0, Ir.unbounded, pad)),
        try ir.seqIntersect(gpa, try ir.seqDelay(gpa, left, 0, Ir.unbounded, pad), right),
    );
    try sameLanguage(&ir, direct, spelled, 3, 5);
    try differential(gpa, &ir, spelled, 3, 5);
}

test "differential: within" {
    const gpa = testing.allocator;

    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
        const outer = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 2), 2, 4);
        try differential(gpa, &ir, try ir.seqWithin(gpa, inner, outer), 3, 5);
    }
    {
        // A one-cycle inner sequence is the case the padding has to get right
        // at both ends at once: it can sit in the first cycle of the outer
        // match, the last, or anywhere between.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try atomCycle(&ir, gpa, 0);
        const outer = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 3, try atomCycle(&ir, gpa, 2));
        try differential(gpa, &ir, try ir.seqWithin(gpa, inner, outer), 3, 5);
    }
}

test "within is the LRM reading, padding and all" {
    const gpa = testing.allocator;

    // `Ir.seqWithin` is an expansion, so the expansion is what needs checking:
    // `bruteForceWithin` reads the operator off the LRM instead, without
    // going near a delay.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const inner = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
    const outer = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 2), 1, 4);
    const pattern = try ir.seqWithin(gpa, inner, outer);

    var traces: Traces = .init(3, 5);
    while (traces.next()) |trace| {
        for (0..trace.len) |index| {
            const start: u32 = @intCast(index);
            const want = bruteForceWithin(&ir, inner, outer, trace, start);
            const got = bruteForceMatches(&ir, pattern, trace, start);
            if (want != got) {
                std.debug.print(
                    "trace bits {b}, start {d}: want ends {b}, got {b}\n",
                    .{ traces.bits, start, want, got },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "differential: first match" {
    const gpa = testing.allocator;

    {
        // A delay range: every cycle it spans is another match of the operand
        // from the same start, so there is something to suppress.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1));
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 2, 5);
    }
    {
        // Repetition, where every exit of the operand also has somewhere to
        // go: ending there and running on are both readings of the cycle, and
        // only the first survives.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 0), 1, 4);
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 1, 5);
    }
    {
        // A disjunction of two lengths: which side is the shortest match
        // depends on the trace, so neither can be chosen while building.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const short = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 2, 2, try atomCycle(&ir, gpa, 1));
        const long = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 4, try atomCycle(&ir, gpa, 2));
        const operand = try ir.seqOr(gpa, short, long);
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 3, 5);
    }
    {
        // An unbounded operand — the shape `first_match` exists for. The
        // filler loop is a subset reached from itself.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try leadingDelay(&ir, gpa, 1, Ir.unbounded, try atomCycle(&ir, gpa, 0));
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 1, 6);
    }
    {
        // Non-consecutive repetition, which is unbounded *and* keeps matching
        // every cycle once it has started: without the cut-off every one of
        // them would be reported.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try ir.seqRepeat(gpa, .seq_repeat_nonconsec, try atomCycle(&ir, gpa, 0), 2, 2);
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 1, 6);
    }
    {
        // Multi-literal guards, which is where a cell stops being one branch
        // per guard: `¬(a ∧ b)` is a disjunction, so failing that guard costs
        // a branch per literal and two of them overlap.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const b = try ir.atom(gpa, @enumFromInt(1));
        const both = try ir.seqBoolean(gpa, try ir.bBinary(gpa, .band, a, b));
        const operand = try ir.seqOr(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 2), 1, 1, both),
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 2), 1, 3, try atomCycle(&ir, gpa, 0)),
        );
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 3, 5);
    }
    {
        // Guards that overlap without either entailing the other, so a cell
        // has to pin both down: `a` and `a ∨ b` share valuations in either
        // direction and neither is the other's complement.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const b = try ir.atom(gpa, @enumFromInt(1));
        const either = try ir.seqBoolean(gpa, try ir.bBinary(gpa, .bor, a, b));
        const operand = try ir.seqOr(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 2), 1, 2, try ir.seqBoolean(gpa, a)),
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 2), 1, 3, either),
        );
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 3, 5);
    }
    {
        // Two multi-literal guards sharing a literal, over one candidate set.
        // Several cells then leave the operand in exactly the same place
        // without any one of their cubes covering the rest — `{¬a, ¬b}` and
        // `{¬b, ¬c}` both leave only the filler live — so a state has to be
        // named by its cell as well as by its subset. Naming it by the subset
        // alone would keep one cube and lose every valuation in the others.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const b = try ir.atom(gpa, @enumFromInt(1));
        const c = try ir.atom(gpa, @enumFromInt(2));
        const operand = try ir.seqDelay(gpa, try ir.seqBoolean(gpa, a), 1, 2, try ir.seqOr(
            gpa,
            try ir.seqBoolean(gpa, try ir.bBinary(gpa, .band, a, b)),
            try ir.seqBoolean(gpa, try ir.bBinary(gpa, .band, b, c)),
        ));
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 3, 5);
    }
    {
        // An unsatisfiable guard among the candidates, and one that is in
        // `last` at that. No cycle can enter it, so it belongs to no subset —
        // it would otherwise report a match in every cycle — and it tells no
        // two valuations apart, so it must not split the space either, which
        // it would do by ruling every cell out.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const never = try ir.seqBoolean(gpa, try ir.boolConst(gpa, false));
        const operand = try ir.seqOr(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, never),
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1)),
        );
        try differential(gpa, &ir, try ir.seqFirstMatch(gpa, operand), 2, 5);
    }
    {
        // A contradictory operand can never match, so there is nothing to
        // determinise and nothing survives pruning.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const a = try ir.atom(gpa, @enumFromInt(0));
        const operand = try ir.seqDelay(
            gpa,
            try ir.seqBoolean(gpa, a),
            0,
            0,
            try ir.seqBoolean(gpa, try ir.bNot(gpa, a)),
        );
        const pattern = try ir.seqFirstMatch(gpa, operand);
        try differential(gpa, &ir, pattern, 1, 4);

        var automaton = try sequence(gpa, &ir, pattern, .{}, null);
        defer automaton.deinit(gpa);
        try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);
    }
}

test "differential: a first match inside a larger sequence" {
    const gpa = testing.allocator;

    const inner_of = struct {
        fn build(ir: *Ir, allocator: Allocator) !Ir.Node.Index {
            return ir.seqDelay(allocator, try atomCycle(ir, allocator, 0), 1, 3, try atomCycle(ir, allocator, 1));
        }
    }.build;

    {
        // The shortest match fixes where the rest of the sequence starts, so
        // a suppressed longer match is a continuation that never happens.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const pattern = try ir.seqDelay(gpa, try ir.seqFirstMatch(gpa, inner), 1, 1, try atomCycle(&ir, gpa, 2));
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // A delay of 0 fuses a neighbouring guard into a cell's cube. A cell
        // is a cube precisely so that this stays a cube.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const pattern = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 2), 0, 0, try ir.seqFirstMatch(gpa, inner));
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // Against an intersection, which has to agree with the end cycle the
        // suppression picked rather than with any the operand offered.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const other = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 2), 2, 4);
        try differential(gpa, &ir, try ir.seqIntersect(gpa, try ir.seqFirstMatch(gpa, inner), other), 3, 5);
    }
    {
        // Against an `and`, whose half-finished copies carry a determinised
        // state alongside `finished`.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const pattern = try ir.seqAnd(gpa, try ir.seqFirstMatch(gpa, inner), try atomCycle(&ir, gpa, 2));
        try differential(gpa, &ir, pattern, 3, 5);
    }
    {
        // A `throughout` conjoins into every guard, cells included: it can
        // shorten a cell but never split one, so coverage survives.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const en = try ir.atom(gpa, @enumFromInt(2));
        try differential(gpa, &ir, try ir.seqThroughout(gpa, en, try ir.seqFirstMatch(gpa, inner)), 3, 5);
    }
    {
        // Determinising a determinised operand. Idempotent, and the second
        // pass runs over a fragment whose exits are all dead ends.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const once = try ir.seqFirstMatch(gpa, inner);
        const twice = try ir.seqFirstMatch(gpa, once);
        try differential(gpa, &ir, twice, 2, 5);
        try sameLanguage(&ir, once, twice, 2, 5);
    }
    {
        // Suppression is on the operand's own matches, so `first_match` of a
        // repetition is not the repetition of a `first_match`.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try inner_of(&ir, gpa);
        const pattern = try ir.seqRepeat(gpa, .seq_repeat_consec, try ir.seqFirstMatch(gpa, inner), 1, 2);
        try differential(gpa, &ir, pattern, 2, 6);
    }
}

test "first match keeps the earliest end and nothing else" {
    const gpa = testing.allocator;

    // Bit 0 is `a`, bit 1 is `b`. `a ##[1:3] b` matches wherever `b` holds in
    // the three cycles after an `a`, so a trace holding `b` throughout gives
    // one attempt three matches.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1));
    const pattern = try ir.seqFirstMatch(gpa, operand);

    const every = [_]u64{ 0b01, 0b10, 0b10, 0b10 };
    try testing.expectEqual(@as(u64, 0b1110), try endsOf(gpa, &ir, operand, &every, 0));
    try testing.expectEqual(@as(u64, 0b0010), try endsOf(gpa, &ir, pattern, &every, 0));

    // The kept end is the earliest one there is, not a fixed one: drop the
    // first `b` and the survivor moves with it.
    const later = [_]u64{ 0b01, 0b00, 0b10, 0b10 };
    try testing.expectEqual(@as(u64, 0b1100), try endsOf(gpa, &ir, operand, &later, 0));
    try testing.expectEqual(@as(u64, 0b0100), try endsOf(gpa, &ir, pattern, &later, 0));

    // And an operand that never matches keeps nothing.
    const never = [_]u64{ 0b01, 0b00, 0b00, 0b00 };
    try testing.expectEqual(@as(u64, 0), try endsOf(gpa, &ir, pattern, &never, 0));
}

test "first match is the identity on an operand that cannot match twice" {
    const gpa = testing.allocator;

    // Every match of `a ##1 b` from one start is the same length, so no match
    // has a shorter one to lose to and the determinisation must not change the
    // language.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
    try sameLanguage(&ir, operand, try ir.seqFirstMatch(gpa, operand), 2, 5);
}

test "first match suppresses inside an attempt, not across them" {
    const gpa = testing.allocator;

    // Every attempt shares one state vector, so an attempt whose first match
    // is still ahead must not be cut off by an earlier attempt's. It is not,
    // because the suppression is in the language rather than in the stepping:
    // two attempts meeting in one determinised state agree on everything the
    // operand could still do, so sharing a bit is exactly right.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1));
    const pattern = try ir.seqFirstMatch(gpa, operand);

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);
    var runner = try Runner.init(gpa, &automaton, &ir);
    defer runner.deinit(gpa);

    var traces: Traces = .init(2, 6);
    while (traces.next()) |trace| {
        var want: u64 = 0;
        for (0..trace.len) |start| {
            want |= bruteForceMatches(&ir, pattern, trace, @intCast(start));
        }
        const got = runner.seededMatches(trace);
        if (want != got) {
            std.debug.print(
                "trace bits {b}: want ends {b}, got {b}\n",
                .{ traces.bits, want, got },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test "determinising costs subsets times cells" {
    const gpa = testing.allocator;

    // The blow-up is the headline cost of the operator, so the shapes it is
    // worth having are pinned rather than described.
    const Case = struct { states: u32, kept: u32 };
    for ([_]Case{
        // `##[1:$] a`: the operand's own filler loop determinises to itself,
        // so the shortest match costs nothing over the operand.
        .{ .states = 3, .kept = 3 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const operand = try leadingDelay(&ir, gpa, 1, Ir.unbounded, try atomCycle(&ir, gpa, 0));
        try testing.expectEqual(case.states, try stateCount(gpa, &ir, operand));
        try testing.expectEqual(case.kept, try stateCount(gpa, &ir, try ir.seqFirstMatch(gpa, operand)));
    }

    // `a ##[1:3] b`: one subset per cycle the range spans, each split by
    // whether `b` held, and the run stops at the first that did.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const ranged = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1));
    try testing.expectEqual(@as(u32, 4), try stateCount(gpa, &ir, ranged));
    try testing.expectEqual(@as(u32, 6), try stateCount(gpa, &ir, try ir.seqFirstMatch(gpa, ranged)));
}

test "the state cap reports the first match that tripped it" {
    const gpa = testing.allocator;

    // Eight independent guards on the candidates cut the valuation space into
    // 256 cells, which overruns the cap long before a state is allocated.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    var wide = try atomCycle(&ir, gpa, 0);
    for (1..8) |handle| wide = try ir.seqOr(gpa, wide, try atomCycle(&ir, gpa, @intCast(handle)));
    const operand = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 8), 1, 1, wide);
    const pattern = try ir.seqFirstMatch(gpa, operand);

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.StateLimitExceeded,
        sequence(gpa, &ir, pattern, .{ .limit = 64 }, &diagnostic),
    );
    try testing.expectEqual(pattern, diagnostic.node);
    try testing.expectEqual(@as(u64, 128), diagnostic.size);
}

// -- The total determinisation -----------------------------------------------
//
// `determinise` is what the counted tracking discipline lowers an unbounded
// obligation through — `always`, `s_eventually` and the `until` family. Its
// contract is checked in full here as well as through those operators, because
// the two things it adds over `first_match` are exactly the two a complement
// reads, and neither survives to a materialised automaton: pruning drops a
// state that leads to no exit, and the trap is one. So these tests run against
// the fragment, with the builder that owns its cube table still alive.

/// Whether `guard` holds under an atom valuation, read off the builder's own
/// cube table. `Nfa.Alphabet.holds` is the same test one materialisation later.
fn cubeHolds(builder: *const Builder, guard: Guard.Index, atoms: []const Word) bool {
    if (guard == .never) return false;
    for (builder.literalsOf(@intFromEnum(guard))) |literal| {
        const root = builder.roots.items[literal.root];
        if (Nfa.evaluateBoolean(builder.ir, root, atoms) == literal.negated) return false;
    }
    return true;
}

/// Lower `pattern` and determinise it totally, keeping only the fragment. The
/// `dead` mask has tests of its own; these are about the shape.
fn totalOf(builder: *Builder, pattern: Ir.Node.Index) Error!Fragment {
    var operand = try builder.lower(pattern);
    defer operand.deinit(builder.gpa);
    const total = try builder.determinise(pattern, &operand, &.{});
    builder.gpa.free(total.dead);
    return total.fragment;
}

/// Run `fragment` over `trace` from a seeding at cycle 0, leaving the states
/// it is in in `current`. `entered` is scratch a cycle wide.
///
/// `Runner` does this against an `Nfa`, which is a materialisation away and so
/// a pruning away: the trap is exactly what pruning removes.
fn runFragment(
    builder: *const Builder,
    fragment: *const Fragment,
    trace: []const u64,
    current: []Word,
    entered: []Word,
) void {
    @memset(current, 0);
    for (trace, 0..) |cycle, index| {
        if (index == 0) {
            @memcpy(entered, fragment.first);
        } else {
            @memset(entered, 0);
            var live: BitIterator = .init(current);
            while (live.next()) |state| Nfa.bitUnion(entered, fragment.rowConst(state));
        }

        const atoms = [_]Word{cycle};
        @memset(current, 0);
        var candidate: BitIterator = .init(entered);
        while (candidate.next()) |state| {
            if (cubeHolds(builder, fragment.guard[state], &atoms)) Nfa.bitSet(current, state);
        }
    }
}

/// One operand the total determinisation is checked over, with what it costs.
///
/// `operand`, `shortest` and `total` are the state counts of the operand, of
/// its `first_match` and of its total determinisation, all before pruning:
/// what each construction costs rather than what survives it. They are pinned
/// because the blow-up dropping the truncation buys is the headline cost of
/// the operators that will call this, and `overlaps` because a cell
/// enumeration that stopped producing overlapping cells would quietly make the
/// bisimulation below vacuous.
const TotalCase = struct {
    name: []const u8,
    atoms: u5,
    length: u32,
    overlaps: bool,
    operand: u32,
    shortest: u32,
    total: u32,
    build: *const fn (*Ir, Allocator) anyerror!Ir.Node.Index,
};

const total_cases = [_]TotalCase{
    .{
        // One state, and a valuation that misses it: the trap is reachable
        // even here, and the state that matched leads nowhere but into it.
        .name = "a",
        .atoms = 1,
        .length = 4,
        .overlaps = false,
        .operand = 1,
        .shortest = 1,
        .total = 3,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return atomCycle(ir, gpa, 0);
            }
        }.build,
    },
    .{
        // No attempt has two matches, so `first_match` is the identity on it
        // and every state the total construction adds is a death.
        .name = "a ##1 b",
        .atoms = 2,
        .length = 5,
        .overlaps = false,
        .operand = 2,
        .shortest = 2,
        .total = 5,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 1, try atomCycle(ir, gpa, 1));
            }
        }.build,
    },
    .{
        // A range, so an attempt can match twice and the subsets a
        // `first_match` never reaches are the ones this has to explore.
        .name = "a ##[1:2] b",
        .atoms = 2,
        .length = 5,
        .overlaps = false,
        .operand = 3,
        .shortest = 4,
        .total = 7,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 2, try atomCycle(ir, gpa, 1));
            }
        }.build,
    },
    .{
        // An open range: the operand's filler loop never dies on its own, so
        // the subsets close into a cycle rather than a chain.
        .name = "a ##[1:$] b",
        .atoms = 2,
        .length = 5,
        .overlaps = false,
        .operand = 3,
        .shortest = 3,
        .total = 5,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, Ir.unbounded, try atomCycle(ir, gpa, 1));
            }
        }.build,
    },
    .{
        // An unbounded repetition, whose every state is an exit: with the
        // truncation gone there is nothing to stop the exploration but the
        // subsets running out.
        .name = "a[*1:$]",
        .atoms = 1,
        .length = 5,
        .overlaps = false,
        .operand = 1,
        .shortest = 1,
        .total = 3,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(ir, gpa, 0), 1, Ir.unbounded);
            }
        }.build,
    },
    .{
        // Two guards, one of them a two-literal cube. Splitting the space by
        // `a ∧ b` leaves the cells `¬a` and `¬b`, and splitting those by `c`
        // leaves `¬a ∧ c` and `¬b ∧ c` — two distinct cells both holding the
        // valuation a=0, b=0, c=1. This is the shape the overlap argument on
        // `firstMatchCells` is about.
        .name = "(a ##0 b) or c",
        .atoms = 3,
        .length = 4,
        .overlaps = true,
        .operand = 4,
        .shortest = 6,
        .total = 9,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                const fused = try ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 0, 0, try atomCycle(ir, gpa, 1));
                return ir.seqOr(gpa, fused, try atomCycle(ir, gpa, 2));
            }
        }.build,
    },
    .{
        // The shape `determinising costs subsets times cells` pins for
        // `first_match`, so the two constructions can be read side by side.
        .name = "a ##[1:3] b",
        .atoms = 2,
        .length = 6,
        .overlaps = false,
        .operand = 4,
        .shortest = 6,
        .total = 9,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 3, try atomCycle(ir, gpa, 1));
            }
        }.build,
    },
    .{
        // An operand that is itself a determinisation, which is the worst case
        // the property layer can hand this: every one of its states is an exit
        // or a dead end, so the truncation `first_match` applied is exactly
        // what has to be undone.
        .name = "first_match(a ##[1:3] b)",
        .atoms = 2,
        .length = 6,
        .overlaps = false,
        .operand = 6,
        .shortest = 6,
        .total = 9,
        .build = struct {
            fn build(ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index {
                const ranged = try ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 3, try atomCycle(ir, gpa, 1));
                return ir.seqFirstMatch(gpa, ranged);
            }
        }.build,
    },
};

test "differential: a total determinisation keeps its operand's language" {
    const gpa = testing.allocator;

    for (total_cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const pattern = try case.build(&ir, gpa);

        var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
        defer builder.deinit();
        try builder.reserveCubes();

        var operand = try builder.lower(pattern);
        defer operand.deinit(gpa);
        var total = try builder.determinise(pattern, &operand, &.{});
        defer total.deinit(gpa);

        // Both over one alphabet, so a guard index means the same thing in
        // each and the two runners can be pointed at one valuation.
        var automata: [2]Nfa = undefined;
        var alphabet = try builder.materialize(
            &.{ &operand, &total.fragment },
            &.{},
            &.{},
            false,
            &.{},
            &automata,
        );
        defer alphabet.deinit(gpa);
        defer for (&automata) |*nfa| nfa.deinit(gpa);

        const before: Automaton = .{ .alphabet = alphabet, .nfa = automata[0] };
        const after: Automaton = .{ .alphabet = alphabet, .nfa = automata[1] };
        var want = try Runner.init(gpa, &before, &ir);
        defer want.deinit(gpa);
        var got = try Runner.init(gpa, &after, &ir);
        defer got.deinit(gpa);

        var traces: Traces = .init(case.atoms, case.length);
        while (traces.next()) |trace| {
            for (0..case.length) |index| {
                const start: u32 = @intCast(index);
                const expected = want.matches(trace, start);
                const actual = got.matches(trace, start);
                if (expected != actual) {
                    std.debug.print(
                        "{s}: trace bits {b}, start {d}: operand ends {b}, total ends {b}\n",
                        .{ case.name, traces.bits, start, expected, actual },
                    );
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

test "a total determinisation leaves every valuation somewhere to go" {
    const gpa = testing.allocator;

    for (total_cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const pattern = try case.build(&ir, gpa);

        var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
        defer builder.deinit();
        try builder.reserveCubes();
        var total = try totalOf(&builder, pattern);
        defer total.deinit(gpa);

        // Every guard is a cube over the atoms, so walking the atom space is
        // exhaustive rather than a sample: nothing finer can tell two
        // valuations apart.
        const space = @as(u64, 1) << case.atoms;
        for (0..@as(usize, total.state_count) + 1) |index| {
            // The slot past the last state stands for the seed row, which has
            // to cover the space as well — an attempt seeded into nothing is
            // as invisible as one that died.
            const row = if (index == total.state_count)
                total.first
            else
                total.rowConst(@intCast(index));

            var valuation: u64 = 0;
            while (valuation < space) : (valuation += 1) {
                const atoms = [_]Word{valuation};
                var reachable = false;
                var target: BitIterator = .init(row);
                while (target.next()) |state| {
                    if (cubeHolds(&builder, total.guard[state], &atoms)) reachable = true;
                }
                if (!reachable) {
                    std.debug.print(
                        "{s}: row {d} has no successor under valuation {b}\n",
                        .{ case.name, index, valuation },
                    );
                    return error.TestUnexpectedResult;
                }
            }
        }
    }
}

test "an attempt that dies falls into the trap and stays there" {
    const gpa = testing.allocator;

    // `a ##1 b`, atom 0 being `a` and atom 1 being `b`. An attempt seeded on
    // an `a` dies the next cycle if `b` does not hold, and after that nothing
    // the trace does can revive it or make it match.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const pattern = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));

    var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
    defer builder.deinit();
    try builder.reserveCubes();
    var total = try totalOf(&builder, pattern);
    defer total.deinit(gpa);

    const current = try maskAlloc(gpa, total.words);
    defer gpa.free(current);
    const entered = try maskAlloc(gpa, total.words);
    defer gpa.free(entered);

    // `a`, then no `b`, and then every valuation there is — the trap absorbs
    // all of them rather than merely the one that entered it.
    const trace = [_]u64{ 0b01, 0b00, 0b11, 0b10, 0b01, 0b00 };
    var settled: ?u32 = null;
    for (1..trace.len + 1) |cycles| {
        runFragment(&builder, &total, trace[0..cycles], current, entered);

        // One state, always: a run of a determinisation is a walk, and a
        // valuation lying in two cells at once would be the one exception.
        var live: BitIterator = .init(current);
        const state = live.next() orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(?u32, null), live.next());
        try testing.expect(!Nfa.bitIsSet(total.last, state));

        // The first cycle seeds the attempt on its `a` and it is still short
        // of its `b`; the second kills it. From the third on it is in the trap
        // proper, which is one state whichever way the trace goes.
        switch (cycles) {
            1 => try testing.expect(Nfa.bitAny(total.rowConst(state))),
            2 => {},
            else => {
                if (settled) |trap| try testing.expectEqual(trap, state);
                settled = state;
            },
        }
    }

    // And the trap is what a caller will look for: unguarded, so it survives
    // any continuation, and looping on nothing but itself.
    const trap = settled.?;
    try testing.expectEqual(Guard.Index.always, total.guard[trap]);
    var out: BitIterator = .init(total.rowConst(trap));
    try testing.expectEqual(@as(?u32, trap), out.next());
    try testing.expectEqual(@as(?u32, null), out.next());
}

test "cells that share a valuation lead to the same place" {
    const gpa = testing.allocator;

    // Two cells may overlap, and `firstMatchCells` argues that costs a
    // duplicate state and nothing else. `first_match` only leaned on that for
    // its exits; a complement reads the whole state vector, so the argument is
    // checked rather than asserted.
    //
    // It holds because a cell decides the truth of every candidate guard — it
    // either entails the guard or negates a literal of it — so two cells
    // holding one valuation agree on every guard and carry the same subset.
    // States with the same subset are entered under different cubes but have
    // the same exits and the same successors, which is a bisimulation: nothing
    // downstream can tell them apart either.
    for (total_cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const pattern = try case.build(&ir, gpa);

        var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
        defer builder.deinit();
        try builder.reserveCubes();
        var total = try totalOf(&builder, pattern);
        defer total.deinit(gpa);

        const space = @as(u64, 1) << case.atoms;
        var seen_overlap = false;
        for (0..@as(usize, total.state_count) + 1) |index| {
            const row = if (index == total.state_count)
                total.first
            else
                total.rowConst(@intCast(index));

            var valuation: u64 = 0;
            while (valuation < space) : (valuation += 1) {
                const atoms = [_]Word{valuation};
                var chosen: ?u32 = null;
                var target: BitIterator = .init(row);
                while (target.next()) |state| {
                    if (!cubeHolds(&builder, total.guard[state], &atoms)) continue;
                    const first = chosen orelse {
                        chosen = state;
                        continue;
                    };
                    seen_overlap = true;
                    try testing.expectEqual(
                        Nfa.bitIsSet(total.last, first),
                        Nfa.bitIsSet(total.last, state),
                    );
                    try testing.expectEqualSlices(
                        Word,
                        total.rowConst(first),
                        total.rowConst(state),
                    );
                }
            }
        }
        try testing.expectEqual(case.overlaps, seen_overlap);
    }
}

test "dropping the truncation costs the whole product" {
    const gpa = testing.allocator;

    // The blow-up is the reason phases 3 and 4 share one construction rather
    // than each growing their own, so the shapes worth having are pinned
    // rather than described. All three counts are pre-pruning, which is what
    // the construction allocates.
    for (total_cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const pattern = try case.build(&ir, gpa);

        var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
        defer builder.deinit();
        try builder.reserveCubes();

        var operand = try builder.lower(pattern);
        defer operand.deinit(gpa);
        var shortest = try builder.lowerFirstMatch(pattern, &operand);
        defer shortest.deinit(gpa);
        var total = try builder.determinise(pattern, &operand, &.{});
        defer total.deinit(gpa);

        testing.expectEqual(case.operand, operand.state_count) catch |err| {
            std.debug.print("{s}: operand states\n", .{case.name});
            return err;
        };
        testing.expectEqual(case.shortest, shortest.state_count) catch |err| {
            std.debug.print("{s}: first_match states\n", .{case.name});
            return err;
        };
        testing.expectEqual(case.total, total.fragment.state_count) catch |err| {
            std.debug.print("{s}: total states\n", .{case.name});
            return err;
        };
        // Every one of these operands can die, and a death is a state now, so
        // the mask a counted monitor reads its failures out of is never empty.
        testing.expect(Nfa.bitAny(total.dead)) catch |err| {
            std.debug.print("{s}: no dead state\n", .{case.name});
            return err;
        };
    }
}

test "a contradictory fusion prunes to nothing" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try ir.atom(gpa, @enumFromInt(0));
    const pattern = try ir.seqDelay(
        gpa,
        try ir.seqBoolean(gpa, a),
        0,
        0,
        try ir.seqBoolean(gpa, try ir.bNot(gpa, a)),
    );

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    // `a ##0 !a` can never match, so every state is either dead by guard or
    // stranded, and pruning leaves an automaton with no states at all.
    try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, 0), automaton.alphabet.roots.len);
}

test "an atom is one root however many nodes name it" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // Nothing in the IR hash-conses, so this is what a front-end building
    // bottom-up actually emits: a fresh `atom` node per mention. Both name
    // atom 0, so `a ##0 !a` is still a contradiction and there is still only
    // one term for a backend to evaluate.
    const pattern = try ir.seqDelay(
        gpa,
        try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(0))),
        0,
        0,
        try ir.seqBoolean(gpa, try ir.bNot(gpa, try ir.atom(gpa, @enumFromInt(0)))),
    );

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);

    // And the same across a plain concatenation, where nothing prunes: two
    // mentions of one atom stay one root.
    var twice: Ir = .init();
    defer twice.deinit(gpa);
    const concatenation = try twice.seqDelay(
        gpa,
        try twice.seqBoolean(gpa, try twice.atom(gpa, @enumFromInt(0))),
        1,
        1,
        try twice.seqBoolean(gpa, try twice.atom(gpa, @enumFromInt(0))),
    );
    var flat_automaton = try sequence(gpa, &twice, concatenation, .{}, null);
    defer flat_automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 2), flat_automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, 1), flat_automaton.alphabet.roots.len);
}

test "a cube sees through structure" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try ir.atom(gpa, @enumFromInt(0));
    const b = try ir.atom(gpa, @enumFromInt(1));

    // `(a && b) ##0 !(a || b)`. Neither guard is a bare atom, so the fusion
    // only sees the contradiction because a cube splits conjunctions and pushes
    // negations down to the leaves. Without that it is two opaque terms and the
    // dead state survives to waste a bit in every backend that consumes it.
    const pattern = try ir.seqDelay(
        gpa,
        try ir.seqBoolean(gpa, try ir.bBinary(gpa, .band, a, b)),
        0,
        0,
        try ir.seqBoolean(gpa, try ir.bNot(gpa, try ir.bBinary(gpa, .bor, a, b))),
    );

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 0), automaton.nfa.state_count);

    // Conjoining two cubes that share a literal has to collapse it, or the
    // guard table stops being canonical: `a && b && c` would intern twice
    // depending on how it was assembled, and every backend would evaluate `a`
    // once per occurrence.
    var shared: Ir = .init();
    defer shared.deinit(gpa);
    const x = try shared.atom(gpa, @enumFromInt(0));
    const y = try shared.atom(gpa, @enumFromInt(1));
    const z = try shared.atom(gpa, @enumFromInt(2));
    const fused = try shared.seqDelay(
        gpa,
        try shared.seqBoolean(gpa, try shared.bBinary(gpa, .band, x, y)),
        0,
        0,
        try shared.seqBoolean(gpa, try shared.bBinary(gpa, .band, x, z)),
    );

    var one_automaton = try sequence(gpa, &shared, fused, .{}, null);
    defer one_automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 1), one_automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, 3), one_automaton.alphabet.literalsOf(one_automaton.nfa.guard_of[0]).len);
}

test "distinct cubes sharing a hash bucket stay distinct" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // Interning is content-addressed, so the guard table has to survive many
    // cubes over one alphabet: `a[i] ##0 a[j]` for every ordered pair gives
    // every two-literal cube exactly once, and each must intern to its own
    // slot rather than colliding with a sibling.
    const atom_count = 6;
    var atoms: [atom_count]Ir.Node.Index = undefined;
    for (&atoms, 0..) |*slot, handle| slot.* = try ir.atom(gpa, @enumFromInt(handle));

    var alternatives: ?Ir.Node.Index = null;
    for (0..atom_count) |i| {
        for (i + 1..atom_count) |j| {
            const fused = try ir.seqDelay(
                gpa,
                try ir.seqBoolean(gpa, atoms[i]),
                0,
                0,
                try ir.seqBoolean(gpa, atoms[j]),
            );
            alternatives = if (alternatives) |have| try ir.seqOr(gpa, have, fused) else fused;
        }
    }

    var automaton = try sequence(gpa, &ir, alternatives.?, .{}, null);
    defer automaton.deinit(gpa);

    // One state per pair, each with its own two-literal cube, plus the two
    // reserved slots.
    const pairs = atom_count * (atom_count - 1) / 2;
    try testing.expectEqual(@as(u32, pairs), automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, pairs + 2), automaton.alphabet.guards.len);
    for (0..automaton.nfa.state_count) |state| {
        try testing.expectEqual(@as(usize, 2), automaton.alphabet.literalsOf(automaton.nfa.guard_of[state]).len);
    }
}

test "interning resolves a hash collision by content" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try ir.atom(gpa, @enumFromInt(0));
    const b = try ir.atom(gpa, @enumFromInt(1));

    var builder: Builder = .{ .gpa = gpa, .ir = &ir, .options = .{}, .diagnostic = null };
    defer builder.deinit();
    try builder.reserveCubes();

    // `{a}` interns into its own bucket and takes root 0, so `{b}` will take
    // root 1 and its content hash is known before it exists.
    const cube_a = try builder.guardOf(a, false);
    const hash_b = Builder.hashLiterals(&.{.{ .negated = false, .root = 1 }});

    // Wyhash will not collide two three-byte keys on demand, so the collision
    // is staged: point `{b}`'s bucket at `{a}`'s slot, which is exactly the
    // state the map would be in had the two hashed equal.
    try builder.cube_bucket.put(gpa, hash_b, @intFromEnum(cube_a));

    // Walking the chain has to reject `{a}` on content and append `{b}` behind
    // it rather than hand back the collided slot.
    const cube_b = try builder.guardOf(b, false);
    try testing.expect(cube_a != cube_b);
    try testing.expectEqual(@as(usize, 4), builder.table.items.len);
    try testing.expectEqualSlices(
        Literal,
        &.{.{ .negated = false, .root = 1 }},
        builder.literalsOf(@intFromEnum(cube_b)),
    );

    // `{b}` heads the bucket now and `{a}` hangs off it. A staged collision
    // cannot make a *lookup* of `{a}` arrive through that bucket — its own hash
    // still finds it directly — so the link itself is what to check.
    try testing.expectEqual(@intFromEnum(cube_a), builder.cube_chain.items[@intFromEnum(cube_b)]);
    try testing.expectEqual(@as(u32, @intFromEnum(cube_b)), builder.cube_bucket.get(hash_b).?);

    // A third cube into the same bucket must traverse both links before
    // concluding it is new, and land in front of them.
    const c = try ir.atom(gpa, @enumFromInt(2));
    const hash_c = Builder.hashLiterals(&.{.{ .negated = false, .root = 2 }});
    try builder.cube_bucket.put(gpa, hash_c, @intFromEnum(cube_b));
    const cube_c = try builder.guardOf(c, false);
    try testing.expect(cube_c != cube_a and cube_c != cube_b);
    try testing.expectEqual(@intFromEnum(cube_b), builder.cube_chain.items[@intFromEnum(cube_c)]);

    // And every one of them stays findable by content.
    try testing.expectEqual(cube_a, try builder.guardOf(a, false));
    try testing.expectEqual(cube_b, try builder.guardOf(b, false));
    try testing.expectEqual(@as(usize, 5), builder.table.items.len);
}

test "a constant true guard is the reserved always cube" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const pattern = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    // An empty conjunction *is* `always`, so it has to intern to that slot
    // rather than to a fresh cube that merely behaves the same. A backend is
    // entitled to skip evaluating `always` and to read `guard_states[always]`
    // as every unguarded state, and both break if `true` lands elsewhere.
    try testing.expectEqual(@as(u32, 1), automaton.nfa.state_count);
    try testing.expectEqual(Guard.Index.always, automaton.nfa.guard_of[0]);
    try testing.expectEqual(@as(usize, 2), automaton.alphabet.guards.len);
    try testing.expectEqual(@as(usize, 0), automaton.alphabet.roots.len);
}

test "states no cycle can enter are pruned" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const pattern = try ir.seqRepeat(
        gpa,
        .seq_repeat_goto,
        try ir.seqBoolean(gpa, try ir.boolConst(gpa, true)),
        2,
        2,
    );

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    // The gap states of `true[->2]` are guarded by `!true`. They sit on a path
    // from `first` to `last`, so reachability alone keeps them; it is the guard
    // that rules them out, leaving just the two occurrences.
    try testing.expectEqual(@as(u32, 2), automaton.nfa.state_count);
}

test "dead states are pruned" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try atomCycle(&ir, gpa, 0);
    const b = try atomCycle(&ir, gpa, 1);
    const pattern = try ir.seqDelay(gpa, a, 0, 0, b);

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    // `a ##0 b` is one cycle in which both hold: the two operand states are
    // left stranded by the fusion and only the fused state survives.
    try testing.expectEqual(@as(u32, 1), automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, 2), automaton.alphabet.literalsOf(automaton.nfa.guard_of[0]).len);
}

test "one state per letter occurrence" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const a = try atomCycle(&ir, gpa, 0);
    const b = try atomCycle(&ir, gpa, 1);

    // Two letters, two states: a position automaton has no ε-transitions to
    // carry and no start state to hold, because `first` is a mask.
    var plain_automaton = try sequence(gpa, &ir, try ir.seqDelay(gpa, a, 1, 1, b), .{}, null);
    defer plain_automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 2), plain_automaton.nfa.state_count);
    try testing.expectEqual(@as(usize, 2), plain_automaton.alphabet.roots.len);

    // A delay range pays for the cycles it may skip: `hi - 1` fillers.
    var ranged_automaton = try sequence(gpa, &ir, try ir.seqDelay(gpa, a, 1, 4, b), .{}, null);
    defer ranged_automaton.deinit(gpa);
    try testing.expectEqual(@as(u32, 5), ranged_automaton.nfa.state_count);
}

test "an attempt seeded every cycle" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const pattern = try ir.seqDelay(
        gpa,
        try atomCycle(&ir, gpa, 0),
        1,
        1,
        try atomCycle(&ir, gpa, 1),
    );

    var automaton = try sequence(gpa, &ir, pattern, .{}, null);
    defer automaton.deinit(gpa);

    var runner = try Runner.init(gpa, &automaton, &ir);
    defer runner.deinit(gpa);

    // Bit 0 is `a`, bit 1 is `b`. `a ##1 b` completes wherever `b` holds and
    // `a` held the cycle before: cycles 1, 2 and 4.
    const trace = [_]u64{ 0b01, 0b11, 0b10, 0b01, 0b10 };

    var ends: u64 = 0;
    @memset(runner.current, 0);
    for (trace, 0..) |cycle, index| {
        runner.evaluate(cycle);
        automaton.nfa.advance(runner.current, runner.satisfied, runner.next);
        @memcpy(runner.current, runner.next);
        automaton.nfa.seed(runner.satisfied, runner.current);
        if (automaton.nfa.matched(runner.current)) ends |= @as(u64, 1) << @intCast(index);
    }

    try testing.expectEqual(@as(u64, 0b10110), ends);
}

test "the state cap reports the node that tripped it" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const operand = try atomCycle(&ir, gpa, 0);
    const pattern = try ir.seqRepeat(gpa, .seq_repeat_consec, operand, 1, 5000);

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.StateLimitExceeded,
        sequence(gpa, &ir, pattern, .{ .limit = 64 }, &diagnostic),
    );
    try testing.expectEqual(pattern, diagnostic.node);
    try testing.expectEqual(@as(u64, 5000), diagnostic.size);
}

test "goto repetition over a compound pattern is rejected" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const ab = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1));
    // Built by hand: `seqRepeat` asserts against this in a safe build, so the
    // lowering's own check is what a release build relies on.
    const extra = try ir.addExtra(gpa, &.{ 1, 2 });
    const pattern = try ir.addNode(gpa, .{
        .tag = .seq_repeat_goto,
        .data = .{ .lhs = @intFromEnum(ab), .rhs = @intFromEnum(extra) },
    });

    try testing.expectError(
        error.UnsupportedRepeatOperand,
        sequence(gpa, &ir, pattern, .{}, null),
    );
}

// -- Property tests ----------------------------------------------------------
//
// Same shape as the sequence tests: a reference derived from the semantics
// rather than from the construction. An obligation attempt's fate is read off
// the definitions — it is discharged in the first cycle its sequence completes
// a match, and it fails in the first cycle from which *no* continuation of the
// trace could complete one — and that is compared against what the monitor
// reports cycle by cycle.

/// What became of one obligation attempt.
const Fate = struct {
    /// Cycle a completed match discharged it.
    matched_at: ?u32 = null,
    /// Cycle it ran out of possible futures.
    failed_at: ?u32 = null,
};

/// Whether a match of `pattern` starting at `start` can still end in `earliest`
/// or later, given the trace observed through `cycle` plus some continuation.
/// This is the definition the automaton's "state set is non-empty" is supposed to
/// implement, arrived at by enumerating continuations instead.
///
/// `earliest` is separate from `cycle` because two different questions want this
/// machinery: an obligation is alive while a match can still end in `cycle` or
/// later, and a followed-by is still owed a witness while its antecedent can
/// match ending *after* `cycle`.
fn canStillMatch(
    ir: *const Ir,
    pattern: Ir.Node.Index,
    trace: []const u64,
    start: u32,
    cycle: u32,
    earliest: u32,
    atom_count: u5,
    bound: u32,
) bool {
    var extended: [16]u64 = undefined;
    const prefix = cycle + 1;
    @memcpy(extended[0..prefix], trace[0..prefix]);

    const per_cycle = (@as(u64, 1) << atom_count) - 1;
    const combinations = @as(u64, 1) << @intCast(@as(u32, atom_count) * bound);
    var assignment: u64 = 0;
    while (assignment < combinations) : (assignment += 1) {
        for (0..bound) |offset| {
            extended[prefix + offset] =
                (assignment >> @intCast(offset * atom_count)) & per_cycle;
        }
        var ends = bruteForceMatches(ir, pattern, extended[0 .. prefix + bound], start);
        ends &= ~((@as(u64, 1) << @intCast(earliest)) - 1);
        if (ends != 0) return true;
    }
    return false;
}

fn bruteForceFate(
    ir: *const Ir,
    pattern: Ir.Node.Index,
    trace: []const u64,
    start: u32,
    atom_count: u5,
    bound: u32,
) Fate {
    const matches = bruteForceMatches(ir, pattern, trace, start);
    if (matches != 0) return .{ .matched_at = @ctz(matches) };

    var cycle = start;
    while (cycle < trace.len) : (cycle += 1) {
        if (!canStillMatch(ir, pattern, trace, start, cycle, cycle, atom_count, bound)) {
            return .{ .failed_at = cycle };
        }
    }
    return .{}; // still pending when the trace ran out
}

fn minOpt(left: ?u32, right: ?u32) ?u32 {
    const a = left orelse return right;
    const b = right orelse return a;
    return @min(a, b);
}

fn maxOpt(left: ?u32, right: ?u32) ?u32 {
    const a = left orelse return null;
    const b = right orelse return null;
    return @max(a, b);
}

/// Both operands must hold, on independent schedules: the pair is decided false
/// by the first side to fail, and true when the later side finally holds.
fn fateAnd(left: Fate, right: Fate) Fate {
    if (minOpt(left.failed_at, right.failed_at)) |at| return .{ .failed_at = at };
    if (maxOpt(left.matched_at, right.matched_at)) |at| return .{ .matched_at = at };
    return .{};
}

/// Either operand holding is enough, so the first to hold decides it true and it
/// takes both failing to decide it false.
fn fateOr(left: Fate, right: Fate) Fate {
    if (minOpt(left.matched_at, right.matched_at)) |at| return .{ .matched_at = at };
    if (maxOpt(left.failed_at, right.failed_at)) |at| return .{ .failed_at = at };
    return .{};
}

/// The whole of negation, as a verdict operation: the two decided values swap
/// and pending is a fixed point.
///
/// Note what it does *not* do — move a cycle. `not p` fails in the cycle `p`
/// matched, not in the earlier cycle from which `p`'s match had become
/// unavoidable. Those are different traces' worth of reports, and this is the
/// definition the lowering is checked against; `Plan.pending` is what the
/// lowering pays to meet it. Being an exact swap is also what makes this an
/// involution, which "not not p behaves exactly like p" pins.
fn fateNot(inner: Fate) Fate {
    return .{ .matched_at = inner.failed_at, .failed_at = inner.matched_at };
}

/// `p implies q` is `not p or q`, written out so the two operators are checked
/// against one definition rather than against each other.
fn fateImplies(left: Fate, right: Fate) Fate {
    return fateOr(fateNot(left), right);
}

/// What became of one attempt of a *property*, by direct recursion over the
/// definitions in `Ir`'s header.
///
/// Compositional, and deliberately: at the leaves it defers to `bruteForceFate`,
/// which is the reference the sequence layer is already tied to, and above them
/// it only ever combines verdicts. Nothing here knows that a conjunction lowers
/// onto `seq_and` or that a followed-by lowers onto a concatenation, which is
/// what makes it an independent opinion on those identities rather than a
/// restatement of them.
fn propFate(
    ir: *const Ir,
    property: Ir.Node.Index,
    trace: []const u64,
    start: u32,
    atom_count: u5,
    bound: u32,
) Fate {
    const data = ir.dataOf(property);
    switch (ir.tagOf(property)) {
        .prop_seq_weak, .prop_seq_strong => {
            return bruteForceFate(ir, @enumFromInt(data.lhs), trace, start, atom_count, bound);
        },
        .prop_not => {
            return fateNot(propFate(ir, @enumFromInt(data.lhs), trace, start, atom_count, bound));
        },
        .prop_and, .prop_or, .prop_implies, .prop_iff => {
            const left = propFate(ir, @enumFromInt(data.lhs), trace, start, atom_count, bound);
            const right = propFate(ir, @enumFromInt(data.rhs), trace, start, atom_count, bound);
            return switch (ir.tagOf(property)) {
                .prop_and => fateAnd(left, right),
                .prop_or => fateOr(left, right),
                .prop_implies => fateImplies(left, right),
                // Never reached from a monitor — every `iff` is refused for
                // `Error.MixedStrength`'s reason — but written out rather than
                // left unreachable, so that the day the strength model is
                // generalised this file already says what the answer is.
                .prop_iff => fateAnd(fateImplies(left, right), fateImplies(right, left)),
                else => unreachable,
            };
        },
        // The property at each offset in the range, folded disjunctively for
        // `prop_delay` and conjunctively for `prop_always`.
        //
        // An open range is not that fold with a bigger bound: the tail beyond the
        // trace is unobservable, and what the two operators do with it is exactly
        // what separates them. A disjunction over it can still come true later, so
        // no offset having matched leaves the attempt PENDING rather than failed;
        // a conjunction over it can never be completed, so every offset holding
        // leaves the attempt pending rather than matched. Which is
        // `s_eventually`'s "fails only at the end of the trace" and `always`'s
        // "never reports a match" respectively, arrived at from the definition
        // rather than from the automaton.
        .prop_delay, .prop_always => {
            const range = ir.propRangeOf(property);
            const conjunctive = ir.tagOf(property) == .prop_always;
            if (range.hi == Ir.unbounded) {
                var offset = range.lo;
                while (start + offset < trace.len) : (offset += 1) {
                    const at = propFate(ir, range.prop, trace, start + offset, atom_count, bound);
                    if (conjunctive) {
                        if (at.failed_at) |cycle| return .{ .failed_at = cycle };
                    } else if (at.matched_at) |cycle| return .{ .matched_at = cycle };
                }
                return .{};
            }
            var folded: ?Fate = null;
            var offset = range.lo;
            while (offset <= range.hi) : (offset += 1) {
                const at = propFate(ir, range.prop, trace, start + offset, atom_count, bound);
                folded = if (folded) |sofar|
                    (if (conjunctive) fateAnd(sofar, at) else fateOr(sofar, at))
                else
                    at;
            }
            return folded.?;
        },
        // The `until` family, straight off LRM 16.12.10 rather than off the
        // construction: `q` holds at some cycle `k`, and `p` holds at every cycle
        // in `[start, k)` — or `[start, k]` for the `_with` forms.
        //
        // So the candidate `k`s are bounded by the first cycle `p` fails in, and
        // only by it. Call that `broke`. For the plain forms `k` may be `broke`
        // itself, since `p` is not asked of that cycle; for `_with` it may only
        // reach `broke - 1`.
        //
        // Then: the attempt matches when some candidate's `q` matches, at the
        // earliest such cycle. It fails once no candidate is left and every one
        // raised has failed, which is the later of `broke` and the last of their
        // failures — a `q` still running keeps the attempt alive past `broke`. And
        // if `p` never failed inside the trace there are candidates the trace does
        // not decide, so the attempt is pending unless one already matched.
        .prop_until, .prop_until_with, .prop_s_until, .prop_s_until_with => {
            const left: Ir.Node.Index = @enumFromInt(data.lhs);
            const right: Ir.Node.Index = @enumFromInt(data.rhs);
            const inclusive = switch (ir.tagOf(property)) {
                .prop_until_with, .prop_s_until_with => true,
                else => false,
            };

            var broke: ?u32 = null;
            var cycle = start;
            while (cycle < trace.len) : (cycle += 1) {
                const held = propFate(ir, left, trace, cycle, atom_count, bound);
                if (held.matched_at == null) {
                    broke = cycle;
                    break;
                }
            }

            // The candidates the trace decides, as a half-open upper end.
            const decided: u32 = @intCast(trace.len);
            const limit = if (broke) |at|
                (if (inclusive) at else at + 1)
            else
                decided;

            var matched: ?u32 = null;
            var last_failure: ?u32 = null;
            var pending = false;
            var candidate = start;
            while (candidate < limit and candidate < decided) : (candidate += 1) {
                const witness = propFate(ir, right, trace, candidate, atom_count, bound);
                if (witness.matched_at) |at| {
                    matched = minOpt(matched, at);
                } else if (witness.failed_at) |at| {
                    last_failure = @max(last_failure orelse at, at);
                } else pending = true;
            }

            if (matched) |at| return .{ .matched_at = at };
            if (pending) return .{};
            // `p` still holding at the end of the trace leaves candidates the
            // trace cannot decide, so nothing is settled.
            const at = broke orelse return .{};
            return .{ .failed_at = @max(at, last_failure orelse at) };
        },
        // Existential over the antecedent's matches: some match, with the
        // consequent holding from its end. It stays pending while the antecedent
        // could still produce a new match, and fails once it cannot and every
        // witness it did produce has failed — which is where it parts company
        // with implication, for which no match at all is a pass.
        .prop_followed_by_overlap, .prop_followed_by_nonoverlap => {
            const antecedent: Ir.Node.Index = @enumFromInt(data.lhs);
            const gap: u32 = @intFromBool(ir.tagOf(property) == .prop_followed_by_nonoverlap);

            const ends = bruteForceMatches(ir, antecedent, trace, start);

            var matched: ?u32 = null;
            var last_hope: ?u32 = null;
            var pending = false;
            var remaining = ends;
            while (remaining != 0) {
                const end: u32 = @ctz(remaining);
                remaining &= remaining - 1;
                const witness = propFate(ir, @enumFromInt(data.rhs), trace, end + gap, atom_count, bound);
                if (witness.matched_at) |at| {
                    matched = minOpt(matched, at);
                } else if (witness.failed_at) |at| {
                    last_hope = @max(last_hope orelse at, at);
                } else pending = true;
            }
            if (matched) |at| return .{ .matched_at = at };
            if (pending) return .{};

            // Every witness so far has failed. The verdict is settled in the
            // first cycle from which the antecedent can no longer match at all —
            // a later match would be a fresh witness — provided the witnesses
            // already raised have finished failing by then.
            var cycle = start;
            while (cycle < trace.len) : (cycle += 1) {
                if (canStillMatch(ir, antecedent, trace, start, cycle, cycle + 1, atom_count, bound)) {
                    continue;
                }
                if (last_hope) |at| if (at > cycle) continue;
                return .{ .failed_at = cycle };
            }
            return .{};
        },
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
        => {
            const inner = propFate(ir, @enumFromInt(data.rhs), trace, start, atom_count, bound);
            const accepts = switch (ir.tagOf(property)) {
                .prop_accept_on, .prop_sync_accept_on => true,
                else => false,
            };

            // The first cycle the condition holds while the attempt is still
            // undecided. LRM 16.12.7 arms an abort at any point *during* the
            // evaluation attempt, and the cycle the attempt resolves in is one
            // of those — hence `>=` rather than `>`, and hence an abort winning
            // a tie against the property's own verdict.
            var fired: ?u32 = null;
            var cycle = start;
            while (cycle < trace.len) : (cycle += 1) {
                const atoms = [_]Word{trace[cycle]};
                if (!Nfa.evaluateBoolean(ir, @enumFromInt(data.lhs), &atoms)) continue;
                if (inner.matched_at) |at| if (at < cycle) break;
                if (inner.failed_at) |at| if (at < cycle) break;
                fired = cycle;
                break;
            }

            const at = fired orelse return inner;
            if (accepts) {
                if (inner.matched_at) |held| if (held < at) return inner;
                return .{ .matched_at = at };
            }
            if (inner.matched_at) |held| if (held < at) return inner;
            return .{ .failed_at = at };
        },
        else => unreachable, // an implication is peeled off before this
    }
}

/// The strength a property carries, which is its leaves' — every operator above
/// them passes it through, `prop_not` inverting as it goes, and the junctions
/// constrain their sides rather than choosing.
fn propStrength(ir: *const Ir, property: Ir.Node.Index) Monitor.Strength {
    const data = ir.dataOf(property);
    return switch (ir.tagOf(property)) {
        .prop_seq_weak => .weak,
        .prop_seq_strong => .strong,
        .prop_delay, .prop_always => propStrength(ir, ir.propRangeOf(property).prop),
        // The one family whose strength does not come from its leaves: LRM
        // 16.12.10 makes the weak/strong distinction the operator's own, which is
        // why there are four tags rather than one with a flag.
        .prop_until, .prop_until_with => .weak,
        .prop_s_until, .prop_s_until_with => .strong,
        // LRM 16.12.2's duality. A pending attempt is still pending under
        // negation, so all that turns over is what the end of the trace makes
        // of it — which is the whole of `Monitor.Strength`.
        .prop_not => propStrength(ir, @enumFromInt(data.lhs)).not(),
        .prop_and,
        .prop_or,
        .prop_followed_by_overlap,
        .prop_followed_by_nonoverlap,
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
        .prop_implies_overlap,
        .prop_implies_nonoverlap,
        // `p implies q` is `not p or q`, whose strength is the disjunction's,
        // which is `q`'s — the lowering having already required `not p` to
        // agree with it. `p iff q` would be its left conjunct's, likewise `q`'s,
        // if it ever lowered.
        .prop_implies,
        .prop_iff,
        => propStrength(ir, @enumFromInt(data.rhs)),
        else => unreachable, // not a property-layer tag
    };
}

/// Check a monitor against the semantics over every trace of `length` cycles.
///
/// `leaf_bound` is the continuation length `bruteForceFate` enumerates at the
/// leaves, for a property whose obligation has no bounded longest match: a
/// counted property's whole point is that it has no bound to borrow, so the
/// caller supplies one that covers its leaves. Null takes the property's own,
/// which is what a ring-tracked shape has.
fn differentialMonitor(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    atom_count: u5,
    length: u32,
    leaf_bound: ?u32,
) !void {
    var monitor = try assertion(gpa, ir, index, .{}, null);
    defer monitor.deinit(gpa);

    // Re-derive the pieces from the IR rather than reading them off the
    // monitor, so the reference does not inherit the lowering's decomposition.
    const directive = ir.assertions.get(@intFromEnum(index));
    var property = directive.prop;
    var antecedent: ?Ir.Node.Index = null;
    var delay: u32 = 0;
    switch (ir.tagOf(property)) {
        .prop_implies_overlap, .prop_implies_nonoverlap => {
            antecedent = @as(Ir.Node.Index, @enumFromInt(ir.dataOf(property).lhs));
            delay = @intFromBool(ir.tagOf(property) == .prop_implies_nonoverlap);
            property = @enumFromInt(ir.dataOf(property).rhs);
        },
        else => {},
    }
    const strong = propStrength(ir, property) == .strong;
    // The continuation length `bruteForceFate` enumerates at the leaves. The
    // whole property's bound covers every leaf's, so it is safe if generous.
    const bound: u32 = leaf_bound orelse @intCast(propBound(ir, property).length.?);

    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |window| {

        // How many obligations start in each cycle: one every cycle without a
        // trigger, and otherwise one per antecedent match ending in it, shifted
        // by the implication's delay.
        //
        // Per match, not per cycle. LRM 16.14.1 starts an evaluation of the
        // consequent for every match of the antecedent, so three matches ending
        // together are three attempts owing three verdicts — a bitmask of end
        // cycles cannot say that, and a reference that used one would agree with
        // a monitor that collapsed them.
        var starts = [_]u32{0} ** 8;
        if (antecedent) |node| {
            for (0..length) |start| {
                var ends = bruteForceMatches(ir, node, window, @intCast(start));
                while (ends != 0) {
                    const end = @ctz(ends);
                    ends &= ends - 1;
                    const seeded = end + delay;
                    if (seeded < length) starts[seeded] += 1;
                }
            }
        } else for (starts[0..length]) |*count| {
            count.* = 1;
        }

        var want_failures = [_]u32{0} ** 8;
        var want_matches = [_]u32{0} ** 8;
        var want_pending: u32 = 0;
        for (starts[0..length], 0..) |count, start| {
            if (count == 0) continue;
            const fate = propFate(ir, property, window, @intCast(start), atom_count, bound);
            if (fate.matched_at) |at| {
                want_matches[at] += count;
            } else if (fate.failed_at) |at| {
                want_failures[at] += count;
            } else want_pending += count;
        }

        state.clear();
        for (0..length) |cycle| {
            const atoms = [_]Word{window[cycle]};
            const report = state.stepAtoms(ir, &atoms);
            if (report.failures != want_failures[cycle] or report.matches != want_matches[cycle]) {
                std.debug.print(
                    "trace bits {b}, cycle {d}: want {d} failures / {d} matches," ++
                        " got {d} / {d}\n",
                    .{
                        traces.bits,         cycle,           want_failures[cycle],
                        want_matches[cycle], report.failures, report.matches,
                    },
                );
                return error.TestUnexpectedResult;
            }
        }

        const final = state.finish();
        const want_final: u32 = if (strong) want_pending else 0;
        if (final.failures != want_final) {
            std.debug.print(
                "trace bits {b}: {d} pending at end, want {d} failures, got {d}\n",
                .{ traces.bits, want_pending, want_final, final.failures },
            );
            return error.TestUnexpectedResult;
        }
    }
}

/// One trigger-covering abort, for `differentialAbortOverImplication`. Listed
/// innermost first, the order lowering appends them in.
const TriggerAbort = struct { cond: Ir.Node.Index, effect: Monitor.Abort.Effect };

/// Whether one antecedent attempt is still being matched in a cycle, for an
/// antecedent spelled `left ##[lo:hi] right` over two one-cycle booleans.
///
/// Read off the operator rather than off the automaton, which is the point of it.
/// A match is an interval `[s, e]` with `left` holding at `s`, `right` at `e`, and
/// `e - s` in `lo..hi` (LRM 16.9 and the delay's own definition). The attempt from
/// `s` is therefore still being matched in cycle `t` when `left` held at `s` and
/// some `e >= t` could still complete it: either `t` is early enough that a later
/// `e` remains available, `t < s + hi`, or `t` is itself such an `e`. The cycles
/// in between constrain nothing — a delay range says nothing about them — which
/// is why nothing else appears here.
const InFlight = struct {
    /// Boolean-layer nodes, not the one-cycle sequences the antecedent is built
    /// out of: this evaluates them per cycle.
    left: Ir.Node.Index,
    right: Ir.Node.Index,
    lo: u32,
    hi: u32,

    fn alive(spec: InFlight, ir: *const Ir, window: []const Word, start: u32, cycle: u32) bool {
        const opening = [_]Word{window[start]};
        if (!Nfa.evaluateBoolean(ir, spec.left, &opening)) return false;
        const span = cycle - start;
        // Still early enough that a later end remains.
        if (span < spec.hi) return true;
        // Or this cycle is an end in its own right.
        if (span > spec.hi or span < spec.lo) return false;
        const closing = [_]Word{window[cycle]};
        return Nfa.evaluateBoolean(ir, spec.right, &closing);
    }
};

/// Check an abort *above* an implication against the semantics over every trace
/// of `length` cycles.
///
/// `differentialMonitor` cannot do this shape: it peels an implication only at
/// the very top of a property, and here the aborts sit above one. The reference
/// is assembled from the pieces the shape is defined out of, none of them read
/// off the monitor — which cycles an antecedent match from a given start ends
/// in, when an abort covering the antecedent catches that attempt, and the fate
/// of what is seeded when no abort intervenes.
///
/// It walks one ATTEMPT at a time rather than one cycle at a time, because that
/// is what the property means: an evaluation starts every cycle, runs while its
/// antecedent can still match, and an abort above the implication ends it. Ending
/// it ends its antecedent too, so a match the trace would have completed later
/// does not happen — the attempt it belonged to is already resolved.
///
/// `seeded` is the last piece: the same abort chain wrapped around the consequent
/// alone, which is what an attempt owes once its antecedent has matched. The
/// caller builds it separately, so agreeing with it is a claim about the lowering
/// rather than a restatement of it.
///
/// `in_flight` says when an attempt is still matching a multi-cycle antecedent,
/// and is null for one that matches in a single cycle — where an attempt is in
/// flight for exactly the cycle it starts in.
fn differentialAbortOverImplication(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    antecedent: Ir.Node.Index,
    overlap: bool,
    seeded: Ir.Node.Index,
    aborts: []const TriggerAbort,
    in_flight: ?InFlight,
    atom_count: u5,
    length: u32,
) !void {
    var monitor = try assertion(gpa, ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    const strong = propStrength(ir, seeded) == .strong;
    const bound: u32 = @intCast(propBound(ir, seeded).length.?);
    const delay: u32 = @intFromBool(!overlap);

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |window| {
        var want_failures = [_]u32{0} ** 8;
        var want_matches = [_]u32{0} ** 8;
        var want_pending: u32 = 0;

        // One attempt per cycle, each walked forward for as long as its
        // antecedent can still match.
        for (0..length) |start| {
            const ends = bruteForceMatches(ir, antecedent, window, @intCast(start));
            for (start..length) |cycle| {
                const here = @as(u64, 1) << @intCast(cycle);
                const alive = if (in_flight) |spec|
                    spec.alive(ir, window, @intCast(start), @intCast(cycle))
                else
                    cycle == start and ends & here != 0;
                if (!alive) break;

                // The abort covers the antecedent, so it resolves this attempt
                // wherever it has got to — including in the cycle a match of it
                // ends, which is why this is tested first. The outermost one that
                // holds wins, which is the last: the same tie rule `propFate`
                // applies when an outer abort meets an inner verdict in one cycle.
                const atoms = [_]Word{window[cycle]};
                var verdict: ?Monitor.Abort.Effect = null;
                for (aborts) |abort| {
                    if (Nfa.evaluateBoolean(ir, abort.cond, &atoms)) verdict = abort.effect;
                }
                if (verdict) |effect| {
                    switch (effect) {
                        .accept => want_matches[cycle] += 1,
                        .reject => want_failures[cycle] += 1,
                    }
                    break;
                }

                // Otherwise a match ending here starts an obligation, still under
                // the same aborts, and the attempt runs on: an antecedent that has
                // matched can match again. A `|=>` match ending in the last cycle
                // seeds nothing at all.
                if (ends & here == 0) continue;
                const seeding = cycle + delay;
                if (seeding >= length) continue;
                const fate = propFate(ir, seeded, window, @intCast(seeding), atom_count, bound);
                if (fate.matched_at) |at| {
                    want_matches[at] += 1;
                } else if (fate.failed_at) |at| {
                    want_failures[at] += 1;
                } else want_pending += 1;
            }
        }

        state.clear();
        for (0..length) |cycle| {
            const atoms = [_]Word{window[cycle]};
            const report = state.stepAtoms(ir, &atoms);
            if (report.failures != want_failures[cycle] or report.matches != want_matches[cycle]) {
                std.debug.print(
                    "trace bits {b}, cycle {d}: want {d} failures / {d} matches," ++
                        " got {d} / {d}\n",
                    .{
                        traces.bits,         cycle,           want_failures[cycle],
                        want_matches[cycle], report.failures, report.matches,
                    },
                );
                return error.TestUnexpectedResult;
            }
        }

        const final = state.finish();
        const want_final: u32 = if (strong) want_pending else 0;
        if (final.failures != want_final) {
            std.debug.print(
                "trace bits {b}: {d} pending at end, want {d} failures, got {d}\n",
                .{ traces.bits, want_pending, want_final, final.failures },
            );
            return error.TestUnexpectedResult;
        }
    }
}

fn addDirective(
    ir: *Ir,
    gpa: Allocator,
    property: Ir.Node.Index,
    reset: ?Ir.Node.Index,
) !Ir.Assertion.Index {
    return ir.addAssertion(gpa, .{
        .id = @enumFromInt(0),
        .kind = .assert,
        .clock = .none,
        // Never resetting is a constant false, not an absent edge.
        .reset = reset orelse try ir.boolConst(gpa, false),
        .prop = property,
    });
}

/// Every failure a monitor reports over `trace`, including what the end of the
/// trace makes of an attempt still pending there.
fn failuresOver(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    trace: []const u64,
) !u64 {
    var monitor = try assertion(gpa, ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    state.clear();
    var failures: u64 = 0;
    for (trace) |cycle| {
        const atoms = [_]Word{cycle};
        failures += state.stepAtoms(ir, &atoms).failures;
    }
    return failures + state.finish().failures;
}

test "differential: implication" {
    const gpa = testing.allocator;

    const Shape = enum { overlap, nonoverlap };
    for ([_]Shape{ .overlap, .nonoverlap }) |shape| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            const tag: Ir.Node.Tag = switch (shape) {
                .overlap => .prop_implies_overlap,
                .nonoverlap => .prop_implies_nonoverlap,
            };

            // `req |-> ack` and `req |=> ack`: the one-cycle consequent, where
            // an overlapping implication has to discharge in the very cycle it
            // seeded the obligation.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 1));
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }

            // A multi-cycle consequent, so more than one attempt is in flight.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 1),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 0),
                );
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }

            // A consequent whose matches vary in length, so an attempt can be
            // discharged at more than one age.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try leadingDelay(&ir, gpa, 1, 2, try atomCycle(&ir, gpa, 1));
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }

            // A multi-cycle antecedent, so triggers are not simply "req held".
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const trigger = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 1),
                );
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const property = try ir.propImplies(gpa, tag, trigger, consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 5, null);
            }

            // An antecedent whose matches VARY in length, which is the shape a
            // trigger carrying one bit gets wrong: `a ##[1:3] b` can have three
            // attempts end a match in the same cycle, started three, two and one
            // cycles earlier, and each of them owes a verdict of its own. The
            // reference counts matches rather than matching cycles, so a monitor
            // that collapsed them would fail here and only here.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const trigger = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    3,
                    try atomCycle(&ir, gpa, 1),
                );
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const property = try ir.propImplies(gpa, tag, trigger, consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 5, null);
            }

            // The same with a multi-cycle consequent, so the ring is holding
            // several slots at once AND some of them stand for several attempts.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const trigger = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    2,
                    try atomCycle(&ir, gpa, 1),
                );
                const body = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 1),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 0),
                );
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, trigger, consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 5, null);
            }

            // An antecedent one attempt can reach the end of along TWO paths in
            // the same cycle: `(a ##[1:2] b) ##[1:2] c` matches [s, s + 3] with
            // gaps of 1 then 2 and of 2 then 1. That is one match over one
            // interval, so it owes ONE verdict — and a trigger that added up
            // threads per state instead of unioning attempt identities would
            // report two. This is the case that separates the two, and nothing
            // else here does.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const head = try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    2,
                    try atomCycle(&ir, gpa, 1),
                );
                const trigger = try ir.seqDelay(gpa, head, 1, 2, try atomCycle(&ir, gpa, 2));
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const property = try ir.propImplies(gpa, tag, trigger, consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 3, 5, null);
            }

            // A consequent with a *unique* match length, so an attempt cannot
            // resolve early and `depth` of them really are in flight at once.
            // This is the case that exercises the ring: the slot the next
            // attempt takes is free only because the attempt that aged out of
            // it was forced to resolve.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.seqDelay(
                    gpa,
                    try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 1, try atomCycle(&ir, gpa, 0)),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 1),
                );
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 5, null);
            }

            // A repetition that cannot match at its first cycle, and a
            // disjunction whose branches differ in length: both are cases where
            // a too-small bound would silently lose an attempt.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.seqRepeat(
                    gpa,
                    .seq_repeat_consec,
                    try atomCycle(&ir, gpa, 1),
                    2,
                    3,
                );
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 5, null);
            }
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.seqOr(
                    gpa,
                    try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 1, try atomCycle(&ir, gpa, 1)),
                    try atomCycle(&ir, gpa, 0),
                );
                const consequent = try ir.propSeq(gpa, strength, body);
                const property = try ir.propImplies(gpa, tag, try atomCycle(&ir, gpa, 0), consequent);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
        }
    }
}

test "several antecedent matches ending in one cycle each owe a verdict" {
    const gpa = testing.allocator;

    // The reduced repro, cycle for cycle. `(a ##[1:3] b) |-> c` with `a` held for
    // three cycles and `b` arriving in the fourth: the attempts started in cycles
    // 0, 1 and 2 all have their antecedent match END in cycle 3, at gaps of three,
    // two and one. LRM 16.14.1 starts an evaluation of the consequent per match,
    // `c` is low there, so three attempts fail in that one cycle.
    //
    // A trigger carrying one bit reports one, which is the bug this exists to
    // catch. The number is not a matter of taste: both commercial oracles report
    // three, and `tests/sva/diff_strongprop.sv` records Questa doing it.
    const trace = [_]u64{
        0b001, // a
        0b001, // a
        0b001, // a
        0b010, // b, and c low: every match ends here and owes a failure
        0b000,
    };

    const Case = struct { kind: Ir.Assertion.Kind, c_at_end: bool };
    for ([_]Case{
        .{ .kind = .assert, .c_at_end = false },
        // The same shape as a `cover`, where the count is the hit TALLY rather
        // than a log line: three matches of the antecedent, three attempts
        // discharged, three hits. A collapsing monitor under-counts coverage
        // silently, with nothing anywhere reporting a failure to notice.
        .{ .kind = .cover, .c_at_end = true },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);

        const property = try ir.propImplies(
            gpa,
            .prop_implies_overlap,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1)),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
        );
        const index = try ir.addAssertion(gpa, .{
            .id = @enumFromInt(0),
            .kind = case.kind,
            .clock = .none,
            .reset = try ir.boolConst(gpa, false),
            .prop = property,
        });

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        // Four ids: an attempt's threads live for as long as its longest match,
        // which is four cycles here, and no two attempts that far apart are ever
        // in flight together.
        try testing.expectEqual(@as(u32, 4), monitor.trigger_window);

        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);

        for (trace, 0..) |cycle, at| {
            const atoms = [_]Word{if (at == 3 and case.c_at_end) cycle | 0b100 else cycle};
            const report = state.stepAtoms(&ir, &atoms);
            const owed: u64 = if (at == 3) 3 else 0;
            if (case.c_at_end) {
                try testing.expectEqual(owed, report.matches);
                try testing.expectEqual(@as(u64, 0), report.failures);
            } else {
                try testing.expectEqual(owed, report.failures);
                try testing.expectEqual(@as(u64, 0), report.matches);
            }
        }
        // Nothing is left over: the three attempts were resolved in cycle 3, and
        // the ones started in cycles 3 and 4 never matched an antecedent.
        try testing.expectEqual(@as(u64, 0), state.finish().failures);
    }
}

test "one attempt matching twice owes a verdict for each match" {
    const gpa = testing.allocator;

    // The other half of "per match, not per cycle": one attempt can owe several
    // verdicts. Same property, with `a` in cycles 0 and 2 and `b` in 1 and 3, so
    // the antecedent matches over [0,1], [0,3] and [2,3] — the attempt from cycle
    // 0 matches TWICE, two cycles apart, and cycle 3 holds one match from it and
    // one from the attempt started in cycle 2.
    //
    // So the count is 1 in cycle 1 and 2 in cycle 3, and both numbers are load
    // bearing. A monitor counting matching cycles reports 1 in cycle 3, and so
    // does one that retired the first attempt when it first matched: an attempt
    // whose antecedent matches keeps matching.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const index = try addDirective(&ir, gpa, try ir.propImplies(
        gpa,
        .prop_implies_overlap,
        try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1)),
        try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
    ), null);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    const trace = [_]u64{ 0b001, 0b010, 0b001, 0b010, 0b000 };
    const want = [_]u64{ 0, 1, 0, 2, 0 };
    for (trace, want) |cycle, owed| {
        const atoms = [_]Word{cycle};
        try testing.expectEqual(owed, state.stepAtoms(&ir, &atoms).failures);
    }
    try testing.expectEqual(@as(u64, 0), state.finish().failures);
}

test "two paths of one attempt ending together are one match" {
    const gpa = testing.allocator;

    // Why the per-state quantity is a SET and not an integer. `(a ##[1:2] b)
    // ##[1:2] c` with `a` in cycle 0, `b` in cycles 1 and 2 and `c` in cycle 3
    // reaches the end of the antecedent from cycle 0 along two paths at once —
    // gaps of one then two, and of two then one — and both end in cycle 3.
    //
    // That is ONE match: a match is an interval, and both paths cover [0, 3]. So
    // one attempt, one obligation, one failure. Unioning attempt identities gives
    // that for free; adding up threads per state answers two, and no other test
    // here can tell the difference.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const head = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
    const index = try addDirective(&ir, gpa, try ir.propImplies(
        gpa,
        .prop_implies_overlap,
        try ir.seqDelay(gpa, head, 1, 2, try atomCycle(&ir, gpa, 2)),
        try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 3)),
    ), null);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    // a, then b twice, then c — and atom 3, the consequent, never holds.
    const trace = [_]u64{ 0b0001, 0b0010, 0b0010, 0b0100, 0b0000 };
    const want = [_]u64{ 0, 0, 0, 1, 0 };
    for (trace, want) |cycle, owed| {
        const atoms = [_]Word{cycle};
        try testing.expectEqual(owed, state.stepAtoms(&ir, &atoms).failures);
    }
    try testing.expectEqual(@as(u64, 0), state.finish().failures);
}

test "differential: a bare sequence property" {
    const gpa = testing.allocator;

    // No trigger, so an obligation starts every cycle. This is the case where
    // per-attempt tracking is most obviously load-bearing: with `depth`
    // attempts in flight at once, merging them into one state vector would hide
    // every failure that overlaps a later success.
    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const body = try ir.seqDelay(
                gpa,
                try atomCycle(&ir, gpa, 0),
                1,
                1,
                try atomCycle(&ir, gpa, 1),
            );
            const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
            try differentialMonitor(gpa, &ir, index, 2, 4, null);
        }
        {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const body = try ir.seqRepeat(
                gpa,
                .seq_repeat_consec,
                try atomCycle(&ir, gpa, 0),
                1,
                3,
            );
            const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
            try differentialMonitor(gpa, &ir, index, 1, 5, null);
        }
    }
}

test "differential: an and obligation" {
    const gpa = testing.allocator;

    // An `and` under per-attempt tracking. Its two sides can finish in either
    // order, and an attempt is only discharged once the later one has, so a
    // wait copy that accepted too early would show up here as a match reported
    // a cycle before the semantics allow one.
    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.seqAnd(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1)),
            try atomCycle(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
        try differentialMonitor(gpa, &ir, index, 2, 4, null);
    }
}

test "differential: a throughout obligation" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.seqThroughout(
            gpa,
            try ir.atom(gpa, @enumFromInt(0)),
            try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 1), 1, 3),
        );
        const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
        try differentialMonitor(gpa, &ir, index, 2, 4, null);
    }
}

test "differential: a first match obligation" {
    const gpa = testing.allocator;

    // Per-attempt tracking over an obligation whose states are subsets, whose
    // exits have empty follow rows, and which therefore goes empty the cycle
    // it matches. That is what is being checked: the monitor path handles the
    // shape. It is *not* a check of the suppression, which is invisible from
    // here — see the test below for why, and for where it is visible.
    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.seqFirstMatch(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 3, try atomCycle(&ir, gpa, 1)),
        );
        const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
        try differentialMonitor(gpa, &ir, index, 2, 4, null);
    }
}

test "first match is invisible as a property and visible as an antecedent" {
    const gpa = testing.allocator;

    // Where the operator can be observed at all, which is not everywhere.
    //
    // A sequence property holds as soon as its sequence completes a match —
    // the earliest, which is exactly the one `first_match` keeps — and fails
    // as soon as it can no longer match, which while pending is the same
    // question for both, because a match having ended would already have made
    // it hold. Pending-at-end coincides too. So over a *property* the operator
    // is a no-op on the verdict, whatever its strength.
    //
    // It bites where the operand's later matches are themselves observable:
    // as an antecedent, because every antecedent match raises an obligation of
    // its own, and inside a larger sequence, because a suppressed match is a
    // continuation that never happens.
    //
    // Bit 0 is `a`, bit 1 is `b`, bit 2 is `c`. `a ##[1:2] b` matches from
    // cycle 0 at cycle 1 *and* at cycle 2, which is what gives the suppression
    // something to do.
    const paired = [_]u64{ 0b001, 0b010, 0b110, 0b000 };

    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        var verdict: [2]u64 = undefined;
        for ([_]bool{ false, true }, &verdict) |suppress, *into| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            var body = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
            if (suppress) body = try ir.seqFirstMatch(gpa, body);
            const property = try ir.propImplies(
                gpa,
                .prop_implies_overlap,
                try atomCycle(&ir, gpa, 2),
                try ir.propSeq(gpa, strength, body),
            );
            into.* = try failuresOver(gpa, &ir, try addDirective(&ir, gpa, property, null), &paired);
        }
        try testing.expectEqual(verdict[0], verdict[1]);
    }

    // As an antecedent the later match raises an obligation of its own, and
    // that one is discharged by nothing.
    var raised: [2]u64 = undefined;
    for ([_]bool{ false, true }, &raised) |suppress, *into| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        var trigger = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
        if (suppress) trigger = try ir.seqFirstMatch(gpa, trigger);
        const property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            trigger,
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
        );
        into.* = try failuresOver(gpa, &ir, try addDirective(&ir, gpa, property, null), &paired);
    }
    try testing.expectEqual(@as(u32, 1), raised[0]);
    try testing.expectEqual(@as(u32, 0), raised[1]);

    // And inside a larger sequence it is the difference between a match and
    // none at all: with `c` only in cycle 3, the run-on is reachable from the
    // operand's later match and from nothing else.
    const late = [_]u64{ 0b001, 0b010, 0b010, 0b100 };
    var reached: [2]u64 = undefined;
    for ([_]bool{ false, true }, &reached) |suppress, *into| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        var inner = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 2, try atomCycle(&ir, gpa, 1));
        if (suppress) inner = try ir.seqFirstMatch(gpa, inner);
        const whole = try ir.seqDelay(gpa, inner, 1, 1, try atomCycle(&ir, gpa, 2));
        into.* = try endsOf(gpa, &ir, whole, &late, 0);
    }
    try testing.expectEqual(@as(u64, 0b1000), reached[0]);
    try testing.expectEqual(@as(u64, 0), reached[1]);
}

test "a first match obligation inherits its operand's bound" {
    const gpa = testing.allocator;

    {
        // `req |=> first_match(##[1:$] ack)`. Taking only the earliest match
        // does not bound how late it arrives, so there is still no ring to size —
        // which is now the counted discipline's job rather than a refusal.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.seqFirstMatch(
            gpa,
            try leadingDelay(&ir, gpa, 1, Ir.unbounded, try atomCycle(&ir, gpa, 1)),
        );
        const consequent = try ir.propSeq(gpa, .prop_seq_weak, body);
        const property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            consequent,
        );
        var monitor = try assertion(
            gpa,
            &ir,
            try addDirective(&ir, gpa, property, null),
            .{},
            null,
        );
        defer monitor.deinit(gpa);
        try testing.expectEqual(Monitor.Mode.counted, monitor.mode);
        try testing.expectEqual(@as(u32, 0), monitor.depth);
    }
    {
        // The conservative side of inheriting, pinned so that tightening it is a
        // visible change: `first_match(a[*1:$])` is just `a`, and a longest path
        // over the automaton would say so — but `maxLength` reads the tree, so
        // the answer is a determinised obligation rather than a ring one deep.
        // Cheap either way at this size; what is pinned is that the tree-reading
        // is what decides.
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.seqFirstMatch(
            gpa,
            try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 1), 1, Ir.unbounded),
        );
        const consequent = try ir.propSeq(gpa, .prop_seq_weak, body);
        const property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            consequent,
        );
        var monitor = try assertion(
            gpa,
            &ir,
            try addDirective(&ir, gpa, property, null),
            .{},
            null,
        );
        defer monitor.deinit(gpa);
        try testing.expectEqual(Monitor.Mode.counted, monitor.mode);

        // The sequence itself lowers fine, and to exactly `a`: one state, which
        // is the shortening the bound declines to notice.
        try testing.expectEqual(@as(u32, 1), try stateCount(gpa, &ir, body));
    }
}

test "no match survives first match that outlives the depth it reports" {
    const gpa = testing.allocator;

    // `depth` is an over-estimate here rather than the exact longest match, so
    // the property that matters is the inequality: every match kept is one of
    // the operand's, and none of them reaches past the bound the obligation
    // was allocated for.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const operand = try ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(&ir, gpa, 0), 1, 4);
    const kept = try ir.seqFirstMatch(gpa, operand);
    const depth = maxLength(&ir, kept).?;

    var all = try sequence(gpa, &ir, operand, .{}, null);
    defer all.deinit(gpa);
    var one = try sequence(gpa, &ir, kept, .{}, null);
    defer one.deinit(gpa);
    var over_all = try Runner.init(gpa, &all, &ir);
    defer over_all.deinit(gpa);
    var over_one = try Runner.init(gpa, &one, &ir);
    defer over_one.deinit(gpa);

    var traces: Traces = .init(1, 6);
    while (traces.next()) |trace| {
        for (0..trace.len) |index| {
            const start: u32 = @intCast(index);
            const ends = over_one.matches(trace, start);
            // Kept matches are a subset of the operand's...
            try testing.expectEqual(ends, ends & over_all.matches(trace, start));
            // ...and none of them spans more cycles than were allocated.
            if (ends != 0) try testing.expect(@ctz(ends) - start + 1 <= depth);
        }
    }
}

test "an obligation that prunes to nothing fails every attempt" {
    const gpa = testing.allocator;

    // `throughout` and `and` make an empty automaton easy to reach — a boolean
    // that contradicts the operand it guards, or two sides that cannot both
    // start — and an obligation with no states at all is a shape a monitor has
    // to handle rather than a malformed one. Every attempt is seeded, finds no
    // live state and fails in the cycle it started, which is the case the
    // `occupied` flag exists for.
    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const body = try ir.seqThroughout(
                gpa,
                try ir.boolConst(gpa, false),
                try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, try atomCycle(&ir, gpa, 1)),
            );
            const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
            try differentialMonitor(gpa, &ir, index, 2, 4, null);
        }
        {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const a = try ir.atom(gpa, @enumFromInt(0));
            const body = try ir.seqAnd(
                gpa,
                try ir.seqBoolean(gpa, a),
                try ir.seqBoolean(gpa, try ir.bNot(gpa, a)),
            );
            const index = try addDirective(&ir, gpa, try ir.propSeq(gpa, strength, body), null);
            try differentialMonitor(gpa, &ir, index, 2, 4, null);
        }
    }
}

test "depth is the obligation's longest match" {
    const gpa = testing.allocator;

    // `depth` is how many state vectors a backend allocates, so an
    // over-estimate is invisible at runtime and an under-estimate loses
    // attempts. Neither shows up in a verdict, so the bound is pinned directly.
    const Case = struct { length: u32, build: *const fn (*Ir, Allocator) anyerror!Ir.Node.Index };
    const cases = [_]Case{
        .{ .length = 1, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return atomCycle(ir, gpa_, 1);
            }
        }.make },
        // `a ##1 b` spans two cycles: the delay is the gap, not the span.
        .{ .length = 2, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 1, try atomCycle(ir, gpa_, 1));
            }
        }.make },
        // `a ##0 b` shares one, so it spans one.
        .{ .length = 1, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 0, 0, try atomCycle(ir, gpa_, 2));
            }
        }.make },
        // A leading delay is measured from the attempt's start, so `##[1:2] a`
        // reaches one cycle further than `x ##[1:2] a` would past `x`.
        .{ .length = 3, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return leadingDelay(ir, gpa_, 1, 2, try atomCycle(ir, gpa_, 1));
            }
        }.make },
        .{ .length = 3, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqRepeat(gpa_, .seq_repeat_consec, try atomCycle(ir, gpa_, 1), 2, 3);
            }
        }.make },
        // A disjunction is as long as its longest branch.
        .{ .length = 2, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqOr(
                    gpa_,
                    try ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 1, try atomCycle(ir, gpa_, 1)),
                    try atomCycle(ir, gpa_, 2),
                );
            }
        }.make },
        // An intersection is as long as its *shorter* branch, because both have
        // to span the same cycles.
        .{ .length = 2, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqIntersect(
                    gpa_,
                    try ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 1, try atomCycle(ir, gpa_, 1)),
                    try ir.seqRepeat(gpa_, .seq_repeat_consec, try atomCycle(ir, gpa_, 2), 1, 5),
                );
            }
        }.make },
        // An `and` is as long as its *longer* branch, which is the one place
        // its bound parts company with `intersect`'s.
        .{ .length = 3, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqAnd(
                    gpa_,
                    try ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 1, try atomCycle(ir, gpa_, 1)),
                    try ir.seqRepeat(gpa_, .seq_repeat_consec, try atomCycle(ir, gpa_, 2), 1, 3),
                );
            }
        }.make },
        // A boolean required throughout rides on the operand's cycles and adds
        // none of its own.
        .{ .length = 2, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqThroughout(
                    gpa_,
                    try ir.atom(gpa_, @enumFromInt(2)),
                    try ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 1, try atomCycle(ir, gpa_, 1)),
                );
            }
        }.make },
        // `first_match` keeps a subset of its operand's matches, so it can only
        // shorten the longest one. The bound is the operand's, which here is
        // exact: with `b` first holding in the last cycle of the range, the
        // earliest match is also the longest one there is.
        .{ .length = 3, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqFirstMatch(
                    gpa_,
                    try ir.seqDelay(gpa_, try atomCycle(ir, gpa_, 1), 1, 2, try atomCycle(ir, gpa_, 2)),
                );
            }
        }.make },
        // And here it is a strict over-estimate: every match of `a[*1:3]` has
        // the one-cycle match as a prefix, so the earliest is always one cycle
        // and three vectors are allocated where one would do. Over-estimating
        // costs memory; under-estimating would lose attempts.
        .{ .length = 3, .build = struct {
            fn make(ir: *Ir, gpa_: Allocator) anyerror!Ir.Node.Index {
                return ir.seqFirstMatch(
                    gpa_,
                    try ir.seqRepeat(gpa_, .seq_repeat_consec, try atomCycle(ir, gpa_, 1), 1, 3),
                );
            }
        }.make },
    };

    for (cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const consequent = try ir.propSeq(gpa, .prop_seq_weak, try case.build(&ir, gpa));
        const property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            consequent,
        );
        var monitor = try assertion(gpa, &ir, try addDirective(&ir, gpa, property, null), .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(case.length, monitor.depth);
    }
}

test "req implies ack the cycle after" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        consequent,
    );
    const index = try addDirective(&ir, gpa, property, null);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);

    // One cycle of obligation, so one state vector, and one bit of trigger.
    try testing.expectEqual(@as(u32, 1), monitor.depth);
    try testing.expectEqual(@as(u32, 1), monitor.obligation.state_count);
    try testing.expectEqual(@as(u32, 1), monitor.trigger.?.state_count);
    try testing.expectEqual(Monitor.Strength.weak, monitor.strength);
    // Never resetting is a constant false root, not an absent one.
    try testing.expect(monitor.reset < monitor.alphabet.roots.len);
    const never = monitor.alphabet.roots[monitor.reset];
    try testing.expectEqual(Ir.Node.Tag.bool_const, ir.tagOf(never));
    try testing.expectEqual(@as(u32, 0), ir.dataOf(never).lhs);

    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    // Bit 0 is `req`, bit 1 is `ack`.
    const trace = [_]u64{
        0b01, // req, no ack        -> obligation for cycle 1
        0b11, // req and ack        -> discharged, and a new obligation
        0b10, // ack, no req        -> discharged
        0b01, // req, no ack        -> obligation for cycle 4
        0b00, // neither            -> fails
    };
    const want_failures = [_]u32{ 0, 0, 0, 0, 1 };
    const want_matches = [_]u32{ 0, 1, 1, 0, 0 };

    for (trace, 0..) |cycle, index_of| {
        const atoms = [_]Word{cycle};
        const report = state.stepAtoms(&ir, &atoms);
        try testing.expectEqual(want_failures[index_of], report.failures);
        try testing.expectEqual(want_matches[index_of], report.matches);
    }
    // Nothing is left in flight, so the trace ending changes nothing.
    try testing.expectEqual(@as(u32, 0), state.finish().failures);
}

test "one alphabet across both automata" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // `req |=> req`: the same term on both sides of the implication, reached
    // through two separate `atom` nodes as a front-end would emit them. A
    // backend has to be asked for it once.
    const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0));
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        consequent,
    );
    const index = try addDirective(&ir, gpa, property, null);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);

    // Two roots: the shared term, and the constant false that says the
    // assertion never resets.
    try testing.expectEqual(@as(usize, 2), monitor.alphabet.roots.len);
    // And both automata name the shared term by the same guard.
    try testing.expectEqual(monitor.obligation.guard_of[0], monitor.trigger.?.guard_of[0]);
}

test "a reset abandons everything in flight" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const reset = try ir.atom(gpa, @enumFromInt(2));
    const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        consequent,
    );
    const index = try addDirective(&ir, gpa, property, reset);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);

    // The reset condition is a root of its own, kept even though no guard
    // mentions it.
    try testing.expect(monitor.reset < monitor.alphabet.roots.len);
    try testing.expectEqual(@as(usize, 3), monitor.alphabet.roots.len);
    // And `reset` indexes it after compaction, not before: the condition is
    // interned last but pinned first, so the two indices differ.
    try testing.expectEqual(reset, monitor.alphabet.roots[monitor.reset]);

    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    // Bit 0 `req`, bit 1 `ack`, bit 2 `reset`. The obligation raised in cycle 0
    // would fail in cycle 1 for want of `ack`; the reset abandons it instead,
    // and also discards the trigger, so the `req` in cycle 1 raises nothing.
    const trace = [_]u64{ 0b001, 0b100, 0b000 };
    const want_reset = [_]bool{ false, true, false };

    for (trace, 0..) |cycle, index_of| {
        const atoms = [_]Word{cycle};
        const report = state.stepAtoms(&ir, &atoms);
        try testing.expectEqual(want_reset[index_of], report.reset);
        try testing.expectEqual(@as(u32, 0), report.failures);
        try testing.expectEqual(@as(u32, 0), report.matches);
    }
    try testing.expectEqual(@as(u32, 0), state.finish().failures);
}

test "strength decides what a pending attempt does at the end" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        // `req |=> a ##1 b`, so an obligation raised in the last cycle is still
        // pending when the trace ends.
        const body = try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 1),
            1,
            1,
            try atomCycle(&ir, gpa, 1),
        );
        const property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            try ir.propSeq(gpa, strength, body),
        );
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);

        // `req` then `a`: the obligation is one cycle into a two-cycle match.
        for ([_]u64{ 0b01, 0b10 }) |cycle| {
            const atoms = [_]Word{cycle};
            const report = state.stepAtoms(&ir, &atoms);
            try testing.expectEqual(@as(u32, 0), report.failures);
        }
        const final = state.finish();
        try testing.expectEqual(
            @as(u32, if (strength == .prop_seq_strong) 1 else 0),
            final.failures,
        );
    }
}

test "an obligation without a bounded length is counted rather than ringed" {
    const gpa = testing.allocator;

    // `req |=> ##[1:$] ack`. However many attempts are pending, one more can
    // always arrive, so no fixed ring of state vectors tracks them — and the
    // determinised obligation with a counter per state is what does. The trigger
    // is untouched by the choice: it is existential, one vector for all of its
    // attempts, so it needs no bound and composes with either discipline. This is
    // the shape the counted mode exists for.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const body = try leadingDelay(&ir, gpa, 1, Ir.unbounded, try atomCycle(&ir, gpa, 1));
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        try ir.propSeq(gpa, .prop_seq_weak, body),
    );
    const index = try addDirective(&ir, gpa, property, null);

    var monitor = try assertion(gpa, &ir, index, .{}, null);
    defer monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, monitor.mode);
    try testing.expectEqual(@as(u32, 0), monitor.depth);
    try testing.expect(monitor.trigger != null);
    // No dead state, and that is the right answer rather than a missing one:
    // while the filler loop is live a later `ack` is always possible, so no
    // continuation is ruled out and the attempt can only ever pend. Which is
    // exactly why the LRM's unbounded eventually is the *strong* one — the
    // verdict is the end of the trace, not a cycle inside it.
    try testing.expect(!Nfa.bitAny(monitor.obligation.dead));

    // Goto repetition is unbounded for the same reason — the gaps between
    // occurrences are, however few occurrences are asked for — and is counted for
    // the same reason too.
    var marks_ir: Ir = .init();
    defer marks_ir.deinit(gpa);
    const marks = try marks_ir.seqRepeat(
        gpa,
        .seq_repeat_goto,
        try atomCycle(&marks_ir, gpa, 1),
        2,
        2,
    );
    const marked = try marks_ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&marks_ir, gpa, 0),
        try marks_ir.propSeq(gpa, .prop_seq_weak, marks),
    );
    var goto_monitor = try assertion(
        gpa,
        &marks_ir,
        try addDirective(&marks_ir, gpa, marked, null),
        .{},
        null,
    );
    defer goto_monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, goto_monitor.mode);

    // An `and` is unbounded as soon as *either* side is, where an intersection
    // survives one unbounded side: an intersection's two ends coincide, so the
    // bounded side bounds the pair, but an `and` runs on to the later end. The
    // asymmetry is what is pinned; that the unbounded one is now counted rather
    // than refused is the change.
    var waited: Ir = .init();
    defer waited.deinit(gpa);
    const open = try leadingDelay(&waited, gpa, 1, Ir.unbounded, try atomCycle(&waited, gpa, 1));
    const later = try waited.seqAnd(gpa, open, try atomCycle(&waited, gpa, 2));
    const waits = try waited.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&waited, gpa, 0),
        try waited.propSeq(gpa, .prop_seq_weak, later),
    );
    var and_monitor = try assertion(
        gpa,
        &waited,
        try addDirective(&waited, gpa, waits, null),
        .{},
        null,
    );
    defer and_monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, and_monitor.mode);

    // The contrast, so that asymmetry is pinned rather than assumed: a bound
    // means a ring, and a ring means a depth.
    const met = try waited.seqIntersect(gpa, open, try atomCycle(&waited, gpa, 2));
    const meets = try waited.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&waited, gpa, 0),
        try waited.propSeq(gpa, .prop_seq_weak, met),
    );
    var ringed = try assertion(gpa, &waited, try addDirective(&waited, gpa, meets, null), .{}, null);
    defer ringed.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.ring, ringed.mode);
    try testing.expectEqual(@as(u32, 1), ringed.depth);
}

test "a nested implication is rejected" {
    const gpa = testing.allocator;

    // `a |=> (b |=> c)`. The obligation would have to raise obligations of its
    // own, which is a different machine from a ring of state vectors.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const inner = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 1),
        try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
    );
    const outer = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        inner,
    );
    const index = try addDirective(&ir, gpa, outer, null);

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.UnsupportedConsequent,
        assertion(gpa, &ir, index, .{}, &diagnostic),
    );
    try testing.expectEqual(inner, diagnostic.node);
}

test "the depth cap reports the obligation that tripped it" {
    const gpa = testing.allocator;

    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const body = try leadingDelay(&ir, gpa, 1, 400, try atomCycle(&ir, gpa, 1));
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        try ir.propSeq(gpa, .prop_seq_weak, body),
    );
    const index = try addDirective(&ir, gpa, property, null);

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.DepthLimitExceeded,
        assertion(gpa, &ir, index, .{ .depth_limit = 64 }, &diagnostic),
    );
    try testing.expectEqual(body, diagnostic.node);
    // A leading delay of up to 400 puts the match's last cycle at 401 at worst.
    try testing.expectEqual(@as(u64, 401), diagnostic.size);
}

// -- The property layer above a sequence -------------------------------------
//
// Every operator here is checked against `propFate`, which reads the verdict
// definitions and knows nothing about the constructions below. The continuation
// enumeration in `bruteForceFate` is exponential in the obligation's bound, so
// the shapes stay small on purpose: a bound of 3 over 2 atoms is 64
// continuations per cycle per start, and that is the budget.

test "differential: property conjunction and disjunction" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_and, .prop_or }) |junction| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            // Two sides that resolve in *different* cycles and can each fail on
            // their own: `a` is decided in the attempt's first cycle and
            // `##1 b` in its second, so a conjunction has to wait for the later
            // one and a disjunction has to take the earlier.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const left = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const right = try ir.propSeq(gpa, strength, try leadingDelay(
                    &ir,
                    gpa,
                    1,
                    1,
                    try atomCycle(&ir, gpa, 1),
                ));
                const property = try ir.propJunction(gpa, junction, left, right);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
            // The same pair as an obligation under a trigger, and with a side
            // whose match length varies so an attempt can be pending with both
            // sides still alive.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const left = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 1));
                const right = try ir.propSeq(gpa, strength, try leadingDelay(
                    &ir,
                    gpa,
                    1,
                    2,
                    try atomCycle(&ir, gpa, 1),
                ));
                const pair = try ir.propJunction(gpa, junction, left, right);
                const property = try ir.propImplies(
                    gpa,
                    .prop_implies_nonoverlap,
                    try atomCycle(&ir, gpa, 0),
                    pair,
                );
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
        }
    }
}

test "differential: nexttime, bounded eventually and bounded always" {
    const gpa = testing.allocator;

    const Case = struct { tag: Ir.Node.Tag, lo: u32, hi: u32 };
    for ([_]Case{
        // `nexttime[0] p` is `p`: no filler at all, which is the one shape
        // where the shift has to add nothing rather than one cycle.
        .{ .tag = .prop_delay, .lo = 0, .hi = 0 },
        .{ .tag = .prop_delay, .lo = 1, .hi = 1 },
        .{ .tag = .prop_delay, .lo = 2, .hi = 2 },
        // `eventually [m:n]`: the range read disjunctively.
        .{ .tag = .prop_delay, .lo = 0, .hi = 2 },
        .{ .tag = .prop_delay, .lo = 1, .hi = 3 },
        // `always [m:n]`: the same range read conjunctively. At `lo == hi` the
        // two coincide, which is worth checking rather than assuming.
        .{ .tag = .prop_always, .lo = 0, .hi = 1 },
        .{ .tag = .prop_always, .lo = 1, .hi = 3 },
        .{ .tag = .prop_always, .lo = 2, .hi = 2 },
    }) |case| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            // A one-cycle body over one atom, so the operator is all there is to
            // see and the range can be wide without the bound growing.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const property = try ir.propRange(gpa, case.tag, case.lo, case.hi, body);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 1, 5, null);
            }
        }
    }

    // A body that takes a cycle of its own to resolve, under a trigger, so the
    // shift composes with an obligation rather than replacing it.
    for ([_]Ir.Node.Tag{ .prop_delay, .prop_always }) |tag| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const body = try ir.propSeq(gpa, strength, try leadingDelay(
                &ir,
                gpa,
                1,
                1,
                try atomCycle(&ir, gpa, 1),
            ));
            const shifted = try ir.propRange(gpa, tag, 0, 1, body);
            const property = try ir.propImplies(
                gpa,
                .prop_implies_overlap,
                try atomCycle(&ir, gpa, 0),
                shifted,
            );
            const index = try addDirective(&ir, gpa, property, null);
            try differentialMonitor(gpa, &ir, index, 2, 4, null);
        }
    }
}

test "differential: followed-by" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_followed_by_overlap, .prop_followed_by_nonoverlap }) |tag| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            // A one-cycle antecedent: the witness is unique, so the existential
            // is visible without being tangled up in which witness won.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 1));
                const property = try ir.propFollowedBy(
                    gpa,
                    tag,
                    try atomCycle(&ir, gpa, 0),
                    consequent,
                );
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
            // An antecedent with *several* possible ends, so the existential has
            // a choice and one failing witness must not settle it.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const consequent = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 1));
                const property = try ir.propFollowedBy(
                    gpa,
                    tag,
                    try leadingDelay(&ir, gpa, 0, 1, try atomCycle(&ir, gpa, 0)),
                    consequent,
                );
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
        }
    }
}

test "differential: an accept abort and a reject abort" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_accept_on, .prop_reject_on }) |tag| {
        for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
            // A two-cycle body, so there is a cycle in which the attempt is
            // pending and the condition can catch it mid-flight rather than only
            // ever agreeing with a verdict already reached.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.propSeq(gpa, strength, try leadingDelay(
                    &ir,
                    gpa,
                    1,
                    1,
                    try atomCycle(&ir, gpa, 0),
                ));
                const property = try ir.propAbort(gpa, tag, try ir.atom(gpa, @enumFromInt(1)), body);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
            // Below an implication. The abort is scoped to the obligation, so
            // the antecedent keeps matching and keeps raising obligations in the
            // very cycles the condition holds — which a monitor-wide reset would
            // not do.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.propSeq(gpa, strength, try leadingDelay(
                    &ir,
                    gpa,
                    1,
                    1,
                    try atomCycle(&ir, gpa, 0),
                ));
                const guarded = try ir.propAbort(gpa, tag, try ir.atom(gpa, @enumFromInt(1)), body);
                const property = try ir.propImplies(
                    gpa,
                    .prop_implies_nonoverlap,
                    try atomCycle(&ir, gpa, 0),
                    guarded,
                );
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
            // An abort under a shift: the filler cycles are not part of the
            // abort's subtree, so the condition holding there must do nothing.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const body = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const guarded = try ir.propAbort(gpa, tag, try ir.atom(gpa, @enumFromInt(1)), body);
                const property = try ir.propRange(gpa, .prop_delay, 2, 2, guarded);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
            // An abort under a disjunction, which is where a reject stops
            // resolving the whole attempt and only kills its own disjunct.
            {
                var ir: Ir = .init();
                defer ir.deinit(gpa);
                const guarded = try ir.propAbort(
                    gpa,
                    tag,
                    try ir.atom(gpa, @enumFromInt(1)),
                    try ir.propSeq(gpa, strength, try leadingDelay(
                        &ir,
                        gpa,
                        1,
                        1,
                        try atomCycle(&ir, gpa, 0),
                    )),
                );
                const other = try ir.propSeq(gpa, strength, try atomCycle(&ir, gpa, 0));
                const property = try ir.propJunction(gpa, .prop_or, guarded, other);
                const index = try addDirective(&ir, gpa, property, null);
                try differentialMonitor(gpa, &ir, index, 2, 4, null);
            }
        }
    }
}

/// Everything a monitor does over every trace of `length` cycles, as one flat
/// fingerprint: failures and matches per cycle, then what the end of the trace
/// makes of whatever is still pending.
///
/// Two spellings of the same property have equal fingerprints; two spellings
/// that differ anywhere at all do not. That makes this both the way an identity
/// is checked and the way a mutant is shown to be reachable by a test.
fn behaviour(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    atom_count: u5,
    length: u32,
    into: *std.ArrayListUnmanaged(u64),
) !void {
    var monitor = try assertion(gpa, ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |window| {
        state.clear();
        for (0..length) |cycle| {
            const atoms = [_]Word{window[cycle]};
            const report = state.stepAtoms(ir, &atoms);
            try into.append(gpa, report.failures);
            try into.append(gpa, report.matches);
        }
        const final = state.finish();
        try into.append(gpa, final.failures);
        try into.append(gpa, final.matches);
    }
}

fn sameBehaviour(
    gpa: Allocator,
    left: *const Ir,
    left_index: Ir.Assertion.Index,
    right: *const Ir,
    right_index: Ir.Assertion.Index,
    atom_count: u5,
    length: u32,
) !void {
    var a: std.ArrayListUnmanaged(u64) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayListUnmanaged(u64) = .empty;
    defer b.deinit(gpa);
    try behaviour(gpa, left, left_index, atom_count, length, &a);
    try behaviour(gpa, right, right_index, atom_count, length, &b);
    try testing.expectEqualSlices(u64, a.items, b.items);
}

test "followed-by fails where implication holds vacuously" {
    const gpa = testing.allocator;

    // The whole difference between the two operators, on the trace that shows
    // it: `a` never holds, so there is no antecedent match at all. An
    // implication is vacuously true and reports nothing; a followed-by asserted
    // a witness exists and there is none, so every attempt fails.
    //
    // `first_match` taught us that an operator can be a no-op in the position a
    // test naturally puts it in. This is the position where followed-by is not.
    const trace = [_]u64{ 0, 0, 0, 0 };

    var vacuous: Ir = .init();
    defer vacuous.deinit(gpa);
    {
        const consequent = try vacuous.propSeq(gpa, .prop_seq_weak, try atomCycle(&vacuous, gpa, 1));
        const property = try vacuous.propImplies(
            gpa,
            .prop_implies_overlap,
            try atomCycle(&vacuous, gpa, 0),
            consequent,
        );
        const index = try addDirective(&vacuous, gpa, property, null);
        try testing.expectEqual(@as(u32, 0), try failuresOver(gpa, &vacuous, index, &trace));
    }

    var witnessed: Ir = .init();
    defer witnessed.deinit(gpa);
    {
        const consequent = try witnessed.propSeq(gpa, .prop_seq_weak, try atomCycle(&witnessed, gpa, 1));
        const property = try witnessed.propFollowedBy(
            gpa,
            .prop_followed_by_overlap,
            try atomCycle(&witnessed, gpa, 0),
            consequent,
        );
        const index = try addDirective(&witnessed, gpa, property, null);
        // One attempt per cycle, each failing in the cycle it starts: the
        // antecedent is one cycle long, so a match is impossible from the start.
        try testing.expectEqual(@as(u32, 4), try failuresOver(gpa, &witnessed, index, &trace));
    }
}

test "an accept discharges where a reset abandons" {
    const gpa = testing.allocator;

    // The distinction §6 says the reset path does not draw, drawn. Both spellings
    // stop the obligation the moment `b` holds; they differ in what is left
    // behind. `b` at cycle 1, and `ack` never, so every attempt is pending when
    // `b` arrives and would otherwise have failed.
    const trace = [_]u64{ 0, 0b10, 0, 0 };

    // Abandoned: no verdict at all, from a *strong* property that would
    // otherwise fail every pending attempt at the end of the trace.
    var abandoned: Ir = .init();
    defer abandoned.deinit(gpa);
    const reset_failures = blk: {
        const body = try abandoned.propSeq(gpa, .prop_seq_strong, try leadingDelay(
            &abandoned,
            gpa,
            1,
            1,
            try atomCycle(&abandoned, gpa, 0),
        ));
        const index = try addDirective(
            &abandoned,
            gpa,
            body,
            try abandoned.atom(gpa, @enumFromInt(1)),
        );
        break :blk try failuresOver(gpa, &abandoned, index, &trace);
    };

    // Discharged: a pass, and a pass is what a `cover` counts.
    var discharged: Ir = .init();
    defer discharged.deinit(gpa);
    const accept_failures = blk: {
        const body = try discharged.propSeq(gpa, .prop_seq_strong, try leadingDelay(
            &discharged,
            gpa,
            1,
            1,
            try atomCycle(&discharged, gpa, 0),
        ));
        const property = try discharged.propAbort(
            gpa,
            .prop_accept_on,
            try discharged.atom(gpa, @enumFromInt(1)),
            body,
        );
        const index = try addDirective(&discharged, gpa, property, null);

        var monitor = try assertion(gpa, &discharged, index, .{}, null);
        defer monitor.deinit(gpa);
        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);
        state.clear();
        var matches: u64 = 0;
        var failures: u64 = 0;
        for (trace) |cycle| {
            const atoms = [_]Word{cycle};
            const report = state.stepAtoms(&discharged, &atoms);
            matches += report.matches;
            failures += report.failures;
        }
        const final = state.finish();

        // The accept turned pending attempts into passes, which a reset would
        // have left as nothing. This is the assertion that dies if `accept_on`
        // is lowered as a reset.
        try testing.expect(matches > 0);
        break :blk failures + final.failures;
    };

    // Both suppress the failures the obligation would otherwise have produced,
    // so counting failures alone cannot tell them apart — which is exactly why
    // the match count above is the load-bearing check.
    try testing.expectEqual(reset_failures, accept_failures);
}

test "a reject under a disjunction kills only its own disjunct" {
    const gpa = testing.allocator;

    // Where a reject stops propagating. `(reject_on(b) ##1 x) or (##2 y)`
    // against `reject_on(b) ((##1 x) or (##2 y))`: the same two operators nested
    // the other way round, and the trace that separates them is one where the
    // reject fires while the *other* disjunct is still alive and about to
    // succeed. `b` in cycle 0 kills the left disjunct there; `y` in cycle 2 then
    // discharges the right one.
    //
    // Under the disjunction the attempt survives to be discharged. Over it, the
    // reject resolves the whole attempt and the surviving disjunct never gets to
    // finish. Atom 0 is `x`, atom 1 is `b`, atom 2 is `y`.
    const trace = [_]u64{ 0b010, 0, 0b100, 0 };

    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, inside: bool) !Ir.Assertion.Index {
            const cond = try ir.atom(alloc, @enumFromInt(1));
            var left = try ir.propSeq(alloc, .prop_seq_weak, try leadingDelay(
                ir,
                alloc,
                1,
                1,
                try atomCycle(ir, alloc, 0),
            ));
            const right = try ir.propSeq(alloc, .prop_seq_weak, try leadingDelay(
                ir,
                alloc,
                2,
                2,
                try atomCycle(ir, alloc, 2),
            ));
            if (inside) left = try ir.propAbort(alloc, .prop_reject_on, cond, left);
            var property = try ir.propJunction(alloc, .prop_or, left, right);
            if (!inside) property = try ir.propAbort(alloc, .prop_reject_on, cond, property);
            return addDirective(ir, alloc, property, null);
        }
    }.go;

    var under: Ir = .init();
    defer under.deinit(gpa);
    const under_index = try build(&under, gpa, true);

    var over: Ir = .init();
    defer over.deinit(gpa);
    const over_index = try build(&over, gpa, false);

    {
        var monitor = try assertion(gpa, &under, under_index, .{}, null);
        defer monitor.deinit(gpa);
        // The disjunction demoted it: it clears its own states rather than
        // resolving the attempt, and its mask is the left disjunct alone.
        try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
        try testing.expectEqual(Monitor.Abort.Scope.branch, monitor.aborts[0].scope);
        try testing.expectEqual(Monitor.Abort.Effect.reject, monitor.aborts[0].effect);
        try testing.expectEqual(@as(u32, 2), @popCount(monitor.maskOf(0)[0]));
    }
    {
        var monitor = try assertion(gpa, &over, over_index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(Monitor.Abort.Scope.attempt, monitor.aborts[0].scope);
        // Over the disjunction the mask is the whole obligation: two states for
        // the left disjunct and three for the right.
        try testing.expectEqual(@as(u32, 5), @popCount(monitor.maskOf(0)[0]));
    }

    // The attempt starting in cycle 0 is discharged under the disjunction and
    // failed over it, so the totals differ by exactly that attempt.
    const under_failures = try failuresOver(gpa, &under, under_index, &trace);
    const over_failures = try failuresOver(gpa, &over, over_index, &trace);
    try testing.expectEqual(under_failures + 1, over_failures);
}

test "an abort covers its own subtree and not the cycles before it" {
    const gpa = testing.allocator;

    // `nexttime[2] (accept_on(b) a)` against `accept_on(b) (nexttime[2] a)`. The
    // same two operators, nested the other way: inside, the abort covers only
    // the cycle `a` is evaluated in, and outside it covers the two filler cycles
    // as well. `b` in cycle 0 and `a` never, so the outer spelling accepts
    // immediately and the inner one cannot.
    const trace = [_]u64{ 0b10, 0, 0, 0 };

    var inside: Ir = .init();
    defer inside.deinit(gpa);
    const inside_index = blk: {
        const body = try inside.propSeq(gpa, .prop_seq_weak, try atomCycle(&inside, gpa, 0));
        const guarded = try inside.propAbort(
            gpa,
            .prop_accept_on,
            try inside.atom(gpa, @enumFromInt(1)),
            body,
        );
        const property = try inside.propRange(gpa, .prop_delay, 2, 2, guarded);
        break :blk try addDirective(&inside, gpa, property, null);
    };

    var outside: Ir = .init();
    defer outside.deinit(gpa);
    const outside_index = blk: {
        const body = try outside.propSeq(gpa, .prop_seq_weak, try atomCycle(&outside, gpa, 0));
        const shifted = try outside.propRange(gpa, .prop_delay, 2, 2, body);
        const property = try outside.propAbort(
            gpa,
            .prop_accept_on,
            try outside.atom(gpa, @enumFromInt(1)),
            shifted,
        );
        break :blk try addDirective(&outside, gpa, property, null);
    };

    // The mask is what encodes the difference: one state of three inside, all
    // three outside.
    {
        var monitor = try assertion(gpa, &inside, inside_index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(u32, 3), monitor.obligation.state_count);
        try testing.expectEqual(@as(u32, 1), @popCount(monitor.maskOf(0)[0]));
    }
    {
        var monitor = try assertion(gpa, &outside, outside_index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(u32, 3), monitor.obligation.state_count);
        try testing.expectEqual(@as(u32, 3), @popCount(monitor.maskOf(0)[0]));
    }

    // Cycle 0 is a filler cycle for the attempt starting there, so the inner
    // spelling fails it and the outer one accepts it.
    try testing.expect(
        try failuresOver(gpa, &inside, inside_index, &trace) !=
            try failuresOver(gpa, &outside, outside_index, &trace),
    );
}

test "a shift over an implication moves the antecedent, not the obligation" {
    const gpa = testing.allocator;

    // `nexttime[n] (a |-> b)` obliges `b` for exactly the matches of `a` that
    // start `n` cycles into the attempt. A trigger is seeded every cycle, so
    // growing an `n`-cycle filler prefix in front of it selects exactly those —
    // which is a claim about the lowering, checked here against the spelling it
    // claims to equal. `always [m:n]` survives the same move because a
    // conjunction of universals is a universal over the union of their triggers;
    // `eventually [m:n]` does not, and is refused below.
    for ([_]u32{ 1, 2 }) |offset| {
        var shifted: Ir = .init();
        defer shifted.deinit(gpa);
        const shifted_index = blk: {
            const consequent = try shifted.propSeq(gpa, .prop_seq_weak, try atomCycle(&shifted, gpa, 1));
            const implication = try shifted.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                try atomCycle(&shifted, gpa, 0),
                consequent,
            );
            const property = try shifted.propRange(gpa, .prop_delay, offset, offset, implication);
            break :blk try addDirective(&shifted, gpa, property, null);
        };

        var spelled: Ir = .init();
        defer spelled.deinit(gpa);
        const spelled_index = blk: {
            const consequent = try spelled.propSeq(gpa, .prop_seq_weak, try atomCycle(&spelled, gpa, 1));
            const antecedent = try leadingDelay(
                &spelled,
                gpa,
                offset,
                offset,
                try atomCycle(&spelled, gpa, 0),
            );
            const property = try spelled.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                antecedent,
                consequent,
            );
            break :blk try addDirective(&spelled, gpa, property, null);
        };

        try sameBehaviour(gpa, &shifted, shifted_index, &spelled, spelled_index, 2, 5);
    }

    // The conjunctive range over an implication, against the union of its
    // triggers.
    var ranged: Ir = .init();
    defer ranged.deinit(gpa);
    const ranged_index = blk: {
        const consequent = try ranged.propSeq(gpa, .prop_seq_weak, try atomCycle(&ranged, gpa, 1));
        const implication = try ranged.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ranged, gpa, 0),
            consequent,
        );
        const property = try ranged.propRange(gpa, .prop_always, 1, 2, implication);
        break :blk try addDirective(&ranged, gpa, property, null);
    };

    var union_of: Ir = .init();
    defer union_of.deinit(gpa);
    const union_index = blk: {
        const consequent = try union_of.propSeq(gpa, .prop_seq_weak, try atomCycle(&union_of, gpa, 1));
        const antecedent = try leadingDelay(&union_of, gpa, 1, 2, try atomCycle(&union_of, gpa, 0));
        const property = try union_of.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            antecedent,
            consequent,
        );
        break :blk try addDirective(&union_of, gpa, property, null);
    };

    try sameBehaviour(gpa, &ranged, ranged_index, &union_of, union_index, 2, 5);
}

test "the conditional and case forms are a disjunction of followed-bys" {
    const gpa = testing.allocator;

    // `if (c) p else q` needs no operator of its own. Exactly one guard holds,
    // so `(c #-# p) or (!c #-# q)` is it: the disjunct whose guard fails has no
    // antecedent match, and a followed-by without a witness *fails* rather than
    // holding vacuously, so it contributes nothing and the taken branch decides
    // the verdict. An implication there would hold vacuously and the disjunction
    // would always pass.
    //
    // The `case` form is the same shape with one disjunct per item.
    var ir: Ir = .init();
    defer ir.deinit(gpa);

    const guard = try ir.atom(gpa, @enumFromInt(0));
    const taken = try ir.propFollowedBy(
        gpa,
        .prop_followed_by_overlap,
        try ir.seqBoolean(gpa, guard),
        try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1)),
    );
    const otherwise = try ir.propFollowedBy(
        gpa,
        .prop_followed_by_overlap,
        try ir.seqBoolean(gpa, try ir.addNode(gpa, .{
            .tag = .bnot,
            .data = .{ .lhs = @intFromEnum(guard) },
        })),
        try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
    );
    const property = try ir.propJunction(gpa, .prop_or, taken, otherwise);
    const index = try addDirective(&ir, gpa, property, null);

    // `c` selects which of the two atoms has to hold in the same cycle, and the
    // reference semantics agrees over every trace of all three.
    try differentialMonitor(gpa, &ir, index, 3, 3, null);
}

// -- The counted operators ---------------------------------------------------
//
// `always`, `s_eventually` and the four `until` forms, differentially against
// `propFate` like everything else in this file. The reference arms for them are
// written off LRM 16.12.2 and 16.12.10 and know nothing about the fragments the
// lowering builds, which is what makes disagreement mean something.

/// A one-cycle property over an atom.
fn atomProperty(ir: *Ir, gpa: Allocator, atom: u32) !Ir.Node.Index {
    return ir.propSeq(gpa, .prop_seq_weak, try atomCycle(ir, gpa, atom));
}

test "differential: the four until forms" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{
        .prop_until,
        .prop_until_with,
        .prop_s_until,
        .prop_s_until_with,
    }) |tag| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propUntil(
            gpa,
            tag,
            try atomProperty(&ir, gpa, 0),
            try atomProperty(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, property, null);
        // One cycle covers both leaves, which is all `bruteForceFate` needs; the
        // property has no bound of its own to borrow, which is the point.
        differentialMonitor(gpa, &ir, index, 2, 5, 1) catch |err| {
            std.debug.print("{s}\n", .{@tagName(tag)});
            return err;
        };
    }
}

test "differential: until over a multi-cycle right operand" {
    const gpa = testing.allocator;

    // `a until (b ##1 c)`. The right operand is existential — one attempt of it
    // per candidate cycle and only one has to succeed — so it may span several
    // cycles where the left one may not. This is the asymmetry
    // `Error.ObligationNotBounded` describes, checked rather than asserted.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const witness = try ir.seqDelay(
        gpa,
        try atomCycle(&ir, gpa, 1),
        1,
        1,
        try atomCycle(&ir, gpa, 2),
    );
    const property = try ir.propUntil(
        gpa,
        .prop_until,
        try atomProperty(&ir, gpa, 0),
        try ir.propSeq(gpa, .prop_seq_weak, witness),
    );
    const index = try addDirective(&ir, gpa, property, null);
    try differentialMonitor(gpa, &ir, index, 3, 4, 2);
}

test "differential: unbounded always and s_eventually" {
    const gpa = testing.allocator;

    for ([_]Ir.Node.Tag{ .prop_delay, .prop_always }) |tag| {
        for ([_]u32{ 0, 1 }) |lo| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);
            const inner = try ir.propSeq(
                gpa,
                if (tag == .prop_delay) .prop_seq_strong else .prop_seq_weak,
                try atomCycle(&ir, gpa, 0),
            );
            const property = try ir.propRange(gpa, tag, lo, Ir.unbounded, inner);
            const index = try addDirective(&ir, gpa, property, null);
            differentialMonitor(gpa, &ir, index, 1, 5, 1) catch |err| {
                std.debug.print("{s} from offset {d}\n", .{ @tagName(tag), lo });
                return err;
            };
        }
    }
}

test "differential: a trigger above an unbounded obligation" {
    const gpa = testing.allocator;

    // `req |=> s_eventually ack`, the shape the counted mode exists for. A trigger
    // is existential and needs no bound, so it composes with the counted
    // obligation; this is the check that the two do not interfere.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const inner = try ir.propSeq(gpa, .prop_seq_strong, try atomCycle(&ir, gpa, 1));
    const shifted = try ir.propRange(gpa, .prop_delay, 1, Ir.unbounded, inner);
    const property = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        shifted,
    );
    const index = try addDirective(&ir, gpa, property, null);
    try differentialMonitor(gpa, &ir, index, 2, 5, 1);
}

test "weak and strong until differ only at the end of the trace" {
    const gpa = testing.allocator;

    // The easiest thing to leave untested, and the only place the `s_` forms are
    // observable: a trace where `a` holds throughout and `b` never arrives. Every
    // attempt is still pending, so the weak forms report nothing and the strong
    // ones report one failure per attempt — five of them over five cycles, which
    // is also the multiplicity a single counter had to carry.
    const Case = struct { tag: Ir.Node.Tag, finish: u64 };
    for ([_]Case{
        .{ .tag = .prop_until, .finish = 0 },
        .{ .tag = .prop_until_with, .finish = 0 },
        .{ .tag = .prop_s_until, .finish = 5 },
        .{ .tag = .prop_s_until_with, .finish = 5 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propUntil(
            gpa,
            case.tag,
            try atomProperty(&ir, gpa, 0),
            try atomProperty(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(Monitor.Mode.counted, monitor.mode);
        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);

        // `a` high, `b` low, five cycles: nothing resolves.
        var cycle: u32 = 0;
        while (cycle < 5) : (cycle += 1) {
            const report = state.stepAtoms(&ir, &.{1});
            try testing.expectEqual(@as(u64, 0), report.failures);
            try testing.expectEqual(@as(u64, 0), report.matches);
        }
        const final = state.finish();
        testing.expectEqual(case.finish, final.failures) catch |err| {
            std.debug.print("{s}\n", .{@tagName(case.tag)});
            return err;
        };
    }
}

test "an until_with requires its left operand in the terminating cycle" {
    const gpa = testing.allocator;

    // Where the two inclusive forms part company from the plain ones, and the
    // reason the construction fuses rather than concatenates: with `a` low and `b`
    // high in the first cycle, `a until b` holds — nothing is asked of `a` at the
    // cycle `b` arrives — and `a until_with b` fails, because something is.
    const Case = struct { tag: Ir.Node.Tag, matches: u64, failures: u64 };
    for ([_]Case{
        .{ .tag = .prop_until, .matches = 1, .failures = 0 },
        .{ .tag = .prop_until_with, .matches = 0, .failures = 1 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propUntil(
            gpa,
            case.tag,
            try atomProperty(&ir, gpa, 0),
            try atomProperty(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);

        // Atom 1 only: `b` without `a`.
        const report = state.stepAtoms(&ir, &.{0b10});
        testing.expectEqual(case.matches, report.matches) catch |err| {
            std.debug.print("{s}: matches\n", .{@tagName(case.tag)});
            return err;
        };
        testing.expectEqual(case.failures, report.failures) catch |err| {
            std.debug.print("{s}: failures\n", .{@tagName(case.tag)});
            return err;
        };
    }
}

test "a trigger below an unbounded operator is refused as a mode mix" {
    const gpa = testing.allocator;

    // One tracking mode per monitor, and this shape wants both: the implication's
    // obligation is bounded and would be ringed, the operator above it is not and
    // would be counted. Refused, naming the operator that made them disagree.
    //
    // A *bounded* range over the same implication is accepted, because there the
    // shift lands on the antecedent instead — the contrast is what makes this a
    // statement about the modes rather than about implications.
    const Shape = enum { always, eventually, until };
    for ([_]Shape{ .always, .eventually, .until }) |shape| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const implication = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            try atomProperty(&ir, gpa, 1),
        );
        const property = switch (shape) {
            .always => try ir.propRange(gpa, .prop_always, 0, Ir.unbounded, implication),
            .eventually => try ir.propRange(gpa, .prop_delay, 0, Ir.unbounded, implication),
            .until => try ir.propUntil(
                gpa,
                .prop_until,
                implication,
                try atomProperty(&ir, gpa, 2),
            ),
        };
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        testing.expectError(
            error.ObligationModeMixed,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        ) catch |err| {
            std.debug.print("{s}\n", .{@tagName(shape)});
            return err;
        };
        try testing.expectEqual(property, diagnostic.node);
    }

    // The contrast: bounded, so the shift moves to the trigger and it lowers.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const implication = try ir.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&ir, gpa, 0),
        try atomProperty(&ir, gpa, 1),
    );
    const bounded = try ir.propRange(gpa, .prop_always, 0, 2, implication);
    var monitor = try assertion(
        gpa,
        &ir,
        try addDirective(&ir, gpa, bounded, null),
        .{},
        null,
    );
    defer monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.ring, monitor.mode);
}

test "a universal operator refuses a multi-cycle operand, naming it" {
    const gpa = testing.allocator;

    // `always (a ##1 b)` and `(a ##1 b) until c`. Both say their operand holds at
    // every cycle of a window, which is one attempt of it per cycle; a counted
    // attempt is a single state and cannot carry a growing set of them. Refused
    // naming the operand, not the operator, because the operand is what has to
    // change.
    const Shape = enum { always, until };
    for ([_]Shape{ .always, .until }) |shape| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const spanning = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            1,
            try atomCycle(&ir, gpa, 1),
        ));
        const property = switch (shape) {
            .always => try ir.propRange(gpa, .prop_always, 0, Ir.unbounded, spanning),
            .until => try ir.propUntil(
                gpa,
                .prop_until,
                spanning,
                try atomProperty(&ir, gpa, 2),
            ),
        };
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        testing.expectError(
            error.ObligationNotBounded,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        ) catch |err| {
            std.debug.print("{s}\n", .{@tagName(shape)});
            return err;
        };
        try testing.expectEqual(spanning, diagnostic.node);
    }

    // The contrast, twice over. A multi-cycle operand under the *existential*
    // unbounded shift is fine, and so is one as an until's right operand: both ask
    // for some attempt to succeed rather than all of them.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const spanning = try ir.propSeq(gpa, .prop_seq_strong, try ir.seqDelay(
        gpa,
        try atomCycle(&ir, gpa, 0),
        1,
        1,
        try atomCycle(&ir, gpa, 1),
    ));
    const eventually = try ir.propRange(gpa, .prop_delay, 0, Ir.unbounded, spanning);
    var shifted = try assertion(
        gpa,
        &ir,
        try addDirective(&ir, gpa, eventually, null),
        .{},
        null,
    );
    defer shifted.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, shifted.mode);

    const witnessed = try ir.propUntil(
        gpa,
        .prop_until,
        try atomProperty(&ir, gpa, 2),
        spanning,
    );
    var waited = try assertion(
        gpa,
        &ir,
        try addDirective(&ir, gpa, witnessed, null),
        .{},
        null,
    );
    defer waited.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, waited.mode);
}

test "a branch-scoped abort over a counted obligation is refused" {
    const gpa = testing.allocator;

    // `(reject_on(kill) a) or (s_eventually b)`. A reject under a disjunction only
    // kills its own disjunct, which is a mask subtracted from a state vector — and
    // a counted attempt has no vector, only one state standing for a whole subset.
    // An accept in the same position resolves the whole attempt and is carried
    // through, which is the contrast that makes this about the scope rather than
    // about aborts.
    const Case = struct { tag: Ir.Node.Tag, refused: bool };
    for ([_]Case{
        .{ .tag = .prop_reject_on, .refused = true },
        .{ .tag = .prop_accept_on, .refused = false },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const kill = try ir.atom(gpa, @enumFromInt(2));
        const guarded = try ir.propAbort(gpa, case.tag, kill, try atomProperty(&ir, gpa, 0));
        const inner = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
        const forever = try ir.propRange(gpa, .prop_delay, 0, Ir.unbounded, inner);
        const property = try ir.propJunction(gpa, .prop_or, guarded, forever);
        const index = try addDirective(&ir, gpa, property, null);

        if (case.refused) {
            var diagnostic: Diagnostic = undefined;
            try testing.expectError(
                error.UnsupportedAbortScope,
                assertion(gpa, &ir, index, .{}, &diagnostic),
            );
            try testing.expectEqual(guarded, diagnostic.node);
        } else {
            var monitor = try assertion(gpa, &ir, index, .{}, null);
            defer monitor.deinit(gpa);
            try testing.expectEqual(Monitor.Mode.counted, monitor.mode);
            try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
            try testing.expectEqual(Monitor.Abort.Scope.attempt, monitor.aborts[0].scope);
            // The mask came through the determinisation, so it names determinised
            // states and is not empty: an abort that armed for nothing would never
            // fire.
            try testing.expect(Nfa.bitAny(monitor.maskOf(0)));
        }
    }
}

test "counted state counts are the determinisation's" {
    const gpa = testing.allocator;

    // Pinned so that the blow-up an unbounded operator costs is a visible number
    // rather than a shrug. `until` is the expensive one: the run, the witness, and
    // the union of the two spellings all determinise together.
    const Case = struct { name: []const u8, tag: Ir.Node.Tag, states: u32 };
    for ([_]Case{
        .{ .name = "a until b", .tag = .prop_until, .states = 7 },
        .{ .name = "a until_with b", .tag = .prop_until_with, .states = 5 },
        .{ .name = "a s_until b", .tag = .prop_s_until, .states = 7 },
        .{ .name = "a s_until_with b", .tag = .prop_s_until_with, .states = 5 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propUntil(
            gpa,
            case.tag,
            try atomProperty(&ir, gpa, 0),
            try atomProperty(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, property, null);
        testing.expectEqual(case.states, try obligationStates(gpa, &ir, index)) catch |err| {
            std.debug.print("{s}\n", .{case.name});
            return err;
        };
    }

    // `always a` is the cheap end: one state for "a has held", plus the trap and
    // its cell sibling.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const inner = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0));
    const property = try ir.propRange(gpa, .prop_always, 0, Ir.unbounded, inner);
    const index = try addDirective(&ir, gpa, property, null);
    try testing.expectEqual(@as(u32, 3), try obligationStates(gpa, &ir, index));
}

/// The obligation's state count, with the caps lifted so a shape can be measured
/// rather than rejected.
fn obligationStates(gpa: Allocator, ir: *const Ir, index: Ir.Assertion.Index) !u32 {
    var monitor = try assertion(gpa, ir, index, .{
        .limit = 1 << 22,
        .depth_limit = 1 << 22,
    }, null);
    defer monitor.deinit(gpa);
    return monitor.obligation.state_count;
}

// -- Negation, implication and iff -------------------------------------------
//
// `propFate`'s arms for these are `fateNot` and `fateImplies`, which are the
// LRM 16.12.2 definitions as verdict algebra and know nothing about
// determinisation, mask swapping or `Plan.pending`. Disagreement therefore
// means the construction is wrong rather than that two spellings of it differ.

/// A weak or strong property over one atom, per `strength`.
fn atomPropertyOf(
    ir: *Ir,
    gpa: Allocator,
    atom: u32,
    strength: Monitor.Strength,
) !Ir.Node.Index {
    const tag: Ir.Node.Tag = switch (strength) {
        .weak => .prop_seq_weak,
        .strong => .prop_seq_strong,
    };
    return ir.propSeq(gpa, tag, try atomCycle(ir, gpa, atom));
}

test "differential: negation" {
    const gpa = testing.allocator;

    // Both strengths at the leaf, because `not` inverts strength and the
    // end-of-trace half of the verdict is the half an automaton cannot carry.
    // Every shape here leaves attempts pending when the trace runs out — the
    // last cycle always seeds one — so the disposition is exercised on every
    // trace rather than on a lucky one.
    const Shape = enum {
        one_cycle,
        two_cycle,
        range,
        disjunction,
        // `a` followed by something that cannot fail. Once `a` has held the
        // operand is *certain* to match, but it has not matched yet, so the
        // complement is pending rather than failed for one more cycle. This is
        // the shape `Plan.pending` exists for: co-reachability pruning would
        // report the failure here a cycle early.
        inevitable,
    };

    for ([_]Monitor.Strength{ .weak, .strong }) |strength| {
        for (std.enums.values(Shape)) |shape| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);

            const a = try atomCycle(&ir, gpa, 0);
            const b = try atomCycle(&ir, gpa, 1);
            const sequence_node = switch (shape) {
                .one_cycle => a,
                .two_cycle => try ir.seqDelay(gpa, a, 1, 1, b),
                .range => try ir.seqDelay(gpa, a, 1, 2, b),
                .disjunction => try ir.seqOr(gpa, a, try ir.seqDelay(gpa, a, 1, 1, b)),
                .inevitable => tautology: {
                    const atom = try ir.atom(gpa, @enumFromInt(1));
                    const either = try ir.bBinary(gpa, .bor, atom, try ir.bNot(gpa, atom));
                    break :tautology try ir.seqDelay(gpa, a, 1, 1, try ir.seqBoolean(gpa, either));
                },
            };
            const tag: Ir.Node.Tag = switch (strength) {
                .weak => .prop_seq_weak,
                .strong => .prop_seq_strong,
            };
            const property = try ir.propNot(gpa, try ir.propSeq(gpa, tag, sequence_node));
            const index = try addDirective(&ir, gpa, property, null);
            differentialMonitor(gpa, &ir, index, 2, 4, null) catch |err| {
                std.debug.print("{s} {s}\n", .{ @tagName(strength), @tagName(shape) });
                return err;
            };
        }
    }
}

test "differential: property implication" {
    const gpa = testing.allocator;

    // `implies` wants its operands' strengths to DIFFER, so both orders are
    // tried and both must lower. Anything else is `MixedStrength`, which the
    // refusal test below covers.
    for ([_]Monitor.Strength{ .weak, .strong }) |left_strength| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);

        const left = try atomPropertyOf(&ir, gpa, 0, left_strength);
        const right = try ir.propSeq(
            gpa,
            if (left_strength == .weak) .prop_seq_strong else .prop_seq_weak,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 1, try atomCycle(&ir, gpa, 0)),
        );
        const property = try ir.propJunction(gpa, .prop_implies, left, right);
        const index = try addDirective(&ir, gpa, property, null);
        differentialMonitor(gpa, &ir, index, 2, 4, null) catch |err| {
            std.debug.print("left {s}\n", .{@tagName(left_strength)});
            return err;
        };
    }
}

test "not not p behaves exactly like p" {
    const gpa = testing.allocator;

    // The involution, and the sharpest statement of `not pending == pending`
    // there is. A complement that resolved an attempt as soon as its operand's
    // outcome became inevitable would still round-trip *matches*, because the
    // second complement would move them back — but it would move the cycle, and
    // a fingerprint records the cycle.
    //
    // `inevitable` below is the trace family where that happens: after `a`, the
    // operand cannot fail, so a pruning complement reports one cycle early and
    // the double negation lands one cycle before `p` does.
    const Shape = enum { one_cycle, two_cycle, inevitable };
    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, shape: Shape) !Ir.Node.Index {
            const a = try atomCycle(ir, alloc, 0);
            return switch (shape) {
                .one_cycle => a,
                .two_cycle => try ir.seqDelay(alloc, a, 1, 1, try atomCycle(ir, alloc, 1)),
                .inevitable => tautology: {
                    const atom = try ir.atom(alloc, @enumFromInt(1));
                    const either = try ir.bBinary(alloc, .bor, atom, try ir.bNot(alloc, atom));
                    break :tautology try ir.seqDelay(alloc, a, 1, 1, try ir.seqBoolean(alloc, either));
                },
            };
        }
    }.go;

    for (std.enums.values(Shape)) |shape| {
        var plain: Ir = .init();
        defer plain.deinit(gpa);
        const bare = try plain.propSeq(gpa, .prop_seq_weak, try build(&plain, gpa, shape));
        const plain_index = try addDirective(&plain, gpa, bare, null);

        var doubled: Ir = .init();
        defer doubled.deinit(gpa);
        const inner = try doubled.propSeq(gpa, .prop_seq_weak, try build(&doubled, gpa, shape));
        const twice = try doubled.propNot(gpa, try doubled.propNot(gpa, inner));
        const doubled_index = try addDirective(&doubled, gpa, twice, null);

        // Strength round-trips too: two inversions is the identity.
        try testing.expectEqual(propStrength(&plain, bare), propStrength(&doubled, twice));
        sameBehaviour(gpa, &plain, plain_index, &doubled, doubled_index, 2, 4) catch |err| {
            std.debug.print("{s}\n", .{@tagName(shape)});
            return err;
        };
    }
}

test "implies is its desugaring, and is a tag only so a refusal can name it" {
    const gpa = testing.allocator;

    // The measurement the scoping document asked for before choosing between a
    // dedicated tag and a desugaring, run on the shape most likely to separate
    // them: an operand that costs a real determinisation.
    //
    // The answer is that there is nothing to choose. `lowerPropJunction`'s
    // `implies` arm *is* the desugaring — it complements the left operand and
    // hands both to the same `lowerOr` a front-end would have reached for — so
    // the two produce the same automaton, state for state. The numbers are
    // pinned below rather than merely compared, so that a future construction
    // that changed them has to say so.
    //
    // The tag earns its place on diagnostics alone: desugaring in the front-end
    // would put `prop_or` on `Diagnostic.node` and leave the front-end to work
    // out which of its own operators produced it. That is what
    // "a junction of operands with different strength is refused" checks.
    var direct: Ir = .init();
    defer direct.deinit(gpa);
    const direct_index = index: {
        const p = try direct.propSeq(
            gpa,
            .prop_seq_strong,
            try direct.seqDelay(gpa, try atomCycle(&direct, gpa, 0), 1, 2, try atomCycle(&direct, gpa, 1)),
        );
        const q = try atomPropertyOf(&direct, gpa, 1, .weak);
        break :index try addDirective(&direct, gpa, try direct.propJunction(gpa, .prop_implies, p, q), null);
    };

    var sugar: Ir = .init();
    defer sugar.deinit(gpa);
    const sugar_index = index: {
        const p = try sugar.propSeq(
            gpa,
            .prop_seq_strong,
            try sugar.seqDelay(gpa, try atomCycle(&sugar, gpa, 0), 1, 2, try atomCycle(&sugar, gpa, 1)),
        );
        const q = try atomPropertyOf(&sugar, gpa, 1, .weak);
        const desugared = try sugar.propJunction(gpa, .prop_or, try sugar.propNot(gpa, p), q);
        break :index try addDirective(&sugar, gpa, desugared, null);
    };

    const measured = try obligationStates(gpa, &direct, direct_index);
    try testing.expectEqual(@as(u32, 6), measured);
    try testing.expectEqual(measured, try obligationStates(gpa, &sugar, sugar_index));
    try sameBehaviour(gpa, &direct, direct_index, &sugar, sugar_index, 2, 4);
}

test "strength inverts under negation, and decides a pending complement" {
    const gpa = testing.allocator;

    // The end-of-trace half of `not`, on the one trace that isolates it: `a`
    // holds in the last cycle, so the attempt seeded there has `a ##1 b` still
    // pending — neither matched nor dead — and nothing but the strength decides
    // it.
    //
    // Over `weak(a ##1 b)` the operand holds at the end, so `not` fails there:
    // one failure the operand itself would never have reported. Over
    // `strong(a ##1 b)` the operand fails at the end and `not` discharges, so
    // the complement reports nothing at all — the exact opposite, from the same
    // automaton, with only `Monitor.Strength` different between them.
    const trace = [_]u64{ 0b00, 0b01 };

    const Case = struct { operand: Monitor.Strength, complement: Monitor.Strength, failures: u64 };
    for ([_]Case{
        .{ .operand = .weak, .complement = .strong, .failures = 1 },
        .{ .operand = .strong, .complement = .weak, .failures = 0 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const tag: Ir.Node.Tag = switch (case.operand) {
            .weak => .prop_seq_weak,
            .strong => .prop_seq_strong,
        };
        const inner = try ir.propSeq(gpa, tag, try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            1,
            try atomCycle(&ir, gpa, 1),
        ));
        const property = try ir.propNot(gpa, inner);
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(case.complement, monitor.strength);
        try testing.expectEqual(case.complement.atEnd(), monitor.strength.atEnd());
        try testing.expectEqual(case.failures, try failuresOver(gpa, &ir, index, &trace));
    }
}

test "differential: an abort above a complement" {
    const gpa = testing.allocator;

    // The positive control for the refusal below, and the thing that makes that
    // refusal attributable to the *nesting* rather than to the two operators
    // merely appearing together. An abort ABOVE a `not` is legal and lowers: its
    // scope is the whole complemented property, so its mask is every state of
    // the complement rather than a set of states inside one, and none of the
    // arming argument below applies.
    //
    // Worth having as a differential rather than as a "does it lower" check,
    // because the interesting cycle is the one where the abort's condition and
    // the complement's own verdict land together — the complement resolving is
    // the operand *failing*, which is the tie an accept has to win.
    for ([_]Ir.Node.Tag{ .prop_accept_on, .prop_reject_on }) |tag| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);

        const body = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            1,
            try atomCycle(&ir, gpa, 1),
        ));
        const negated = try ir.propNot(gpa, body);
        const guarded = try ir.propAbort(gpa, tag, try ir.atom(gpa, @enumFromInt(2)), negated);
        const index = try addDirective(&ir, gpa, guarded, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
        // Above the complement, so the effect is the one written: nothing
        // inverts it, because the abort is not inside the negated property.
        const effect: Monitor.Abort.Effect = switch (tag) {
            .prop_accept_on => .accept,
            else => .reject,
        };
        try testing.expectEqual(effect, monitor.aborts[0].effect);

        differentialMonitor(gpa, &ir, index, 3, 4, null) catch |err| {
            std.debug.print("{s}\n", .{@tagName(tag)});
            return err;
        };
    }
}

test "an abort under a complement is refused, naming the abort" {
    const gpa = testing.allocator;

    // Not because the mask cannot be moved through the subsets — it can — but
    // because the arming question stops being answerable, and stops on the one
    // cycle that decides an accept.
    //
    // `Monitor.State` arms an abort from the one-step *reachable* set of the
    // pre-advance vector rather than from the live one, deliberately, so that
    // it covers the cycle the guarded subtree's last thread dies in. That cycle
    // is what an accept is for: `accept_on(c) p` with `c` holding as `p` dies
    // discharges the attempt rather than failing it.
    //
    // A determinised successor is already filtered by the valuation, so that
    // set is not recoverable — and the complement then gives a `never` guard to
    // exactly the state that would have answered, the one whose subset meets
    // the operand's exits. Only the trap is left reachable, its subset is
    // empty, it meets no abort mask, and the abort reads as unarmed in the very
    // cycle it should fire. Lowering it would silently turn every such accept
    // into a failure.
    //
    // Every scope and effect, since none of them survives.
    for ([_]Ir.Node.Tag{
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
    }) |tag| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);

        const body = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            1,
            try atomCycle(&ir, gpa, 1),
        ));
        const guarded = try ir.propAbort(gpa, tag, try ir.atom(gpa, @enumFromInt(2)), body);
        const index = try addDirective(&ir, gpa, try ir.propNot(gpa, guarded), null);

        var diagnostic: Diagnostic = undefined;
        const result = assertion(gpa, &ir, index, .{}, &diagnostic);
        testing.expectError(error.UnsupportedAbortScope, result) catch |err| {
            std.debug.print("{s}\n", .{@tagName(tag)});
            return err;
        };
        // The abort that cannot be carried, not the `not` that could not carry
        // it — the same node an abort refusal names anywhere else.
        try testing.expectEqual(guarded, diagnostic.node);
    }
}

test "a complement is ring-tracked when its operand is" {
    const gpa = testing.allocator;

    // The mode question, pinned. Determinising is what `.counted` does, so the
    // natural guess is that a complement forces it; it does not. `.counted`
    // exists for obligations with no longest match, and it reads failure out of
    // `Nfa.dead` because its determinisation is *total*. A complement's is not
    // left total — the states where the operand matched are deleted — so the
    // vector empties there and a ring reads it like any other fragment.
    var bounded: Ir = .init();
    defer bounded.deinit(gpa);
    const inner = try bounded.propSeq(gpa, .prop_seq_weak, try bounded.seqDelay(
        gpa,
        try atomCycle(&bounded, gpa, 0),
        1,
        1,
        try atomCycle(&bounded, gpa, 1),
    ));
    const index = try addDirective(&bounded, gpa, try bounded.propNot(gpa, inner), null);
    var monitor = try assertion(gpa, &bounded, index, .{}, null);
    defer monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.ring, monitor.mode);
    // The operand's own longest match, borrowed unchanged: after two cycles
    // every thread of `a ##1 b` has matched or died, so the complement has
    // resolved too.
    try testing.expectEqual(@as(u32, 2), monitor.depth);

    // And an unbounded operand keeps its mode across the complement. `not
    // (always a)` is `s_eventually !a` and has no longest match either.
    var unbounded: Ir = .init();
    defer unbounded.deinit(gpa);
    const forever = try unbounded.propRange(
        gpa,
        .prop_always,
        0,
        Ir.unbounded,
        try atomProperty(&unbounded, gpa, 0),
    );
    const open = try addDirective(&unbounded, gpa, try unbounded.propNot(gpa, forever), null);
    var counted = try assertion(gpa, &unbounded, open, .{}, null);
    defer counted.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.counted, counted.mode);
    try testing.expectEqual(@as(u32, 0), counted.depth);
}

test "complement state counts are the determinisation's" {
    const gpa = testing.allocator;

    // The blow-up `not` costs, on record as numbers rather than as a warning.
    // Each row is a shape phase 3 measured, complemented — so the pair of
    // tables can be read against each other.
    //
    // A complement is `|subsets| * |cells|` like every other determinisation
    // here, less the states where the operand matched, which are deleted, plus
    // whatever the trap costs. What it is *not* is cheap: the last row is a
    // range whose subsets multiply out, and it is the one to watch if a front
    // end starts emitting `not` over wide sequences.
    const Case = struct { name: []const u8, operand: u32, states: u32 };
    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, which: usize) !Ir.Node.Index {
            const a = try atomCycle(ir, alloc, 0);
            const b = try atomCycle(ir, alloc, 1);
            return switch (which) {
                0 => a,
                1 => try ir.seqDelay(alloc, a, 1, 1, b),
                2 => try ir.seqOr(alloc, a, b),
                3 => try ir.seqDelay(alloc, a, 1, 3, b),
                else => unreachable,
            };
        }
    }.go;

    for ([_]Case{
        // One cell for `a` and one for its complement, so the operand's single
        // state becomes two subsets; the accepting one is deleted and the trap
        // and its cell sibling are what is left, both exits of the complement.
        .{ .name = "not a", .operand = 1, .states = 2 },
        .{ .name = "not (a ##1 b)", .operand = 2, .states = 4 },
        .{ .name = "not (a or b)", .operand = 2, .states = 2 },
        // The row to watch. A range multiplies the subsets out, and this is the
        // narrowest range there is; `Options.limit` is what stands between a
        // front-end and the rest of that curve.
        .{ .name = "not (a ##[1:3] b)", .operand = 4, .states = 6 },
    }, 0..) |case, which| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try ir.propSeq(gpa, .prop_seq_weak, try build(&ir, gpa, which));
        const index = try addDirective(&ir, gpa, try ir.propNot(gpa, inner), null);

        // The operand alone, so each row records what the complement cost on
        // top of it rather than an absolute number with nothing to read against.
        var plain: Ir = .init();
        defer plain.deinit(gpa);
        const operand = try plain.propSeq(gpa, .prop_seq_weak, try build(&plain, gpa, which));
        const plain_index = try addDirective(&plain, gpa, operand, null);

        testing.expectEqual(case.operand, try obligationStates(gpa, &plain, plain_index)) catch |err| {
            std.debug.print("{s}, operand\n", .{case.name});
            return err;
        };
        testing.expectEqual(case.states, try obligationStates(gpa, &ir, index)) catch |err| {
            std.debug.print("{s}\n", .{case.name});
            return err;
        };
    }
}

test "the shapes a complement cannot carry are refused, naming the operator" {
    const gpa = testing.allocator;

    const Case = enum {
        /// `not (s |-> p)`: the operand owns a trigger, which is a union of
        /// attempts and has thrown away what complementing needs.
        over_a_trigger,
        /// A product renumbers its operands, so the pending mask cannot come
        /// through — the same obstruction that refuses an abort mask there.
        under_a_conjunction,
        /// A fusion, likewise.
        under_a_followed_by,
        /// `always` rewires every state of its operand.
        under_an_unbounded_always,
    };

    for (std.enums.values(Case)) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);

        const complemented = try ir.propNot(gpa, try atomProperty(&ir, gpa, 0));
        // The node a diagnostic should name: the `not` itself when the operand
        // is what it cannot complement, and the operator above it when that
        // operator is what cannot carry the result.
        var blame = complemented;
        const property = switch (case) {
            .over_a_trigger => blamed: {
                const implication = try ir.propImplies(
                    gpa,
                    .prop_implies_nonoverlap,
                    try atomCycle(&ir, gpa, 1),
                    try atomProperty(&ir, gpa, 0),
                );
                const negated = try ir.propNot(gpa, implication);
                blame = negated;
                break :blamed negated;
            },
            .under_a_conjunction => blamed: {
                const pair = try ir.propJunction(
                    gpa,
                    .prop_and,
                    complemented,
                    // Strength has to agree or `MixedStrength` fires first and
                    // this stops testing what it is named after: `not` of a
                    // weak operand is strong, so the sibling is strong too.
                    try atomPropertyOf(&ir, gpa, 1, .strong),
                );
                blame = pair;
                break :blamed pair;
            },
            .under_a_followed_by => blamed: {
                const witnessed = try ir.propFollowedBy(
                    gpa,
                    .prop_followed_by_overlap,
                    try atomCycle(&ir, gpa, 1),
                    complemented,
                );
                blame = witnessed;
                break :blamed witnessed;
            },
            .under_an_unbounded_always => blamed: {
                const forever = try ir.propRange(gpa, .prop_always, 0, Ir.unbounded, complemented);
                blame = forever;
                break :blamed forever;
            },
        };

        const index = try addDirective(&ir, gpa, property, null);
        var diagnostic: Diagnostic = undefined;
        const result = assertion(gpa, &ir, index, .{}, &diagnostic);
        testing.expectError(error.UnsupportedComplement, result) catch |err| {
            std.debug.print("{s}\n", .{@tagName(case)});
            return err;
        };
        testing.expectEqual(blame, diagnostic.node) catch |err| {
            std.debug.print("{s}\n", .{@tagName(case)});
            return err;
        };
    }
}

test "implies wants operands of differing strength, and iff can never have them" {
    const gpa = testing.allocator;

    // The consequence of `not` inverting strength, pinned in both directions so
    // that a change to either half is a failing test rather than a surprise.
    //
    // It is not a conservative rule. With `p` and `q` both weak, a trace where
    // `q` has definitively failed while `p` is still pending ends with `p`
    // holding — so `q` was required, and the implication fails; a trace where
    // both are pending ends with it holding. One `Monitor.Strength` cannot
    // carry both, and refusing is the mechanism being honest.
    const Case = struct {
        tag: Ir.Node.Tag,
        left: Monitor.Strength,
        right: Monitor.Strength,
        ok: bool,
    };
    for ([_]Case{
        .{ .tag = .prop_implies, .left = .weak, .right = .strong, .ok = true },
        .{ .tag = .prop_implies, .left = .strong, .right = .weak, .ok = true },
        .{ .tag = .prop_implies, .left = .weak, .right = .weak, .ok = false },
        .{ .tag = .prop_implies, .left = .strong, .right = .strong, .ok = false },
        // Every spelling of iff, including the two an implication accepts.
        .{ .tag = .prop_iff, .left = .weak, .right = .strong, .ok = false },
        .{ .tag = .prop_iff, .left = .strong, .right = .weak, .ok = false },
        .{ .tag = .prop_iff, .left = .weak, .right = .weak, .ok = false },
        .{ .tag = .prop_iff, .left = .strong, .right = .strong, .ok = false },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propJunction(
            gpa,
            case.tag,
            try atomPropertyOf(&ir, gpa, 0, case.left),
            try atomPropertyOf(&ir, gpa, 1, case.right),
        );
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        const result = assertion(gpa, &ir, index, .{}, &diagnostic);
        if (case.ok) {
            var monitor = try result;
            defer monitor.deinit(gpa);
            // `not p or q`, and a disjunction carries its operands' shared
            // strength — which is `q`'s.
            try testing.expectEqual(case.right, monitor.strength);
            continue;
        }
        testing.expectError(error.MixedStrength, result) catch |err| {
            std.debug.print("{s} {s}/{s}\n", .{
                @tagName(case.tag),
                @tagName(case.left),
                @tagName(case.right),
            });
            return err;
        };
        // The operator as written, not the disjunction it reduces onto.
        try testing.expectEqual(property, diagnostic.node);
    }
}

test "a complement that blows up names the not" {
    const gpa = testing.allocator;

    // Determinisation is exponential and the cap is the existing one, reported
    // through `Builder.fail` like every other limit — no second ceiling, and
    // `Diagnostic.node` names the operator that asked for the subsets.
    var ir: Ir = .init();
    defer ir.deinit(gpa);

    // `a ##[1:3] b` is four states and its complement is six, so a cap of four
    // admits the sequence and refuses the subsets — which is the only way to
    // see that the complement is what reported, rather than the sequence
    // lowering tripping first and naming one of its own nodes.
    const pattern = try ir.seqDelay(
        gpa,
        try atomCycle(&ir, gpa, 0),
        1,
        3,
        try atomCycle(&ir, gpa, 1),
    );
    const negated = try ir.propNot(gpa, try ir.propSeq(gpa, .prop_seq_weak, pattern));
    const index = try addDirective(&ir, gpa, negated, null);

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.StateLimitExceeded,
        assertion(gpa, &ir, index, .{ .limit = 4 }, &diagnostic),
    );
    try testing.expectEqual(negated, diagnostic.node);
    try testing.expect(diagnostic.size > 4);
}

test "a bounded always costs the copies that overlap, not the width of the range" {
    const gpa = testing.allocator;

    // The measurement §4 and §10 of the scoping document asked for. The worry was
    // that `always [1:100] p` is a hundred-way conjunction and so the case that
    // justifies a dedicated construction over a desugaring. It is not: the cost
    // is *linear* in the width of the range either way.
    //
    // Why: a conjunct is `p` shifted to one offset, and two conjuncts only have a
    // reachable product where their copies of `p` are live at the same time. That
    // window is |p| cycles wide however wide the range is, so the product prunes
    // to a band rather than a hypercube and the slope is set by `p` alone.
    //
    // The consequence for the library is that there is nothing to win here. This
    // lowering *is* the desugaring — a shift per offset, folded with the same
    // `lowerAnd` a front-end would have reached for — and it produces exactly the
    // state counts the sequence-layer spelling produces. The operator earns its
    // place by keeping the semantics and the refusals in one place, not by
    // beating the desugaring on size.
    const Shape = enum { one_cycle, two_cycle, short_range, wide_range };
    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, shape: Shape) !Ir.Node.Index {
            const a = try atomCycle(ir, alloc, 0);
            const b = try atomCycle(ir, alloc, 1);
            return switch (shape) {
                .one_cycle => a,
                .two_cycle => try ir.seqDelay(alloc, a, 1, 1, b),
                .short_range => try ir.seqDelay(alloc, a, 1, 2, b),
                .wide_range => try ir.seqDelay(alloc, a, 1, 4, b),
            };
        }
    }.go;

    const Case = struct {
        shape: Shape,
        /// States `p` alone lowers to.
        body: u32,
        /// `always [1:100] p`, and `eventually [1:100] p` for comparison.
        conjunctive: u32,
        disjunctive: u32,
    };
    for ([_]Case{
        // A deterministic body has one copy live at a time, so the conjunction
        // is a chain and costs exactly what the disjunction does.
        .{ .shape = .one_cycle, .body = 1, .conjunctive = 101, .disjunctive = 101 },
        .{ .shape = .two_cycle, .body = 2, .conjunctive = 102, .disjunctive = 102 },
        // A body whose match length varies has several copies live at once, and
        // *that* is what the product pays for: still linear in the range, with a
        // slope of a few states per offset.
        .{ .shape = .short_range, .body = 3, .conjunctive = 400, .disjunctive = 103 },
        .{ .shape = .wide_range, .body = 5, .conjunctive = 3535, .disjunctive = 105 },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try ir.propSeq(gpa, .prop_seq_weak, try build(&ir, gpa, case.shape));

        const alone = try ir.propRange(gpa, .prop_delay, 0, 0, inner);
        const conjunctive = try ir.propRange(gpa, .prop_always, 1, 100, inner);
        const disjunctive = try ir.propRange(gpa, .prop_delay, 1, 100, inner);

        try testing.expectEqual(case.body, try obligationStates(
            gpa,
            &ir,
            try addDirective(&ir, gpa, alone, null),
        ));
        try testing.expectEqual(case.conjunctive, try obligationStates(
            gpa,
            &ir,
            try addDirective(&ir, gpa, conjunctive, null),
        ));
        try testing.expectEqual(case.disjunctive, try obligationStates(
            gpa,
            &ir,
            try addDirective(&ir, gpa, disjunctive, null),
        ));
    }

    // Linear, so the default cap is a ceiling on `n · slope` rather than
    // something a range of 100 trips on its own. The widest body above is 36
    // states per offset, which reaches 4096 at around n = 113.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try ir.propSeq(gpa, .prop_seq_weak, try build(&ir, gpa, .wide_range));
        const property = try ir.propRange(gpa, .prop_always, 1, 200, inner);
        const index = try addDirective(&ir, gpa, property, null);
        try testing.expectError(
            error.StateLimitExceeded,
            assertion(gpa, &ir, index, .{ .depth_limit = 1 << 22 }, null),
        );
    }
}

test "an unbounded always or eventually is counted, not silently bounded" {
    const gpa = testing.allocator;

    // `always p` and `s_eventually p` are an open range, and an open range leaves
    // the obligation without a longest match. That used to be a refusal; it now
    // selects the counted discipline, and what has to be pinned is that the range
    // is neither clamped to something finite nor tracked by a ring that cannot
    // hold it. `depth` being zero is the load-bearing half: a ring of any size
    // here would be a wrong answer waiting for a long enough trace.
    for ([_]Ir.Node.Tag{ .prop_delay, .prop_always }) |tag| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const inner = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0));
        const property = try ir.propRange(gpa, tag, 0, Ir.unbounded, inner);
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(Monitor.Mode.counted, monitor.mode);
        try testing.expectEqual(@as(u32, 0), monitor.depth);
        // The two are exact duals, and the pair of masks is where that shows.
        //
        // `s_eventually p` can never fail inside the trace: while the filler loop
        // is live a later cycle could still satisfy `p`, so nothing is ruled out
        // and there is no dead state to reach. It can complete a match, in the
        // cycle `p` holds.
        //
        // `always p` is the other way round entirely. It can never complete a
        // match — "for ever" has no finite witness — and it fails the moment `p`
        // is violated, which is the dead state. An empty `last` here is the
        // honest automaton for a safety property, not a degenerate one.
        const eventually = tag == .prop_delay;
        try testing.expectEqual(eventually, Nfa.bitAny(monitor.obligation.last));
        try testing.expectEqual(!eventually, Nfa.bitAny(monitor.obligation.dead));
    }
}

test "a property operator over an operand with a trigger of its own is refused" {
    const gpa = testing.allocator;

    // One monitor, one trigger. A conjunction of two implications wants two, and
    // a disjunctive range over one implication wants a disjunction of vacuities
    // that no single trigger expresses. Both are legal SVA and both are refused
    // here rather than mis-lowered.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
        const implication = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            consequent,
        );
        const property = try ir.propJunction(gpa, .prop_and, implication, implication);
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        try testing.expectError(
            error.UnsupportedConsequent,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        );
        try testing.expectEqual(implication, diagnostic.node);
    }
    // `eventually [1:2] (a |=> b)`. `always [1:2]` of the same is *not* refused —
    // see the identity test above — because a conjunction of universals is a
    // universal over the union of their triggers and a disjunction of them is
    // not.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
        const implication = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            consequent,
        );
        const property = try ir.propRange(gpa, .prop_delay, 1, 2, implication);
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        try testing.expectError(
            error.UnsupportedConsequent,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        );
        try testing.expectEqual(property, diagnostic.node);
    }
}

test "an abort above an implication covers the trigger" {
    const gpa = testing.allocator;

    // The shape essentially every real use writes, and the one a pushed-down
    // abort is not equivalent to. All four forms lower, and what comes out is
    // the ordinary obligation-masked abort with one flag set: covering the
    // trigger costs no state, no depth, and no second abort entry.
    const Case = struct { tag: Ir.Node.Tag, effect: Monitor.Abort.Effect, sampled: bool };
    for ([_]Case{
        .{ .tag = .prop_accept_on, .effect = .accept, .sampled = false },
        .{ .tag = .prop_reject_on, .effect = .reject, .sampled = false },
        .{ .tag = .prop_sync_accept_on, .effect = .accept, .sampled = true },
        .{ .tag = .prop_sync_reject_on, .effect = .reject, .sampled = true },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const implication = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1)),
        );
        const property = try ir.propAbort(
            gpa,
            case.tag,
            try ir.atom(gpa, @enumFromInt(2)),
            implication,
        );
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);

        try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
        try testing.expectEqual(case.effect, monitor.aborts[0].effect);
        try testing.expectEqual(case.sampled, monitor.aborts[0].sampled);
        try testing.expectEqual(Monitor.Abort.Scope.attempt, monitor.aborts[0].scope);
        try testing.expect(monitor.aborts[0].covers_trigger);
        // The mask still names obligation states and only those. Reaching the
        // trigger is the flag's job — there is no trigger mask, because the
        // abort covers every state of the one trigger there is.
        try testing.expectEqual(@as(u32, 1), @popCount(monitor.maskOf(0)[0]));
        try testing.expect(monitor.trigger != null);
    }

    // Against the bare implication, to pin the cost at zero rather than assert
    // three numbers and hope they are the same three. An abort only ever
    // resolves an attempt sooner than it would have resolved itself, so it
    // cannot lengthen the obligation, and `depth` is that longest match.
    var bare: Ir = .init();
    defer bare.deinit(gpa);
    const bare_index = try addDirective(&bare, gpa, try bare.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&bare, gpa, 0),
        try bare.propSeq(gpa, .prop_seq_weak, try atomCycle(&bare, gpa, 1)),
    ), null);
    var without = try assertion(gpa, &bare, bare_index, .{}, null);
    defer without.deinit(gpa);

    var guarded: Ir = .init();
    defer guarded.deinit(gpa);
    const guarded_index = try addDirective(&guarded, gpa, try guarded.propAbort(
        gpa,
        .prop_accept_on,
        try guarded.atom(gpa, @enumFromInt(2)),
        try guarded.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&guarded, gpa, 0),
            try guarded.propSeq(gpa, .prop_seq_weak, try atomCycle(&guarded, gpa, 1)),
        ),
    ), null);
    var with = try assertion(gpa, &guarded, guarded_index, .{}, null);
    defer with.deinit(gpa);

    try testing.expectEqual(without.obligation.state_count, with.obligation.state_count);
    try testing.expectEqual(without.trigger.?.state_count, with.trigger.?.state_count);
    try testing.expectEqual(without.depth, with.depth);
}

test "an abort above a multi-cycle antecedent counts the attempts it covers" {
    const gpa = testing.allocator;

    // What used to be refused. Covering a trigger works because the abort's
    // scope is the whole trigger, and what stopped a multi-cycle antecedent was
    // that an attempt caught halfway owes a verdict of its own with nothing to
    // count it. `trigger_window` counts it, so all four forms lower — and the
    // window is what says the count is available, so it is what to assert.
    //
    // `a ##1 b` needs no ids for its verdicts: every match spans two cycles, so
    // two of them cannot end together. It needs them for exactly this abort, and
    // two is one id per cycle a match can be in flight.
    for ([_]Ir.Node.Tag{
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
    }) |tag| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const property = try ir.propAbort(
            gpa,
            tag,
            try ir.atom(gpa, @enumFromInt(3)),
            try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 1),
                ),
                try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
            ),
        );
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expect(monitor.aborts[0].covers_trigger);
        try testing.expectEqual(@as(u32, 2), monitor.trigger_window);
    }

    // Without the abort the same antecedent tracks nothing at all: its matches
    // cannot end together, so a bit answers and the whole mechanism folds away.
    // This is the half of the sizing rule an abort is the only reason to widen.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const index = try addDirective(&ir, gpa, try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try ir.seqDelay(
                gpa,
                try atomCycle(&ir, gpa, 0),
                1,
                1,
                try atomCycle(&ir, gpa, 1),
            ),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
        ), null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(u32, 0), monitor.trigger_window);
    }

    // Multi-*state* is not multi-cycle, and it is the cycles that matter. `a or
    // b` matches in one cycle however many ways it can do it, so no partial
    // match survives a cycle and no id is needed even under an abort.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const index = try addDirective(&ir, gpa, try ir.propAbort(
            gpa,
            .prop_reject_on,
            try ir.atom(gpa, @enumFromInt(2)),
            try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                try ir.seqOr(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    try atomCycle(&ir, gpa, 1),
                ),
                try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1)),
            ),
        ), null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expect(monitor.aborts[0].covers_trigger);
        try testing.expectEqual(@as(u32, 2), monitor.trigger.?.state_count);
        try testing.expectEqual(@as(u32, 0), monitor.trigger_window);
    }
}

test "an unbounded antecedent is refused" {
    const gpa = testing.allocator;

    // An obligation is owed per antecedent match, so a cycle several matches end
    // in owes several verdicts, and telling those attempts apart is what bounds
    // the ids. `a ##[1:$] b` has unboundedly many attempts matching at once —
    // `merge` measures eight resolving in one cycle over eight cycles — so there
    // is no window that counts them and no bit that does either. Refused, naming
    // the antecedent, rather than answered with the number of matching cycles.
    //
    // Both spellings of unbounded reach, and both tracking modes: the
    // obligation's discipline has no bearing on the antecedent, so a counted
    // consequent refuses for the same reason a ring one does.
    const Shape = enum { delay, nonconsec };
    for ([_]bool{ false, true }) |counted| {
        for ([_]Shape{ .delay, .nonconsec }) |shape| {
            var ir: Ir = .init();
            defer ir.deinit(gpa);

            const antecedent = switch (shape) {
                .delay => try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 0),
                    1,
                    Ir.unbounded,
                    try atomCycle(&ir, gpa, 1),
                ),
                .nonconsec => try ir.seqRepeat(
                    gpa,
                    .seq_repeat_nonconsec,
                    try atomCycle(&ir, gpa, 0),
                    2,
                    2,
                ),
            };
            const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2));
            const property = try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                antecedent,
                if (counted)
                    try ir.propRange(gpa, .prop_delay, 0, Ir.unbounded, consequent)
                else
                    consequent,
            );
            const index = try addDirective(&ir, gpa, property, null);

            var diagnostic: Diagnostic = undefined;
            try testing.expectError(
                error.AntecedentNotBounded,
                assertion(gpa, &ir, index, .{}, &diagnostic),
            );
            try testing.expectEqual(antecedent, diagnostic.node);
        }
    }

    // And an abort above one is refused by the same rule rather than by the
    // abort's own, which is now only about how wide the window has to be.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const antecedent = try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            Ir.unbounded,
            try atomCycle(&ir, gpa, 1),
        );
        const index = try addDirective(&ir, gpa, try ir.propAbort(
            gpa,
            .prop_reject_on,
            try ir.atom(gpa, @enumFromInt(3)),
            try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                antecedent,
                try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
            ),
        ), null);

        var diagnostic: Diagnostic = undefined;
        try testing.expectError(
            error.AntecedentNotBounded,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        );
        try testing.expectEqual(antecedent, diagnostic.node);
    }

    // A bounded antecedent of the same shape lowers, and reports a window as
    // wide as its longest match: `a ##[1:3] b` has three attempts able to end in
    // one cycle, which is the bug this whole mechanism exists to report.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const index = try addDirective(&ir, gpa, try ir.propImplies(
            gpa,
            .prop_implies_overlap,
            try ir.seqDelay(
                gpa,
                try atomCycle(&ir, gpa, 0),
                1,
                3,
                try atomCycle(&ir, gpa, 1),
            ),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
        ), null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(u32, 4), monitor.trigger_window);
    }
}

test "an abort above an implication resolves the antecedent match instead of seeding" {
    const gpa = testing.allocator;

    // `tests/sva/diff_abort.sv`'s stimulus, which VCS X-2025.06 and QuestaSim
    // 2024.3_2 were both measured against. Atom 0 is `trig`, 1 is `e`, 2 is the
    // abort condition; cycle N is that fixture's t = 5 + 10N.
    //
    //   cyc   trig  e  cond
    //    0     0    1   0
    //    1     1    1   0   seeds an obligation on `e` at cycle 2
    //    2     0    0   0   <- it fails, with no abort anywhere near
    //    3     1    1   0   seeds again
    //    4     0    0   0   <- it fails too
    //    5     0    1   0
    //    6     0    1   0
    //    7     1    1   1   <- a fresh real obligation meets the condition
    //    8     0    1   0   `e` is high, so cycle 7's attempt would have PASSED
    //    9     1    1   0   seeds an obligation the trace ends before starting
    //
    // Cycle 7 is the whole point. The condition is high in the cycle the
    // antecedent matches, which for `|=>` is one cycle before anything would be
    // seeded, and the fixture's oracle golden puts a report there for both
    // reject forms and nothing at cycle 8. So the match has to be resolved
    // rather than seeded: seeding it and aborting the attempt in cycle 8 would
    // read the condition low and let it pass instead.
    const trace = [_]u64{
        0b010, 0b011, 0b000, 0b011, 0b000,
        0b010, 0b010, 0b111, 0b010, 0b011,
    };

    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, tag: Ir.Node.Tag, inside: bool) !Ir.Assertion.Index {
            const cond = try ir.atom(alloc, @enumFromInt(2));
            var consequent = try ir.propSeq(alloc, .prop_seq_weak, try atomCycle(ir, alloc, 1));
            if (inside) consequent = try ir.propAbort(alloc, tag, cond, consequent);
            var property = try ir.propImplies(
                alloc,
                .prop_implies_nonoverlap,
                try atomCycle(ir, alloc, 0),
                consequent,
            );
            if (!inside) property = try ir.propAbort(alloc, tag, cond, property);
            return addDirective(ir, alloc, property, null);
        }
    }.go;

    const Case = struct {
        tag: Ir.Node.Tag,
        inside: bool,
        failures: [10]u32,
        matches: [10]u32,
    };
    for ([_]Case{
        // A reject above the implication: the fixture's `rr` and `sr` column,
        // reporting at cycles 2, 4 and 7 and nothing at 8.
        .{
            .tag = .prop_reject_on,
            .inside = false,
            .failures = .{ 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 },
            .matches = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
        // An accept in the same place discharges rather than fails, so cycle 7
        // is a pass. Counting failures alone cannot tell an accept from a reset
        // — the match is the whole of the difference.
        .{
            .tag = .prop_accept_on,
            .inside = false,
            .failures = .{ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
            .matches = .{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
        },
        // Pushed down into the consequent: the rewrite that is not one. The
        // abort now arms a cycle later, at the seed, where the condition has
        // already gone low — so cycle 7 raises nothing and the attempt it should
        // have resolved passes at cycle 8 instead. Two reports lost.
        .{
            .tag = .prop_reject_on,
            .inside = true,
            .failures = .{ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
            .matches = .{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
        },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const index = try build(&ir, gpa, case.tag, case.inside);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(!case.inside, monitor.aborts[0].covers_trigger);

        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);
        for (trace, 0..) |cycle, at| {
            const atoms = [_]Word{cycle};
            const report = state.stepAtoms(&ir, &atoms);
            try testing.expectEqual(case.failures[at], report.failures);
            try testing.expectEqual(case.matches[at], report.matches);
        }
        // Cycle 9's antecedent matched, but the trace ends before its obligation
        // is seeded and the property is weak, so nothing is owed. This is the
        // fixture's "nothing at 105".
        const final = state.finish();
        try testing.expectEqual(@as(u32, 0), final.failures);
    }
}

test "an accept above an implication discharges where a reset abandons" {
    const gpa = testing.allocator;

    // The distinction drawn one layer up from where the obligation-scoped test
    // draws it, on the path only a trigger-covering abort takes. All three
    // spellings below stop the attempt from failing and no count of failures
    // separates them; the accept reports a *match* for the attempt it resolved,
    // and the reset reports nothing at all, because an abandoned attempt leaves
    // no verdict.
    //
    // Cycle 0 has the antecedent and the condition both high, so the abort lands
    // in the cycle the match ends. Cycle 1 has `e` low, which is what the bare
    // property fails on.
    const trace = [_]u64{ 0b101, 0b000 };

    const Case = struct {
        wrap: enum { none, accept, reset },
        failures: u32,
        matches: u32,
        reset_at_zero: bool,
    };
    for ([_]Case{
        .{ .wrap = .none, .failures = 1, .matches = 0, .reset_at_zero = false },
        .{ .wrap = .accept, .failures = 0, .matches = 1, .reset_at_zero = false },
        .{ .wrap = .reset, .failures = 0, .matches = 0, .reset_at_zero = true },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const cond = try ir.atom(gpa, @enumFromInt(2));
        var property = try ir.propImplies(
            gpa,
            .prop_implies_nonoverlap,
            try atomCycle(&ir, gpa, 0),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1)),
        );
        if (case.wrap == .accept) {
            property = try ir.propAbort(gpa, .prop_accept_on, cond, property);
        }
        const index = try addDirective(
            &ir,
            gpa,
            property,
            if (case.wrap == .reset) cond else null,
        );

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        var state = try Monitor.State.init(gpa, &monitor);
        defer state.deinit(gpa);

        var failures: u64 = 0;
        var matches: u64 = 0;
        for (trace, 0..) |cycle, at| {
            const atoms = [_]Word{cycle};
            const report = state.stepAtoms(&ir, &atoms);
            failures += report.failures;
            matches += report.matches;
            if (at == 0) try testing.expectEqual(case.reset_at_zero, report.reset);
        }
        try testing.expectEqual(case.failures, failures);
        try testing.expectEqual(case.matches, matches);
    }
}

test "differential: an abort above an implication" {
    const gpa = testing.allocator;

    // Atom 0 is the antecedent, 1 the consequent's term, 2 the condition. The
    // consequent takes two cycles, so an attempt is pending in a cycle the
    // condition can catch it in as well as in the cycle its antecedent matched
    // — the two paths an abort above an implication takes are both exercised on
    // the same property.
    for ([_]Ir.Node.Tag{
        .prop_accept_on,
        .prop_reject_on,
        .prop_sync_accept_on,
        .prop_sync_reject_on,
    }) |tag| {
        for ([_]Ir.Node.Tag{ .prop_implies_overlap, .prop_implies_nonoverlap }) |implies| {
            for ([_]Ir.Node.Tag{ .prop_seq_weak, .prop_seq_strong }) |strength| {
                var ir: Ir = .init();
                defer ir.deinit(gpa);

                const cond = try ir.atom(gpa, @enumFromInt(2));
                const antecedent = try atomCycle(&ir, gpa, 0);
                const consequent = try ir.propSeq(gpa, strength, try ir.seqDelay(
                    gpa,
                    try atomCycle(&ir, gpa, 1),
                    1,
                    1,
                    try atomCycle(&ir, gpa, 1),
                ));
                const index = try addDirective(&ir, gpa, try ir.propAbort(
                    gpa,
                    tag,
                    cond,
                    try ir.propImplies(gpa, implies, antecedent, consequent),
                ), null);

                // The same abort around the consequent alone: what an attempt
                // owes once its antecedent has matched and no abort intervened.
                const seeded = try ir.propAbort(gpa, tag, cond, consequent);
                const effect: Monitor.Abort.Effect = switch (tag) {
                    .prop_accept_on, .prop_sync_accept_on => .accept,
                    else => .reject,
                };

                try differentialAbortOverImplication(
                    gpa,
                    &ir,
                    index,
                    antecedent,
                    implies == .prop_implies_overlap,
                    seeded,
                    &.{.{ .cond = cond, .effect = effect }},
                    // A one-cycle antecedent cannot be caught halfway.
                    null,
                    3,
                    4,
                );
            }
        }
    }
}

test "differential: an abort above a multi-cycle antecedent" {
    const gpa = testing.allocator;

    // The lift. An antecedent spanning up to three cycles means an attempt can be
    // caught halfway through matching one, and each of those owes a verdict of its
    // own — which is what `trigger_window` recovered and what used to make this
    // shape a refusal.
    //
    // The reference counts them from the delay's meaning rather than from the
    // monitor: `InFlight` is "left held at the start and some end is still
    // available", and nothing in it knows what a trigger state is. So agreeing
    // with it over every trace is a claim about the semantics, and a monitor that
    // resolved only the matches ENDING in the abort's cycle disagrees on the first
    // trace where an attempt is halfway.
    for ([_]Ir.Node.Tag{ .prop_accept_on, .prop_reject_on }) |tag| {
        for ([_]Ir.Node.Tag{ .prop_implies_overlap, .prop_implies_nonoverlap }) |implies| {
            const Range = struct { lo: u32, hi: u32 };
            for ([_]Range{
                // Fixed length: no two matches can end together, so the window
                // exists for the halfway attempts alone.
                .{ .lo = 1, .hi = 1 },
                // And varying, where both reasons for a window apply at once.
                .{ .lo = 1, .hi = 3 },
            }) |range| {
                var ir: Ir = .init();
                defer ir.deinit(gpa);

                const cond = try ir.atom(gpa, @enumFromInt(2));
                // `InFlight` reads the two booleans, so they are named here and
                // the antecedent is built from them rather than the other way on.
                const opening = try ir.atom(gpa, @enumFromInt(0));
                const closing = try ir.atom(gpa, @enumFromInt(1));
                const antecedent = try ir.seqDelay(
                    gpa,
                    try ir.seqBoolean(gpa, opening),
                    range.lo,
                    range.hi,
                    try ir.seqBoolean(gpa, closing),
                );
                const consequent = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 1));
                const index = try addDirective(&ir, gpa, try ir.propAbort(
                    gpa,
                    tag,
                    cond,
                    try ir.propImplies(gpa, implies, antecedent, consequent),
                ), null);

                const seeded = try ir.propAbort(gpa, tag, cond, consequent);
                const effect: Monitor.Abort.Effect = switch (tag) {
                    .prop_accept_on => .accept,
                    else => .reject,
                };

                try differentialAbortOverImplication(
                    gpa,
                    &ir,
                    index,
                    antecedent,
                    implies == .prop_implies_overlap,
                    seeded,
                    &.{.{ .cond = cond, .effect = effect }},
                    .{ .left = opening, .right = closing, .lo = range.lo, .hi = range.hi },
                    3,
                    5,
                );
            }
        }
    }
}

test "differential: nested aborts of opposite effect" {
    const gpa = testing.allocator;

    // Which abort wins when two of them resolve the same attempt in the same
    // cycle. LRM 16.12.14 reads outward: the inner `reject_on` makes the inner
    // evaluation FALSE, and the outer `accept_on` then sees its condition true
    // during an evaluation that is still its own, so the overall result is TRUE.
    // The outermost abort wins, and lowering appends outward, so that is the
    // LAST entry of `monitor.aborts` rather than the first.
    //
    // Both orders exist here, so this fails if the runtime scans the list in
    // either direction unconditionally rather than resolving outward.
    for ([_][2]Ir.Node.Tag{
        .{ .prop_reject_on, .prop_accept_on },
        .{ .prop_accept_on, .prop_reject_on },
    }) |order| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqDelay(
            gpa,
            try atomCycle(&ir, gpa, 0),
            1,
            1,
            try atomCycle(&ir, gpa, 0),
        ));
        const inner = try ir.propAbort(gpa, order[0], try ir.atom(gpa, @enumFromInt(1)), body);
        const outer = try ir.propAbort(gpa, order[1], try ir.atom(gpa, @enumFromInt(2)), inner);
        const index = try addDirective(&ir, gpa, outer, null);
        try differentialMonitor(gpa, &ir, index, 3, 4, null);
    }
}

test "a junction of operands with different strength is refused" {
    const gpa = testing.allocator;

    // The pair is pending until both sides resolve, and a pending pair with one
    // weak and one strong side has no single disposition to carry to the end of
    // the trace. Refusing beats picking one.
    for ([_]Ir.Node.Tag{ .prop_and, .prop_or }) |junction| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const weak = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0));
        const strong = try ir.propSeq(gpa, .prop_seq_strong, try atomCycle(&ir, gpa, 1));
        const property = try ir.propJunction(gpa, junction, weak, strong);
        const index = try addDirective(&ir, gpa, property, null);

        var diagnostic: Diagnostic = undefined;
        try testing.expectError(
            error.MixedStrength,
            assertion(gpa, &ir, index, .{}, &diagnostic),
        );
        try testing.expectEqual(property, diagnostic.node);
    }
}

test "an abort below an operator that renumbers its operand is refused" {
    const gpa = testing.allocator;

    // An abort is a mask over obligation states. A shift and a disjunction move
    // an operand's states as a block, so the mask survives; a product and a
    // fusion do not, and an accept under a product would not be a mask operation
    // at all — it has to record that one side of a pair is satisfied, which is a
    // state remap.
    //
    // TODO: carrying a mask through a product needs the position mapping every
    // combinator would have to publish. Until then these are refused.
    const build = struct {
        fn go(ir: *Ir, alloc: Allocator, over: Ir.Node.Tag) !Ir.Assertion.Index {
            const guarded = try ir.propAbort(
                alloc,
                .prop_reject_on,
                try ir.atom(alloc, @enumFromInt(1)),
                try ir.propSeq(alloc, .prop_seq_weak, try atomCycle(ir, alloc, 0)),
            );
            const property = switch (over) {
                .prop_and => try ir.propJunction(
                    alloc,
                    .prop_and,
                    guarded,
                    try ir.propSeq(alloc, .prop_seq_weak, try atomCycle(ir, alloc, 2)),
                ),
                .prop_always => try ir.propRange(alloc, .prop_always, 1, 2, guarded),
                .prop_followed_by_overlap => try ir.propFollowedBy(
                    alloc,
                    .prop_followed_by_overlap,
                    try atomCycle(ir, alloc, 2),
                    guarded,
                ),
                else => unreachable,
            };
            return addDirective(ir, alloc, property, null);
        }
    }.go;

    for ([_]Ir.Node.Tag{ .prop_and, .prop_always, .prop_followed_by_overlap }) |over| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const index = try build(&ir, gpa, over);
        try testing.expectError(
            error.UnsupportedAbortScope,
            assertion(gpa, &ir, index, .{}, null),
        );
    }

    // The two that do carry it, for contrast: a disjunction and a shift.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const guarded = try ir.propAbort(
            gpa,
            .prop_reject_on,
            try ir.atom(gpa, @enumFromInt(1)),
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0)),
        );
        const disjunction = try ir.propJunction(
            gpa,
            .prop_or,
            guarded,
            try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
        );
        const property = try ir.propRange(gpa, .prop_delay, 1, 2, disjunction);
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
        // Still exactly the one state it started as, two operators later.
        try testing.expectEqual(@as(u32, 1), @popCount(monitor.maskOf(0)[0]));
    }
}

test "the sync forms differ from the plain ones only in which valuation they ask for" {
    const gpa = testing.allocator;

    // libsva is handed root bits already evaluated, so it cannot tell a sampled
    // valuation from a live one and neither can any test here. What it *can* do
    // is carry the request to the backend, which is the whole of the `sync_`
    // distinction at this layer — exactly as `Assertion.reset` carries it.
    //
    // Anything past this is a front-end obligation and unverifiable from inside
    // the library.
    const Case = struct { tag: Ir.Node.Tag, effect: Monitor.Abort.Effect, sampled: bool };
    for ([_]Case{
        .{ .tag = .prop_accept_on, .effect = .accept, .sampled = false },
        .{ .tag = .prop_reject_on, .effect = .reject, .sampled = false },
        .{ .tag = .prop_sync_accept_on, .effect = .accept, .sampled = true },
        .{ .tag = .prop_sync_reject_on, .effect = .reject, .sampled = true },
    }) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const body = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 0));
        const property = try ir.propAbort(gpa, case.tag, try ir.atom(gpa, @enumFromInt(1)), body);
        const index = try addDirective(&ir, gpa, property, null);

        var monitor = try assertion(gpa, &ir, index, .{}, null);
        defer monitor.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), monitor.aborts.len);
        try testing.expectEqual(case.effect, monitor.aborts[0].effect);
        try testing.expectEqual(case.sampled, monitor.aborts[0].sampled);
        // The condition is a root of its own, kept whether or not a guard
        // mentions it, exactly like the reset.
        try testing.expect(monitor.aborts[0].cond < monitor.alphabet.roots.len);
    }
}

/// One way of spelling a property, for the mutation sweep below.
const Spelling = enum {
    conjunction,
    disjunction,
    nexttime_1,
    nexttime_2,
    nexttime_3,
    always_1_2,
    eventually_1_2,
    followed_by_overlap,
    followed_by_nonoverlap,
    implies_overlap,
    accept_on,
    reject_on,
    reset_wide,
    abort_in_shift,
    abort_over_shift,
    reject_in_or,
    reject_over_or,
    trigger_accept_over,
    trigger_reject_over,
    trigger_accept_under,
    trigger_reject_under,
    trigger_reset,
    trigger_every_match,
    trigger_first_match,
    negation,
    negation_operand,
    negation_strong,
    implication,
    implication_swapped,

    /// Atom 0 is the body's term, atom 1 the abort condition, atom 2 a second
    /// body term, so every spelling here is comparable against every other.
    fn build(spelling: Spelling, ir: *Ir, gpa: Allocator) !Ir.Assertion.Index {
        const cond = try ir.atom(gpa, @enumFromInt(1));
        const now = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(ir, gpa, 0));
        const next = try ir.propSeq(gpa, .prop_seq_weak, try leadingDelay(
            ir,
            gpa,
            1,
            1,
            try atomCycle(ir, gpa, 0),
        ));
        const later = try ir.propSeq(gpa, .prop_seq_weak, try leadingDelay(
            ir,
            gpa,
            2,
            2,
            try atomCycle(ir, gpa, 2),
        ));
        const other = try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(ir, gpa, 2));

        const property = switch (spelling) {
            .conjunction => try ir.propJunction(gpa, .prop_and, now, later),
            .disjunction => try ir.propJunction(gpa, .prop_or, now, later),
            .nexttime_1 => try ir.propRange(gpa, .prop_delay, 1, 1, now),
            .nexttime_2 => try ir.propRange(gpa, .prop_delay, 2, 2, now),
            .nexttime_3 => try ir.propRange(gpa, .prop_delay, 3, 3, now),
            .always_1_2 => try ir.propRange(gpa, .prop_always, 1, 2, now),
            .eventually_1_2 => try ir.propRange(gpa, .prop_delay, 1, 2, now),
            .followed_by_overlap => try ir.propFollowedBy(
                gpa,
                .prop_followed_by_overlap,
                try atomCycle(ir, gpa, 0),
                other,
            ),
            .followed_by_nonoverlap => try ir.propFollowedBy(
                gpa,
                .prop_followed_by_nonoverlap,
                try atomCycle(ir, gpa, 0),
                other,
            ),
            .implies_overlap => try ir.propImplies(
                gpa,
                .prop_implies_overlap,
                try atomCycle(ir, gpa, 0),
                other,
            ),
            .accept_on => try ir.propAbort(gpa, .prop_accept_on, cond, next),
            .reject_on => try ir.propAbort(gpa, .prop_reject_on, cond, next),
            // The same obligation with the condition as the directive's reset:
            // `disable iff` rather than an abort.
            .reset_wide => return addDirective(ir, gpa, next, cond),
            .abort_in_shift => try ir.propRange(
                gpa,
                .prop_delay,
                2,
                2,
                try ir.propAbort(gpa, .prop_accept_on, cond, now),
            ),
            .abort_over_shift => try ir.propAbort(
                gpa,
                .prop_accept_on,
                cond,
                try ir.propRange(gpa, .prop_delay, 2, 2, now),
            ),
            .reject_in_or => try ir.propJunction(
                gpa,
                .prop_or,
                try ir.propAbort(gpa, .prop_reject_on, cond, next),
                later,
            ),
            .reject_over_or => try ir.propAbort(
                gpa,
                .prop_reject_on,
                cond,
                try ir.propJunction(gpa, .prop_or, next, later),
            ),
            // An abort above an implication, and the same abort inside its
            // consequent. Atom 2 is the antecedent here rather than a second
            // body term, which is what makes these five a family of their own.
            .trigger_accept_over => try ir.propAbort(
                gpa,
                .prop_accept_on,
                cond,
                try ir.propImplies(gpa, .prop_implies_nonoverlap, try atomCycle(ir, gpa, 2), now),
            ),
            .trigger_reject_over => try ir.propAbort(
                gpa,
                .prop_reject_on,
                cond,
                try ir.propImplies(gpa, .prop_implies_nonoverlap, try atomCycle(ir, gpa, 2), now),
            ),
            .trigger_accept_under => try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                try atomCycle(ir, gpa, 2),
                try ir.propAbort(gpa, .prop_accept_on, cond, now),
            ),
            .trigger_reject_under => try ir.propImplies(
                gpa,
                .prop_implies_nonoverlap,
                try atomCycle(ir, gpa, 2),
                try ir.propAbort(gpa, .prop_reject_on, cond, now),
            ),
            // The trigger-covering abort lowered as a monitor-wide reset.
            .trigger_reset => return addDirective(
                ir,
                gpa,
                try ir.propImplies(gpa, .prop_implies_nonoverlap, try atomCycle(ir, gpa, 2), now),
                cond,
            ),
            // Two spellings of one antecedent that differ in NOTHING BUT how many
            // attempts match in a cycle, which is what makes them the pair a
            // collapsing trigger cannot tell apart.
            //
            // `x ##[1:2] x` over three consecutive `x` has two matches ending in
            // the third cycle, from the first and second. `first_match` keeps only
            // the earliest match of each start, so it drops the longer of the two
            // — and the cycle it ended in is still an end, because the start one
            // cycle later reaches it at the shorter gap. So the two spellings
            // always match in the same CYCLES and sometimes a different number of
            // times, and a monitor reporting one verdict per matching cycle
            // reports the same thing for both.
            .trigger_every_match => try ir.propImplies(
                gpa,
                .prop_implies_overlap,
                try ir.seqDelay(gpa, try atomCycle(ir, gpa, 2), 1, 2, try atomCycle(ir, gpa, 2)),
                now,
            ),
            .trigger_first_match => try ir.propImplies(
                gpa,
                .prop_implies_overlap,
                try ir.seqFirstMatch(gpa, try ir.seqDelay(
                    gpa,
                    try atomCycle(ir, gpa, 2),
                    1,
                    2,
                    try atomCycle(ir, gpa, 2),
                )),
                now,
            ),
            // A complement, and the operand it complements. Every row below
            // uses a two-cycle operand so that an attempt can still be pending
            // when the trace ends, which is where strength is observable at
            // all; a one-cycle operand resolves every attempt in its own cycle
            // and would hide the difference between the last two.
            .negation => try ir.propNot(gpa, try ir.propSeq(
                gpa,
                .prop_seq_weak,
                try ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 1, try atomCycle(ir, gpa, 2)),
            )),
            .negation_operand => try ir.propSeq(
                gpa,
                .prop_seq_weak,
                try ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 1, try atomCycle(ir, gpa, 2)),
            ),
            // The same complement over a strong operand. It differs from
            // `negation` in the end-of-trace disposition and in nothing else, so
            // a lowering that swapped the masks but left the strength alone
            // produces one where the other is meant.
            .negation_strong => try ir.propNot(gpa, try ir.propSeq(
                gpa,
                .prop_seq_strong,
                try ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 1, try atomCycle(ir, gpa, 2)),
            )),
            // `p implies q`, and the same two operands the other way round —
            // which is what complementing the wrong side produces.
            .implication => try ir.propJunction(gpa, .prop_implies, now, other_strong: {
                break :other_strong try ir.propSeq(
                    gpa,
                    .prop_seq_strong,
                    try atomCycle(ir, gpa, 2),
                );
            }),
            .implication_swapped => try ir.propJunction(
                gpa,
                .prop_implies,
                try ir.propSeq(gpa, .prop_seq_strong, try atomCycle(ir, gpa, 2)),
                now,
            ),
        };
        return addDirective(ir, gpa, property, null);
    }
};

test "mutation sweep: every wrong spelling is reachable by a test" {
    const gpa = testing.allocator;

    // A mutation sweep with no compiler to mutate. Each row is a way of getting
    // one of these operators wrong that the *lowering* could plausibly implement
    // — a swapped tag, an off-by-one offset, a mask that is too wide, an abort
    // lowered as a reset — spelled out as the IR it would have produced. If the
    // right and wrong spellings are indistinguishable over every short trace,
    // then no test can be failing on that mutation and the operator is untested
    // however green the suite is.
    //
    // `killed_by` names the test that does the failing. That test and this row
    // are a pair: this one shows the mutation is observable, that one shows the
    // observation is checked against the semantics.
    const Mutation = struct {
        right: Spelling,
        wrong: Spelling,
        killed_by: []const u8,
    };
    for ([_]Mutation{
        .{
            .right = .conjunction,
            .wrong = .disjunction,
            .killed_by = "differential: property conjunction and disjunction",
        },
        .{
            .right = .nexttime_2,
            .wrong = .nexttime_1,
            .killed_by = "differential: nexttime, bounded eventually and bounded always",
        },
        .{
            .right = .nexttime_2,
            .wrong = .nexttime_3,
            .killed_by = "differential: nexttime, bounded eventually and bounded always",
        },
        // The conjunctive and disjunctive readings of one range. This is the
        // mutation a shared delay spine would have made invisible.
        .{
            .right = .always_1_2,
            .wrong = .eventually_1_2,
            .killed_by = "differential: nexttime, bounded eventually and bounded always",
        },
        .{
            .right = .followed_by_overlap,
            .wrong = .followed_by_nonoverlap,
            .killed_by = "differential: followed-by",
        },
        // Followed-by lowered as though it were implication: the same automaton
        // with vacuity the right way up instead of inverted.
        .{
            .right = .followed_by_overlap,
            .wrong = .implies_overlap,
            .killed_by = "followed-by fails where implication holds vacuously",
        },
        .{
            .right = .accept_on,
            .wrong = .reject_on,
            .killed_by = "differential: an accept abort and a reject abort",
        },
        // The one the scoping document called out: an accept *discharges* and a
        // reset *abandons*, and lowering the first as the second suppresses the
        // same failures while producing none of the passes.
        .{
            .right = .accept_on,
            .wrong = .reset_wide,
            .killed_by = "an accept discharges where a reset abandons",
        },
        // A mask covering the whole obligation instead of the abort's subtree,
        // which arms the abort during the cycles before its property starts.
        .{
            .right = .abort_in_shift,
            .wrong = .abort_over_shift,
            .killed_by = "an abort covers its own subtree and not the cycles before it",
        },
        // A reject under a disjunction left resolving the whole attempt instead
        // of being demoted to its own branch.
        .{
            .right = .reject_in_or,
            .wrong = .reject_over_or,
            .killed_by = "a reject under a disjunction kills only its own disjunct",
        },
        // An abort above an implication lowered as one inside its consequent:
        // the same abort armed a cycle late, from the seed rather than from the
        // antecedent match. This is the rewrite that looks like a rewrite and is
        // not, and the reject row is the one `diff_abort.sv` is built on.
        .{
            .right = .trigger_accept_over,
            .wrong = .trigger_accept_under,
            .killed_by = "an abort above an implication resolves the antecedent match instead of seeding",
        },
        .{
            .right = .trigger_reject_over,
            .wrong = .trigger_reject_under,
            .killed_by = "an abort above an implication resolves the antecedent match instead of seeding",
        },
        // The discharge/abandon confusion again, at the trigger this time. Both
        // suppress the same failure and only the accept reports the pass.
        .{
            .right = .trigger_accept_over,
            .wrong = .trigger_reset,
            .killed_by = "an accept above an implication discharges where a reset abandons",
        },
        .{
            .right = .trigger_accept_over,
            .wrong = .trigger_reject_over,
            .killed_by = "differential: an abort above an implication",
        },
        // The collapse. An antecedent match imposes an obligation per match, and
        // these two spellings match in the same cycles and differ only in how
        // many times — so this row goes red the moment a trigger answers with a
        // bit instead of a count, and it is the only row that does.
        .{
            .right = .trigger_every_match,
            .wrong = .trigger_first_match,
            .killed_by = "several antecedent matches ending in one cycle each owe a verdict",
        },
        // A `not` lowered as the identity — the complement dropped, or the two
        // masks left unswapped, which comes to the same automaton.
        .{
            .right = .negation,
            .wrong = .negation_operand,
            .killed_by = "differential: negation",
        },
        // The complement built correctly and the strength left alone. Only the
        // end of the trace can see it, and only on an attempt still pending
        // there, which is the case a `not` test is likeliest to omit.
        .{
            .right = .negation,
            .wrong = .negation_strong,
            .killed_by = "strength inverts under negation, and decides a pending complement",
        },
        // `p implies q` with the wrong side complemented, which is the same
        // disjunction with its operands exchanged.
        .{
            .right = .implication,
            .wrong = .implication_swapped,
            .killed_by = "differential: property implication",
        },
    }) |mutation| {
        var right: Ir = .init();
        defer right.deinit(gpa);
        const right_index = try mutation.right.build(&right, gpa);

        var wrong: Ir = .init();
        defer wrong.deinit(gpa);
        const wrong_index = try mutation.wrong.build(&wrong, gpa);

        var a: std.ArrayListUnmanaged(u64) = .empty;
        defer a.deinit(gpa);
        var b: std.ArrayListUnmanaged(u64) = .empty;
        defer b.deinit(gpa);
        try behaviour(gpa, &right, right_index, 3, 4, &a);
        try behaviour(gpa, &wrong, wrong_index, 3, 4, &b);

        if (std.mem.eql(u64, a.items, b.items)) {
            std.debug.print(
                "mutant {s} -> {s} is indistinguishable; '{s}' cannot be failing on it\n",
                .{ @tagName(mutation.right), @tagName(mutation.wrong), mutation.killed_by },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test {
    testing.refAllDecls(@This());
}

test "a match item's assignments land on the exit positions of its subsequence" {
    const gpa = testing.allocator;

    // LRM 16.10's `data_check`, shortened: `a ##1 (!a, x = data) ##1 b && (data_out
    // == x)`. The assignment is attached to the one-cycle subsequence `!a`, so it
    // belongs to that position and to no other -- and the position that READS `x`
    // carries no program of its own.
    var ir: Ir = .init();
    defer ir.deinit(gpa);

    const slot = try ir.localDeclare(gpa, 0, Ir.local_no_init);
    try testing.expectEqual(@as(u32, 0), slot);

    const capture = try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 1), &.{ slot, 7 });
    const tail = try atomCycle(&ir, gpa, 2);
    // Atom 2 is the comparison against `x`, so it reads slot 0.
    var reads = [_]Word{1};
    try ir.atomReads(gpa, @enumFromInt(2), &reads);

    const whole = try ir.seqDelay(
        gpa,
        try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1, capture),
        1,
        1,
        tail,
    );
    const directive = try addDirective(&ir, gpa, try ir.propSeq(gpa, .prop_seq_weak, whole), null);

    var monitor = try assertion(gpa, &ir, directive, .{}, null);
    defer monitor.deinit(gpa);

    // One site, writing slot 0 with handle 7, and the monitor is threaded because
    // it has a slot at all.
    try testing.expectEqual(Monitor.Mode.threaded, monitor.mode);
    try testing.expectEqual(@as(usize, 1), monitor.locals.len);
    try testing.expectEqual(@as(usize, 1), monitor.assign_slot.len);
    try testing.expectEqual(@as(u32, 0), monitor.assign_slot[0]);
    try testing.expectEqual(@as(u32, 7), monitor.assign_handle[0]);

    // Exactly one state carries a program, and it is not the state that reads.
    const obligation = &monitor.obligation;
    var carrying: u32 = 0;
    for (0..obligation.state_count) |index| {
        if (obligation.programOf(@intCast(index)).len != 0) carrying += 1;
    }
    try testing.expectEqual(@as(u32, 1), carrying);

    // The slot is live where it is still going to be read, and dead after the
    // comparison: liveness is what lets two groups holding different dead values
    // merge again.
    var live_anywhere = false;
    for (0..obligation.state_count) |index| {
        if (obligation.liveOf(@intCast(index))[0] & 1 != 0) live_anywhere = true;
    }
    try testing.expect(live_anywhere);

    // A root that reads a slot is evaluated per group; the others are not.
    var per_group: u32 = 0;
    for (0..monitor.alphabet.roots.len) |index| {
        if (Nfa.bitIsSet(monitor.root_local, @intCast(index))) per_group += 1;
    }
    try testing.expectEqual(@as(u32, 1), per_group);
}

test "the shapes one valuation per thread cannot represent are refused" {
    const gpa = testing.allocator;

    // Each of these is a shape where a group's single valuation would be wrong
    // rather than merely coarse; see the design doc's hazard list.
    const cases = [_]struct { which: []const u8, want: anyerror }{
        .{ .which = "first_match", .want = error.LocalUnderDeterminisation },
        .{ .which = "cross", .want = error.LocalCrossRead },
        .{ .which = "until", .want = error.LocalUnboundedAttempts },
    };

    for (cases) |case| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const slot = try ir.localDeclare(gpa, 0, Ir.local_no_init);
        const capture = try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 0), &.{ slot, 0 });

        // The reading atom exists only in the cross-read case, and declaring a
        // read for an atom no node mentions is a caller bug the IR asserts on.
        const reader = if (std.mem.eql(u8, case.which, "cross")) try atomCycle(&ir, gpa, 1) else undefined;
        if (std.mem.eql(u8, case.which, "cross")) {
            var reads = [_]Word{1};
            try ir.atomReads(gpa, @enumFromInt(1), &reads);
        }

        const property = if (std.mem.eql(u8, case.which, "first_match"))
            try ir.propSeq(gpa, .prop_seq_weak, try ir.seqFirstMatch(gpa, capture))
        else if (std.mem.eql(u8, case.which, "cross"))
            // The right operand reads the slot the left one assigns, which F.5.5
            // gives it no view of.
            try ir.propSeq(gpa, .prop_seq_weak, try ir.seqIntersect(gpa, capture, reader))
        else
            try ir.propUntil(
                gpa,
                .prop_until,
                try ir.propSeq(gpa, .prop_seq_weak, capture),
                try ir.propSeq(gpa, .prop_seq_weak, try atomCycle(&ir, gpa, 2)),
            );

        const directive = try addDirective(&ir, gpa, property, null);
        var diagnostic: Diagnostic = undefined;
        try testing.expectError(case.want, assertion(gpa, &ir, directive, .{}, &diagnostic));
    }
}

test "F.5.4's flow sets, and the product read that is legal after all" {
    const gpa = testing.allocator;

    // LRM 16.10's s8, which the standard calls LEGAL: both operands of an `and`
    // assign x, and the right operand READS it -- but only after assigning it
    // itself, so the read cannot see the sibling's value and one valuation per
    // thread is enough. The conservative test this replaces refused it.
    //
    //   (a ##1 (b, x = 1) ##1 c) and (d ##1 (`true, x = 2) ##0 (e == x))
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const x = try ir.localDeclare(gpa, 0, Ir.local_no_init);

        const left = try ir.seqDelay(
            gpa,
            try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 0), 1, 1,
                            try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 1), &.{ x, 0 })),
            1,
            1,
            try atomCycle(&ir, gpa, 2),
        );
        const right = try ir.seqDelay(
            gpa,
            try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 3), &.{ x, 1 }),
            0,
            0,
            try atomCycle(&ir, gpa, 4),
        );
        // Atom 4 is `e == x`, and its read mask can only be declared once the atom
        // exists -- the IR asserts on a handle no node mentions.
        var reads = [_]Word{1};
        try ir.atomReads(gpa, @enumFromInt(4), &reads);

        const property = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqAnd(gpa, left, right));
        const directive = try addDirective(&ir, gpa, property, null);
        var monitor = try assertion(gpa, &ir, directive, .{}, null);
        monitor.deinit(gpa);
    }

    // And the shape that really is unrepresentable: LRM 16.10's rule (a), where one
    // operand assigns x and the SIBLING reads it with nothing of its own written
    // first. F.5.5 gives that read the incoming context, which a shared valuation
    // cannot show it.
    {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const x = try ir.localDeclare(gpa, 0, Ir.local_no_init);
        const left = try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 0), &.{ x, 0 });
        const right = try ir.seqDelay(gpa, try atomCycle(&ir, gpa, 1), 1, 1,
                                      try atomCycle(&ir, gpa, 2));
        var reads = [_]Word{1};
        try ir.atomReads(gpa, @enumFromInt(2), &reads); // atom 2 is `e == x`
        const property = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqIntersect(gpa, left, right));
        const directive = try addDirective(&ir, gpa, property, null);
        var diagnostic: Diagnostic = undefined;
        try testing.expectError(error.LocalCrossRead, assertion(gpa, &ir, directive, .{}, &diagnostic));
    }
}

test "F.5.4's or rule: a slot flows out only if it flows out of both operands" {
    const gpa = testing.allocator;

    // `((a, x = 1) or (b, x = 2))` flows x out -- both operands assign it -- while
    // `((a, x = 1) or b)` does not, because the right branch leaves it unassigned.
    // The sets are what the state liveness and the merge are derived from, so this
    // pins the rule rather than its consequences.
    var ir: Ir = .init();
    defer ir.deinit(gpa);
    const x = try ir.localDeclare(gpa, 0, Ir.local_no_init);

    const both = try ir.seqOr(
        gpa,
        try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 0), &.{ x, 0 }),
        try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 1), &.{ x, 1 }),
    );
    const one = try ir.seqOr(
        gpa,
        try ir.seqMatch(gpa, try atomCycle(&ir, gpa, 0), &.{ x, 0 }),
        try atomCycle(&ir, gpa, 1),
    );

    var flow = try Flow.compute(gpa, &ir);
    defer flow.deinit(gpa);
    try testing.expectEqual(@as(Word, 1), flow.outOf(both)[0]);
    try testing.expectEqual(@as(Word, 0), flow.outOf(one)[0]);
    // Either way x is SAMPLED by the disjunction: F.5.4's sample is a union.
    try testing.expectEqual(@as(Word, 1), flow.sampleOf(one)[0]);
}

/// Which root of a monitor's alphabet is the bare atom `handle`, for a test that
/// has to set one bit of a valuation by hand.
fn rootOfAtom(ir: *const Ir, monitor: *const Monitor, handle: u32) ?u32 {
    for (monitor.alphabet.roots, 0..) |root, index| {
        if (ir.tagOf(root) != .atom) continue;
        if (ir.dataOf(root).lhs == handle) return @intCast(index);
    }
    return null;
}

test "the reference run carries a valuation through an attempt" {
    const gpa = testing.allocator;

    // `(a, x = d) ##1 (o == x)`: the value captured when `a` holds is compared one
    // cycle later. The trace makes the comparison true for the attempt that
    // captured 5 and false for the one that captured 9, so a run has to tell the
    // two attempts apart by their values and not merely by their cycles -- which is
    // the whole of LRM 16.10 in four cycles.
    var ir: Ir = .init();
    defer ir.deinit(gpa);

    const x = try ir.localDeclare(gpa, 0, Ir.local_no_init);
    const capture = try ir.seqMatch(gpa, try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(0))), &.{ x, 0 });
    const compare = try ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(1)));
    var reads = [_]Word{1};
    try ir.atomReads(gpa, @enumFromInt(1), &reads);

    const whole = try ir.seqDelay(gpa, capture, 1, 1, compare);
    const directive = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(0),
        .kind = .assert,
        .clock = .none,
        .reset = try ir.boolConst(gpa, false),
        .prop = try ir.propSeq(gpa, .prop_seq_weak, whole),
    });

    var monitor = try assertion(gpa, &ir, directive, .{}, null);
    defer monitor.deinit(gpa);
    try testing.expectEqual(Monitor.Mode.threaded, monitor.mode);

    // The harness: assignment site 0 stores whatever `d` holds this cycle, and the
    // comparison root holds when the frame's slot equals `o`. Both are the
    // front-end's business in a real backend; here they are two closures over the
    // trace.
    const Harness = struct {
        d: u64 = 0,
        o: u64 = 0,

        fn assign(user: ?*anyopaque, site: u32, frame: [*]const u64) callconv(.c) u64 {
            _ = site;
            _ = frame;
            const self: *@This() = @ptrCast(@alignCast(user.?));
            return self.d;
        }
        fn root(user: ?*anyopaque, index: u32, frame: [*]const u64) callconv(.c) bool {
            _ = index;
            const self: *@This() = @ptrCast(@alignCast(user.?));
            return frame[0] == self.o;
        }
    };
    var harness: Harness = .{};

    var run = try Monitor.State.init(gpa, &monitor);
    defer run.deinit(gpa);
    run.locals = .{ .assign = &Harness.assign, .root = &Harness.root, .user = &harness };

    const words = monitor.alphabet.words();
    const bits = try gpa.alloc(Word, words);
    defer gpa.free(bits);

    // Cycle 0: `a` holds and d is 5, so one attempt captures 5. Nothing resolves --
    // the comparison root is the second cycle's.
    harness.d = 5;
    @memset(bits, 0);
    Nfa.bitSet(bits, rootOfAtom(&ir, &monitor, 0).?);
    var report = run.step(bits);
    try testing.expectEqual(@as(u64, 0), report.matches);

    // Cycle 1: `a` holds again with d = 9, and o = 5. The attempt from cycle 0 sees
    // its own 5 and matches; the one starting now captures 9.
    harness.d = 9;
    harness.o = 5;
    @memset(bits, 0);
    Nfa.bitSet(bits, rootOfAtom(&ir, &monitor, 0).?);
    report = run.step(bits);
    try testing.expectEqual(@as(u64, 1), report.matches);

    // Cycle 2: o is 5 still, but the live attempt captured 9, so it fails -- a run
    // that shared one valuation would have reported a match here instead. Two
    // failures, not one: a bare sequence starts an attempt every cycle (the IR's
    // implicit quantification), and the one starting now has no first cycle because
    // `a` is low, which is a failure in the cycle it began.
    @memset(bits, 0);
    report = run.step(bits);
    try testing.expectEqual(@as(u64, 0), report.matches);
    try testing.expectEqual(@as(u64, 2), report.failures);
}
