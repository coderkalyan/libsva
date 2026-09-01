//! A compiled assertion: what a backend runs, once per cycle.
//!
//! # Structure
//!
//! An assertion is one *obligation* automaton plus, for an implication, a
//! *trigger* automaton that says when to start one.
//!
//! The two are tracked differently, and the asymmetry is the whole design:
//!
//!   - The **trigger** answers a question about *matches*: how many antecedent
//!     matches end this cycle. One state vector holds every attempt of it, and
//!     each state carries the set of attempts whose threads sit there, so the
//!     answer is a count and not a bit. What collapses is the *obligation* the
//!     matches impose — matches ending in one cycle impose the identical
//!     future, so one instance of it serves them all — and what does not
//!     collapse is how many attempts owe it. LRM 16.14.1 starts an evaluation
//!     of the consequent per match, so a verdict on the shared instance is that
//!     many verdicts, and reporting one is a wrong answer rather than a coarse
//!     one. See `# Counting the trigger`.
//!   - The **obligation** is per-attempt. Merging attempts into one state vector
//!     is *unsound*: one attempt's threads can die while another's live, and the
//!     union hides the failure. Each pending attempt therefore gets its own
//!     state vector, carrying the number of attempts it stands for.
//!
//! How the pending attempts are held is `mode`, and there are two disciplines
//! because there are two kinds of obligation.
//!
//! ## `.ring` — a pool of state vectors
//!
//! `depth` bounds how many attempts can be pending at once. It is the
//! obligation's longest match: an attempt seeded in cycle `t` has either matched
//! or run out of live states by cycle `t + depth - 1`, so `depth` vectors in a
//! ring is exact rather than a heuristic.
//!
//! ## `.counted` — a counter per state
//!
//! An obligation with no bounded longest match has no such ring: `always p`,
//! `s_eventually p` and the `until` family can all leave attempts pending
//! arbitrarily long, so arbitrarily many are in flight. Their obligation is
//! **determinised** instead, and totally — the death of an attempt is a state
//! (see `Glushkov.determinise`), not the absence of one.
//!
//! That buys back merging. One determinised state fully characterises an
//! attempt's future, so two attempts sitting in the same state may share it, and
//! the monitor carries a `u64` count per state rather than a vector per attempt.
//! What the count is *for* is the verdict: two attempts is two failures, and a
//! merged vector could only ever report one. A count reaching an exit is that
//! many matches; a count reaching a `dead` state is that many failures.
//!
//! Counters are `u64` and are not saturated. At the measured seven and a half
//! million ticks a second a `u32` overflows in about ten minutes of wall clock,
//! which is a reachable simulation; `u64` is not reachable by construction, so
//! the add stays one instruction.
//!
//! `depth` is meaningless in this mode and is zero. Two things follow that a
//! backend has to get right, and neither is optional:
//!
//!   - **One successor, chosen by lowest index.** A count must move along
//!     exactly ONE enabled transition. Several can be enabled at once even
//!     though the obligation is determinised, because a determinised state is a
//!     (subset, cell) pair and two cells may share a valuation. Such siblings
//!     are bisimilar — same subset, same exits, same successors — so the rule is
//!     to take the lowest-indexed enabled one, in `Nfa.chosenFollow`. The seed
//!     obeys the same rule over `first`, in `Nfa.chosenSeed`. Summing a count
//!     into every enabled successor instead manufactures attempts, and does so
//!     only on obligations whose cells happen to overlap, which is the worst
//!     kind of bug to go looking for.
//!   - **`dead`, not emptiness.** The construction is total, so a count never
//!     falls off the automaton and "no live states" never happens. Failure is
//!     `Nfa.dead`.
//!
//! # Counting the trigger
//!
//! `trigger_window` is the same insight one automaton over, and it is all the
//! per-attempt state an antecedent needs.
//!
//! An obligation is owed per *match*, and a match is an interval: LRM 16.9 has
//! a sequence matching over `[start, end]`, so what the cycle `end` has to
//! count is the number of distinct `start`s. Three facts follow, in the order
//! they constrain the representation.
//!
//! **Threads are not matches.** One attempt can reach an exit along several
//! paths in one cycle — `a ##[1:2] b ##[1:2] c` matches `[t, t+3]` through
//! either gap — and that is one match over one interval, not two. So the
//! per-state quantity cannot be an integer: adding thread counts at the exits
//! answers two. It has to be a *set*, whose union is idempotent.
//!
//! **A state cannot stand for an attempt, but it can stand for a set of them.**
//! Two attempts seeded in different cycles with a thread in one state do not
//! have the same future — each holds threads elsewhere too, and only this
//! state's own future is shared — so a shared state may not merge them. What it
//! can carry is which attempts are there: `State.trigger_ids` holds, per trigger
//! state, the set of attempts with a thread in it, and the advance maps those
//! sets forward along transitions by union. That maintains exactly the invariant
//! "an attempt is here if a thread of it is here", and the number of matches
//! ending now is the size of the union over the exit states.
//!
//! **An attempt's identity may be a rotating index.** Every trigger state is
//! co-reachable to an exit — `Glushkov.prune` keeps nothing else — so a thread
//! alive `k` cycles after its attempt was seeded witnesses a match of at least
//! `k + 1` cycles. With the window set to the trigger's longest match, an
//! attempt therefore has no thread left `trigger_window` cycles on, two ids that
//! far apart never coexist, and the index can rotate with the cycle. Nothing has
//! to retire an id: the attempt that last held it died of its own accord, which
//! `step` asserts rather than clearing.
//!
//! Two things that puts on a backend, and one it gets for free.
//!
//!   - **The longest match has to be bounded.** An antecedent whose reach is
//!     unbounded — `a ##[1:$] b`, `b[=2]` — has unboundedly many start cycles
//!     alive at once, and no fixed set of ids can name them. Such a property is
//!     refused, by `Glushkov.Error.AntecedentNotBounded` naming the antecedent,
//!     rather than counted to a bound and under-reported. `merge`'s
//!     `a ##[1:$] b` row measures eight attempts resolving in one cycle where a
//!     bit reports one; a wrong number is worse than no monitor.
//!
//!     TODO: counting one soundly needs the trigger determinised, where an
//!     attempt is a single state and a `u64` per state is exact. That is the
//!     `.counted` argument above, applied to the other automaton.
//!   - **A zero window is zero cost.** `trigger_window` is zero unless a
//!     boolean is genuinely too weak, and that needs two matches of *different*
//!     lengths to exist: an antecedent whose matches all span the same number of
//!     cycles ends them in that many distinct cycles, so at most one can end per
//!     cycle and `Nfa.matched` is the whole answer. `a |=> b` and
//!     `(a ##1 b) |-> c` alike allocate nothing and carry no id. The exception is
//!     an abort above the implication, which has to count the attempts still
//!     matching whether or not two of them can match together — see
//!     `Abort.covers_trigger`.
//!   - Free: a union dedupes an attempt against itself, so an ambiguous
//!     antecedent and a `first_match` need nothing said about them.
//!
//! # Complements
//!
//! `prop_not` adds nothing to this structure, which is the point. A complement
//! is lowered to an ordinary fragment — see `Glushkov.lowerPropNot` — whose
//! exits are the cycles its operand ran out of futures and whose emptiness is
//! the cycles its operand completed a match. Both disciplines above read it
//! without knowing it is one, and `Report` needs no new field: a hold of
//! `not p` arrives as a match and a fail as a failure, like any other property.
//!
//! Two things a complement *does* change, and both are `Verdict`'s business
//! rather than the automaton's. It inverts `strength`, because a pending
//! attempt stays pending under negation and only its end-of-trace disposition
//! turns over. And it refuses an `Abort` beneath it — not for want of a mask,
//! but because arming reads the *unfiltered* one-step image of where an attempt
//! is, and a complement has deleted the states that would have answered. See
//! `Glushkov.Builder.complement`.
//!
//! # The cycle
//!
//! `State` is a reference implementation of exactly this; a backend emits it.
//!
//!  1. Evaluate `alphabet.roots` into a bit per root.
//!  2. If `reset` holds, abandon everything in flight and stop. Nothing is
//!     seeded in a reset cycle; the next cycle starts fresh.
//!  3. Turn the root bits into a satisfied-state mask per automaton.
//!  4. Advance the trigger, then seed it — it starts an attempt every cycle,
//!     like any assertion. Count the antecedent matches that end now.
//!  5. If an `Abort` with `covers_trigger` holds, resolve every attempt with a
//!     live antecedent thread with that abort's verdict — one verdict each, not
//!     one between them — clear their threads out of the trigger, and let them
//!     seed nothing.
//!  6. Advance every pending obligation attempt.
//!  7. Seed a new attempt: every cycle when there is no trigger, in the cycle
//!     the trigger fires for `|->`, and the cycle after for `|=>`. It stands for
//!     as many attempts as step 4 counted matches.
//!  8. Apply every armed `Abort` whose condition holds. An abort is armed for
//!     an attempt when that attempt has a live state in the abort's mask. In
//!     `.counted` mode it is armed for a *state*, so it retires every attempt
//!     counted there at once — see `Abort.Scope`.
//!  9. Take verdicts. An attempt that reached an accepting state is discharged;
//!     one with no live states — in `.counted` mode, one in a `dead` state —
//!     failed. Both leave the pool, reporting as many verdicts as they stand for
//!     attempts.
//! 10. When a finite trace ends, a still-pending attempt holds if the property
//!     is weak and fails if it is strong.
//!
//! Seeding before taking verdicts is not incidental: under `|->` the obligation
//! starts on the trigger's last cycle, so a one-cycle consequent has to be able
//! to discharge in the cycle it was created. Aborting between the two is the
//! same argument — an `accept_on` whose condition holds in the cycle an attempt
//! starts discharges that attempt.
//!
//! Step 5 is that argument one cycle earlier again, and it is why an abort above
//! an implication is a different property from one inside its consequent. Above
//! it, the abort covers the cycles the antecedent is being matched in; below it,
//! it arms only once the consequent starts, which for `|=>` is the cycle after.
//! An abort firing in the cycle the antecedent matches is a verdict the
//! pushed-down spelling never produces, so pushing one down is not a rewrite.
//!
//! Note what step 2 and step 8 do differently, because it is the difference
//! between `disable iff` and `accept_on`: a reset *abandons*, so the attempt
//! leaves no verdict behind, while an accept *discharges*, which is a pass.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Nfa = @import("Nfa.zig");

const Word = Nfa.Word;
const Monitor = @This();

/// The directive this came from, for a front-end to report against.
id: Ir.Assertion.Id,
/// What to do with a verdict.
kind: Ir.Assertion.Kind,
/// What delimits a cycle. Metadata: a backend is stepped by its caller.
clock: Ir.Clock,
/// Terms and interned guards, shared by both automata so a term appearing in
/// both the antecedent and the consequent is evaluated once.
alphabet: Nfa.Alphabet,
/// The antecedent of an implication, or null when the property is a bare
/// sequence and an obligation starts every cycle.
trigger: ?Nfa,
/// Whether an obligation starts on the trigger's last cycle (`|->`) or the
/// cycle after it (`|=>`). Meaningless without a trigger.
overlap: bool,
/// The property's sequence. One attempt per seeding. Determinised, and total,
/// when `mode` is `.counted`.
obligation: Nfa,
/// How the pending attempts are held.
mode: Mode,
/// How many attempts can be pending at once — the obligation's longest match,
/// and so the number of state vectors a backend allocates.
///
/// Zero, and meaningless, when `mode` is `.counted`: there is no bound to
/// report, which is the whole reason that mode exists. A counted backend
/// allocates one `u64` per obligation state instead.
depth: u32,
/// How many antecedent attempts the trigger tracks apart — its longest match,
/// and so the number of attempt ids a backend rotates through.
///
/// Zero when nothing has to be told apart, which is the common case and costs a
/// backend the whole mechanism: `fires` is then `Nfa.matched`, one bit, exactly
/// as it was. See `# Counting the trigger` for when a zero is available and why
/// an unbounded antecedent is refused instead of tracked. Always zero without a
/// trigger.
trigger_window: u32,
/// What a pending attempt does when a finite trace ends.
strength: Strength,
/// Index into `alphabet.roots` of the reset condition. Always a real root: an
/// assertion that never resets carries a constant false there, which costs a
/// backend nothing to fold away and costs this one branch fewer.
reset: u32,
/// Aborts scoped to part of the obligation, in no particular order. Empty for
/// a property that has none, which is the common case.
aborts: []Abort,
/// One obligation-state mask per entry of `aborts`, in that order and each
/// `obligation.words` long. Read it with `maskOf`.
abort_masks: []Word,
/// The LRM 16.10 local variable slots, in declaration order: an opaque type and
/// an opaque initializer handle apiece, neither of which this library reads.
/// Empty for an assertion without any, which is what every local-keyed table
/// below is gated on.
locals: []const Ir.Local,
/// Words in one slot mask: `locals.len` rounded up to 64, or zero without slots.
local_words: u32,
/// Per assignment SITE, dense from zero in the order the `seq_match` nodes were
/// lowered, the slot it writes and the handle the backend evaluates for its
/// value. A state's program (`Nfa.programOf`) is a list of indices into these.
assign_slot: []const u32,
assign_handle: []const u32,
/// Which ROOTS read a local variable slot, one bit each: those are evaluated
/// once per valuation group rather than once per cycle, and the rest stay where
/// they were. `(alphabet.roots.len + 63) / 64` words, all zero without slots --
/// which is what keeps a local-free monitor's step exactly what it was.
root_local: []const Word,

/// What a monitor knows about one attempt. Three values, because a monitor
/// watches a finite prefix of a trace and a prefix does not decide every
/// property: an attempt that has neither completed a match nor run out of
/// futures is *pending*, and pending is a verdict rather than the absence of
/// one.
///
/// Naming it is what `prop_not` costs. Complementing an automaton exchanges
/// match for no-match, which is two of these; the third has no automaton to
/// invert it and needs the rule written down:
///
///     not holds   = fails
///     not fails   = holds
///     not pending = pending
///
/// The last line is the load-bearing one. It says a complement may not resolve
/// an attempt one cycle before its operand resolves, however certain the
/// outcome has become — which is exactly what makes `not` an involution, and is
/// the reason `Glushkov.Plan.pending` exists. Without it `not not p` would drift
/// earlier than `p` on every trace where a match becomes inevitable before it
/// completes.
pub const Verdict = enum(u8) {
    /// Neither decided yet. The trace so far permits both outcomes.
    pending = 0,
    /// The attempt is discharged: it holds, and a `cover` counts it.
    holds = 1,
    /// The attempt failed.
    fails = 2,

    pub fn not(verdict: Verdict) Verdict {
        return switch (verdict) {
            .pending => .pending,
            .holds => .fails,
            .fails => .holds,
        };
    }
};

pub const Strength = enum(u8) {
    /// A pending attempt holds when the trace ends.
    weak,
    /// A pending attempt fails when the trace ends.
    strong,

    /// What the end of a finite trace makes of an attempt still pending — the
    /// whole content of the weak/strong distinction, as a `Verdict`.
    pub fn atEnd(strength: Strength) Verdict {
        return switch (strength) {
            .weak => .holds,
            .strong => .fails,
        };
    }

    /// The dual, LRM 16.12.2. `not` swaps the two, because negating a pending
    /// attempt leaves it pending and only its end-of-trace disposition inverts.
    pub fn not(strength: Strength) Strength {
        return switch (strength) {
            .weak => .strong,
            .strong => .weak,
        };
    }
};

pub const Mode = enum(u8) {
    /// One state vector per pending attempt, `depth` of them. The obligation's
    /// longest match is bounded, so the pool is too.
    ring,
    /// One `u64` counter per obligation state, and the obligation determinised.
    /// See the file comment for the two rules this mode puts on a backend.
    counted,
    /// Group-based stepping with local variables. One group per distinct valuation
    /// per attempt, with state masks and local values tracked separately.
    threaded,
};

/// An abort scoped to a subtree of the obligation: `accept_on` and friends.
///
/// Where `reset` abandons every attempt of the whole monitor, an abort resolves
/// one attempt and only while the subtree it guards is still being evaluated.
/// Which states those are is `maskOf`, and an abort is *armed* for an attempt
/// exactly when that attempt has a live state in the mask — so the two cycles
/// `nexttime[2] (reject_on(b) p)` spends in its filler chain are cycles the
/// abort cannot fire in, without anything having to track a window.
pub const Abort = struct {
    /// Index into `alphabet.roots` of the condition.
    cond: u32,
    /// What firing does to the attempt.
    effect: Effect,
    /// How far up the property that effect reaches.
    scope: Scope,
    /// Whether the abort covers the trigger as well as the obligation: an abort
    /// written *above* the implication rather than inside its consequent.
    ///
    /// A monitor has exactly one trigger — every operator that could put an
    /// abort beside a second one refuses first — so an abort above it covers
    /// every attempt of that trigger, uniformly: the condition is a fact about
    /// this cycle, and every attempt with a live antecedent thread is one this
    /// cycle is inside the evaluation of. Uniformly is not collectively, though.
    /// Each of those attempts is a separate evaluation and owes a verdict of its
    /// own, so `step` resolves as many as there are — `trigger_window` is what
    /// counts them — clears their threads out of the trigger so none can
    /// complete an antecedent later and impose an obligation on an attempt that
    /// has already been resolved, and seeds nothing from them.
    ///
    /// An attempt with no live antecedent thread is not one of them. It can no
    /// longer match, so the implication already holds vacuously and there is
    /// nothing left under evaluation to abort — which is also why a monitor that
    /// reports no vacuous passes reports none here either.
    ///
    /// This is why the window is not zero for a multi-cycle antecedent even when
    /// no two of its matches can end in one cycle: `(a ##1 b) |-> c` needs no
    /// count for its verdicts and does need one for this.
    covers_trigger: bool,
    /// Which valuation of `cond` a backend must supply: the sampled one for
    /// `sync_accept_on`/`sync_reject_on`, the live one for the plain forms
    /// (LRM 16.15, exactly as for `reset`).
    ///
    /// Live means *every* time step, not merely the ones carrying a clocking
    /// event, so a backend stepped only on the clock edge has to latch it: set a
    /// bit whenever the condition holds at any time slot, consume and clear it
    /// at the next edge. Reading the condition at the edge alone silently
    /// collapses the plain forms onto the `sync_` ones, because a pulse landing
    /// strictly between two ticks is the only thing that separates them.
    ///
    /// Metadata. This runtime reads every root out of one array it is handed,
    /// so nothing here can enforce it; it is a contract on whoever evaluates
    /// the roots, and the only reason it rides on the monitor is that a backend
    /// has no other way to learn it.
    sampled: bool,

    pub const Effect = enum(u8) {
        /// The attempt is discharged: it holds, and a `cover` counts it.
        accept,
        /// The attempt fails.
        reject,

        /// The `Verdict` firing hands the attempt. An abort always resolves,
        /// so this is never `.pending`.
        pub fn verdict(effect: Effect) Verdict {
            return switch (effect) {
                .accept => .holds,
                .reject => .fails,
            };
        }
    };

    /// How much of an attempt firing resolves, and — because a `.counted`
    /// monitor's attempts share their state — how much of one it *can* resolve.
    ///
    /// Being armed is a property of the state, not of the attempt's history: an
    /// abort is armed when the subtree it guards could hold a thread this cycle,
    /// which is `⋃ follow` of where the attempt is. In `.counted` mode where the
    /// attempt is *is* one determinised state, so every attempt counted there is
    /// armed or none is, and retiring the whole count is exact rather than an
    /// approximation of retiring "the" attempt. That is what `step` does.
    ///
    /// `.branch` has no such reading and is refused at lowering time for a
    /// counted obligation. Killing part of an attempt means clearing states out
    /// of its vector, and a counted attempt has no vector to clear: its single
    /// state stands for a whole subset of the operand's, and subtracting a mask
    /// from it would have to split that subset into a state that does not exist.
    pub const Scope = enum(u8) {
        /// Firing resolves the whole attempt. An accept always does — a
        /// disjunct holding makes the disjunction hold — and so does a reject
        /// with nothing but conjunctions and shifts above it.
        attempt,
        /// Firing only kills the threads in the mask, leaving the rest of the
        /// attempt to run on. A reject under a disjunction: that disjunct is
        /// dead, the other may still hold. `.ring` mode only.
        branch,
    };
};

pub fn deinit(monitor: *Monitor, gpa: Allocator) void {
    monitor.alphabet.deinit(gpa);
    if (monitor.trigger) |*trigger| trigger.deinit(gpa);
    monitor.obligation.deinit(gpa);
    gpa.free(monitor.aborts);
    gpa.free(monitor.abort_masks);
    gpa.free(monitor.locals);
    gpa.free(monitor.assign_slot);
    gpa.free(monitor.assign_handle);
    gpa.free(monitor.root_local);
}

/// The obligation states the abort at `index` guards.
pub fn maskOf(monitor: *const Monitor, index: usize) []const Word {
    const words = monitor.obligation.words;
    return monitor.abort_masks[index * words ..][0..words];
}

/// Words in one trigger state's attempt-id set. Zero when `trigger_window` is,
/// which is what makes the whole mechanism cost a backend nothing there.
pub fn idWords(monitor: *const Monitor) u32 {
    return Nfa.wordsFor(monitor.trigger_window);
}

/// What one cycle did. A `cover` counts `matches`; an `assert` reports
/// `failures`.
pub const Report = struct {
    /// Obligation attempts that ran out of live states, or in `.counted` mode
    /// reached a `dead` state.
    ///
    /// Wide because a counted cycle can resolve many attempts at once: one
    /// `dead` state may hold the count of every attempt that has just failed
    /// together. See the file comment on why the counters are not saturated.
    failures: u64 = 0,
    /// Obligation attempts discharged by a completed match.
    matches: u64 = 0,
    /// The reset condition held, so everything in flight was abandoned.
    reset: bool = false,
};

/// A reference run of the protocol in the file comment. This is the executable
/// form of the contract a backend has to reproduce, not something a backend
/// would call.
/// Whether the ANTECEDENT assigns a local variable, which is what decides whether
/// the trigger needs valuations of its own. Reading one there needs nothing extra:
/// the values it reads are the attempt's.
fn triggerAssigns(monitor: *const Monitor) bool {
    const trigger = &(monitor.trigger orelse return false);
    if (trigger.program.len == 0) return false;
    for (0..trigger.state_count) |position| {
        if (trigger.programOf(@intCast(position)).len != 0) return true;
    }
    return false;
}

/// One LRM 16.10 valuation group: the states reachable with one valuation of the
/// local variable slots, and that valuation.
pub const Group = struct {
    /// Slot in `State.pending` this group's attempt occupies.
    owner: u32,
    /// `obligation.words` (or `trigger.words`) of state mask, owned.
    mask: []Word,
    /// `local_count` values, owned.
    values: []u64,

    fn deinit(group: *Group, gpa: Allocator) void {
        gpa.free(group.mask);
        gpa.free(group.values);
    }
};

/// The two questions libsva cannot answer about a value, as the header's
/// `sva_locals` asks them.
pub const Locals = struct {
    assign: *const fn (?*anyopaque, u32, [*]const u64) callconv(.c) u64,
    root: *const fn (?*anyopaque, u32, [*]const u64) callconv(.c) bool,
    user: ?*anyopaque,
};

pub const State = struct {
    monitor: *const Monitor,
    /// The allocator the LRM 16.10 group pools grow with. Held rather than passed
    /// because a group is created and dropped inside `step`, which the C ABI calls
    /// with no allocator in sight.
    gpa: Allocator,
    /// A bit per entry of `alphabet.roots`.
    root_bits: []Word,
    /// Trigger and obligation states whose guard holds this cycle.
    trigger_satisfied: []Word,
    obligation_satisfied: []Word,
    /// Every trigger state a thread sits in, over every attempt: the existential
    /// half of the antecedent, and all of it a zero `trigger_window` needs.
    trigger_current: []Word,
    /// Which attempts have a thread in each trigger state: `idWords()` words per
    /// trigger state, holding a bit per live attempt id. What `trigger_current`
    /// cannot say and a count of matches needs — see `# Counting the trigger`.
    /// Empty, along with its scratch row and `id_union`, when `trigger_window`
    /// is zero.
    trigger_ids: []Word,
    next_ids: []Word,
    /// One id set of scratch, for the union a count is taken over.
    id_union: []Word,
    /// The id this cycle's antecedent attempt is seeded with. Rotates through
    /// `trigger_window` with the cycle, not with the seedings: the argument that
    /// an id is free again is about how long a thread can live.
    id_now: u32,
    /// `depth` obligation state vectors, each either free or holding one
    /// pending attempt. The order is deliberately meaningless: what makes a
    /// fixed pool work is that `depth` is the obligation's longest match, so an
    /// attempt is always resolved before `depth` more have started and a free
    /// slot always exists when one is needed.
    pending: []Word,
    /// How many antecedent attempts each slot stands for, and zero for a free
    /// one. A slot holding one behaves exactly as a `bool` would; holding three
    /// it reports three verdicts, because three antecedent matches ended in the
    /// cycle that seeded it and each owes one.
    attempts: []u64,
    /// The LRM 16.10 valuation groups, and the values a threaded run computes on.
    /// All empty for a monitor without local variable slots.
    ///
    /// A group is a valuation plus the mask of states holding it, and belongs to
    /// one pending attempt: `group_owner[i]` is its slot in `pending`. The frames
    /// are `local_count` u64s apiece, which is a stand-in for whatever a backend's
    /// values really are -- enough to check the discipline, not a value model
    /// anyone has to adopt (see `sva_locals` in the header).
    groups: std.ArrayListUnmanaged(Group),
    /// The trigger's own groups, when the ANTECEDENT assigns: the values live there
    /// while the antecedent matches, and the group whose match ends hands them to
    /// the attempt it seeds. Empty otherwise, including when the antecedent only
    /// reads a slot.
    trigger_groups: std.ArrayListUnmanaged(Group),
    /// The valuation that flowed out of the antecedent match, held for the cycle
    /// `|=>` seeds in.
    fired_values: []u64,
    /// What the caller supplies: the value of an assignment site, and whether a
    /// local-reading root holds, both against a given frame. Null until
    /// `sva_run_bind`, and stepping a threaded monitor without them is a caller
    /// bug.
    locals: ?Locals,

    /// How many antecedent matches ended in the previous cycle, for `|=>`.
    fired: u64,
    /// `aborts.len * depth` flags: whether an abort's subtree could hold a
    /// thread of that attempt this cycle. Set before the advance from the
    /// reachable set rather than the live one — see `step`. Empty when there are
    /// no aborts.
    abort_entered: []bool,
    /// One obligation mask of scratch for computing that reachable set.
    abort_reach: []Word,
    scratch: []Word,
    /// `.counted` mode: how many attempts sit in each obligation state, and one
    /// row of scratch to advance them into. Both empty in `.ring` mode, where
    /// `pending`/`attempts` are the live storage instead; exactly one of the two
    /// pairs is ever allocated.
    counts: []u64,
    next_counts: []u64,

    pub fn init(gpa: Allocator, monitor: *const Monitor) Allocator.Error!State {
        const obligation_words = monitor.obligation.words;
        const trigger_words = if (monitor.trigger) |trigger| trigger.words else 0;

        const root_bits = try gpa.alloc(Word, monitor.alphabet.words());
        errdefer gpa.free(root_bits);
        const trigger_satisfied = try gpa.alloc(Word, trigger_words);
        errdefer gpa.free(trigger_satisfied);
        const obligation_satisfied = try gpa.alloc(Word, obligation_words);
        errdefer gpa.free(obligation_satisfied);
        const trigger_current = try gpa.alloc(Word, trigger_words);
        errdefer gpa.free(trigger_current);
        const pending = try gpa.alloc(Word, @as(usize, monitor.depth) * obligation_words);
        errdefer gpa.free(pending);
        const attempts = try gpa.alloc(u64, monitor.depth);
        errdefer gpa.free(attempts);
        // One id set per trigger state, and two more rows to work in. All three
        // are empty when nothing has to be told apart, which is what "zero
        // window, zero cost" means for a backend allocating alongside this one.
        const id_words = monitor.idWords();
        const trigger_states = if (monitor.trigger) |trigger| trigger.state_count else 0;
        const trigger_ids = try gpa.alloc(Word, @as(usize, trigger_states) * id_words);
        errdefer gpa.free(trigger_ids);
        const next_ids = try gpa.alloc(Word, trigger_ids.len);
        errdefer gpa.free(next_ids);
        const id_union = try gpa.alloc(Word, id_words);
        errdefer gpa.free(id_union);
        const abort_entered = try gpa.alloc(bool, monitor.aborts.len * monitor.depth);
        errdefer gpa.free(abort_entered);
        const abort_reach = try gpa.alloc(Word, obligation_words);
        errdefer gpa.free(abort_reach);
        const scratch = try gpa.alloc(Word, @max(trigger_words, obligation_words));
        errdefer gpa.free(scratch);
        // `depth` is zero in `.counted` mode and the counters are what replace
        // the ring, so exactly one of the two allocations is non-empty.
        const counted = monitor.mode == .counted;
        const slots: usize = if (counted) monitor.obligation.state_count else 0;
        const counts = try gpa.alloc(u64, slots);
        errdefer gpa.free(counts);
        const next_counts = try gpa.alloc(u64, slots);

        // The valuation frames a threaded run carries. `fired_values` is one frame
        // wide: at most one antecedent match ends per cycle in every shape that
        // reaches here, which is what refusing a non-zero attempt window with local
        // variables guarantees.
        const fired_values = try gpa.alloc(u64, monitor.locals.len);
        errdefer gpa.free(fired_values);
        @memset(fired_values, 0);

        var state: State = .{
            .monitor = monitor,
            .gpa = gpa,
            .groups = .empty,
            .trigger_groups = .empty,
            .fired_values = fired_values,
            .locals = null,
            .root_bits = root_bits,
            .trigger_satisfied = trigger_satisfied,
            .obligation_satisfied = obligation_satisfied,
            .trigger_current = trigger_current,
            .pending = pending,
            .attempts = attempts,
            .trigger_ids = trigger_ids,
            .next_ids = next_ids,
            .id_union = id_union,
            .id_now = 0,
            .fired = 0,
            .abort_entered = abort_entered,
            .abort_reach = abort_reach,
            .scratch = scratch,
            .counts = counts,
            .next_counts = next_counts,
        };
        state.clear();
        return state;
    }

    /// Every group of both pools, freed. `clear` and a reset use it; the lists keep
    /// their capacity, since a reset is followed by more attempts.
    fn dropGroups(state: *State) void {
        for (state.groups.items) |*group| group.deinit(state.gpa);
        state.groups.clearRetainingCapacity();
        for (state.trigger_groups.items) |*group| group.deinit(state.gpa);
        state.trigger_groups.clearRetainingCapacity();
    }

    pub fn deinit(state: *State, gpa: Allocator) void {
        gpa.free(state.root_bits);
        gpa.free(state.trigger_satisfied);
        gpa.free(state.obligation_satisfied);
        gpa.free(state.trigger_current);
        gpa.free(state.pending);
        gpa.free(state.attempts);
        gpa.free(state.trigger_ids);
        gpa.free(state.next_ids);
        gpa.free(state.id_union);
        gpa.free(state.abort_entered);
        gpa.free(state.abort_reach);
        gpa.free(state.scratch);
        gpa.free(state.counts);
        gpa.free(state.next_counts);
        gpa.free(state.fired_values);
        for (state.groups.items) |*group| group.deinit(gpa);
        state.groups.deinit(gpa);
        for (state.trigger_groups.items) |*group| group.deinit(gpa);
        state.trigger_groups.deinit(gpa);
    }

    /// Abandon everything in flight.
    pub fn clear(state: *State) void {
        // A reset abandons every attempt, and a group belongs to one, so the pools go
        // with them. The allocator is the list's own; only the frames are freed here.
        state.dropGroups();
        @memset(state.trigger_current, 0);
        @memset(state.pending, 0);
        @memset(state.attempts, 0);
        @memset(state.trigger_ids, 0);
        @memset(state.counts, 0);
        @memset(state.abort_entered, false);
        state.fired = 0;
        // The rotation is not reset with it. Reclaiming an id needs the attempt
        // that held it to be gone, and a reset has just made that true of every
        // attempt there was, so where the rotation stands is immaterial — and
        // leaving it where it is keeps the id a cycle's function.
    }

    /// The state vector of the attempt living in `slot`.
    fn slotOf(state: *State, slot: u32) []Word {
        const words = state.monitor.obligation.words;
        return state.pending[@as(usize, slot) * words ..][0..words];
    }

    /// A slot no attempt is living in. Occupancy is the count being non-zero: a
    /// second array saying whether a slot is live could disagree with the count
    /// in it, and one of the two would be the bug.
    fn freeSlot(state: *const State) ?u32 {
        for (state.attempts, 0..) |count, slot| {
            if (count == 0) return @intCast(slot);
        }
        return null;
    }

    /// The attempt ids sitting in trigger state `index`, in `rows` — either
    /// `trigger_ids` or the scratch row set it is advanced into.
    fn idsIn(state: *const State, rows: []Word, index: u32) []Word {
        const words = state.monitor.idWords();
        return rows[@as(usize, index) * words ..][0..words];
    }

    /// Move every attempt one cycle along the trigger, by union.
    ///
    /// Nothing to do when nothing is tracked. The transfer walks the same
    /// `follow ∩ satisfied` relation `Nfa.advance` does, one union per enabled
    /// edge, which is what keeps the two representations from coming apart: a
    /// state carries an attempt's id exactly when it holds a thread of it, so
    /// `trigger_current` is `{ s : ids[s] != 0 }` in every cycle.
    ///
    /// Union, not addition. Two threads of one attempt that reconverge here are
    /// one attempt, and an attempt is what a match belongs to — see
    /// `# Counting the trigger`.
    fn advanceIds(state: *State, trigger: *const Nfa) void {
        if (state.monitor.trigger_window == 0) return;

        @memset(state.next_ids, 0);
        var live: Nfa.BitIterator = .init(state.trigger_current);
        while (live.next()) |from| {
            const ids = state.idsIn(state.trigger_ids, from);
            for (trigger.followOf(from), state.trigger_satisfied, 0..) |edges, holding, word| {
                var enabled = edges & holding;
                while (enabled != 0) {
                    const target = word * Nfa.word_bits + @ctz(enabled);
                    enabled &= enabled - 1;
                    Nfa.bitUnion(state.idsIn(state.next_ids, @intCast(target)), ids);
                }
            }
        }
        @memcpy(state.trigger_ids, state.next_ids);
    }

    /// Start this cycle's antecedent attempt under the rotating id.
    fn seedId(state: *State, trigger: *const Nfa) void {
        const window = state.monitor.trigger_window;
        if (window == 0) return;

        const id = state.id_now;
        state.id_now = if (id + 1 == window) 0 else id + 1;

        // Nothing retires the id first. `window` is the trigger's longest match
        // and every trigger state is co-reachable to an exit, so a thread of the
        // attempt that held this id `window` cycles ago would witness a match
        // longer than the longest one there is. It is gone of its own accord,
        // which this asserts rather than paying a pass over the rows to ensure.
        if (std.debug.runtime_safety) {
            for (0..trigger.state_count) |index| {
                const ids = state.idsIn(state.trigger_ids, @intCast(index));
                std.debug.assert(!Nfa.bitIsSet(ids, id));
            }
        }

        for (trigger.first, state.trigger_satisfied, 0..) |entry, holding, word| {
            var entering = entry & holding;
            while (entering != 0) {
                const target = word * Nfa.word_bits + @ctz(entering);
                entering &= entering - 1;
                Nfa.bitSet(state.idsIn(state.trigger_ids, @intCast(target)), id);
            }
        }
    }

    /// How many attempts hold a thread in one of `states`.
    fn idCount(state: *State, states: []const Word) u64 {
        @memset(state.id_union, 0);
        var live: Nfa.BitIterator = .init(states);
        while (live.next()) |index| {
            Nfa.bitUnion(state.id_union, state.idsIn(state.trigger_ids, index));
        }
        return Nfa.bitCount(state.id_union);
    }

    /// How many antecedent matches end in this cycle — step 4's answer, and how
    /// many attempts the obligation it seeds stands for.
    fn matchCount(state: *State, trigger: *const Nfa) u64 {
        if (state.monitor.trigger_window == 0) {
            return @intFromBool(trigger.matched(state.trigger_current));
        }
        return state.idCount(trigger.last);
    }

    /// How many attempts have a live antecedent thread, whether their match ends
    /// now or they are still matching: what an abort covering the trigger
    /// resolves, one verdict each.
    ///
    /// The matches alone when nothing is tracked, and that is exact rather than a
    /// fallback: a window is zero only when every match spans the same number of
    /// cycles, and an abort covering the trigger forces one whenever that number
    /// exceeds one, so an untracked trigger under an abort holds no thread that
    /// outlives the cycle it was seeded in.
    fn inFlightCount(state: *State, fires: u64) u64 {
        if (state.monitor.trigger_window == 0) return fires;
        return state.idCount(state.trigger_current);
    }

    /// Advance one cycle, given a bit per `alphabet.roots`.
    ///
    /// This is the protocol proper. Producing the root bits is the caller's
    /// half of the contract — it is the only part that needs to know what an
    /// atom means — so a backend that has evaluated them itself steps from
    /// here and never touches an `Ir`.
    // -- The threaded discipline (LRM 16.10) ---------------------------------
    //
    // The header's RUNNING section is the specification; this is it executed. A
    // pending attempt is a list of GROUPS -- a valuation plus the mask of states
    // holding it -- and the mask arithmetic inside a group is the ring's, so what
    // the groups replace is the per-attempt advance and nothing else.

    /// The roots as this group sees them: the ones that read a slot are asked of the
    /// caller against this group's values, the rest are the cycle's.
    fn groupRoots(state: *State, group: *const Group, into: []Word) void {
        const monitor = state.monitor;
        @memcpy(into, state.root_bits);
        const locals = state.locals.?;
        for (0..monitor.alphabet.roots.len) |index| {
            if (!Nfa.bitIsSet(monitor.root_local, @intCast(index))) continue;
            const holds = locals.root(locals.user, @intCast(index), group.values.ptr);
            if (holds) Nfa.bitSet(into, @intCast(index)) else Nfa.bitClear(into, @intCast(index));
        }
    }

    /// A new group, owned by `owner`, holding `mask` and a copy of `values`.
    fn addGroup(
        state: *State,
        list: *std.ArrayListUnmanaged(Group),
        owner: u32,
        words: u32,
        mask: []const Word,
        values: []const u64,
    ) Allocator.Error!void {
        const own_mask = try state.gpa.dupe(Word, mask[0..words]);
        errdefer state.gpa.free(own_mask);
        const own_values = try state.gpa.dupe(u64, values);
        try list.append(state.gpa, .{ .owner = owner, .mask = own_mask, .values = own_values });
    }

    /// Run one state's assignment program over `values`, in order. LRM 16.10
    /// performs the items in the order they appear, each against the values as they
    /// stand, so a compound assignment reads the slot it writes.
    fn runProgram(state: *State, nfa: *const Nfa, position: u32, values: []u64) void {
        const monitor = state.monitor;
        const locals = state.locals.?;
        for (nfa.programOf(position)) |site| {
            values[monitor.assign_slot[site]] = locals.assign(locals.user, site, values.ptr);
        }
    }

    /// Split `mask` on assignment: every state with a program leaves the group it
    /// arrived in and becomes one of its own, carrying what its program produced.
    /// The states that assign nothing stay behind, which is what `mask` is left as.
    fn splitOnAssignment(
        state: *State,
        list: *std.ArrayListUnmanaged(Group),
        nfa: *const Nfa,
        owner: u32,
        mask: []Word,
        values: []const u64,
    ) Allocator.Error!void {
        var bits: Nfa.BitIterator = .init(mask);
        var positions: [64]u32 = undefined;
        var count: usize = 0;
        while (bits.next()) |position| {
            if (nfa.programOf(position).len == 0) continue;
            if (count == positions.len) break;
            positions[count] = position;
            count += 1;
        }
        for (positions[0..count]) |position| {
            const scratch = try state.gpa.alloc(Word, nfa.words);
            defer state.gpa.free(scratch);
            @memset(scratch, 0);
            Nfa.bitSet(scratch, position);

            const own = try state.gpa.dupe(u64, values);
            defer state.gpa.free(own);
            state.runProgram(nfa, position, own);
            try state.addGroup(list, owner, nfa.words, scratch, own);
            Nfa.bitClear(mask, position);
        }
    }

    /// Drop the groups that died, and merge the ones that agree.
    ///
    /// Two groups of one attempt with equal values have the same future, so their
    /// masks join. That is not an optimisation: without it an `or` whose branches
    /// both match under a repetition adds a group per cycle for ever. Groups of
    /// different attempts never merge, however equal their values -- their verdicts
    /// are owed separately.
    fn compactGroups(state: *State, list: *std.ArrayListUnmanaged(Group), owned: bool) void {
        var index: usize = 0;
        while (index < list.items.len) {
            const group = &list.items[index];
            if (Nfa.bitAny(group.mask)) {
                var probe: usize = 0;
                var merged = false;
                while (probe < index) : (probe += 1) {
                    const other = &list.items[probe];
                    if (owned and other.owner != group.owner) continue;
                    if (!std.mem.eql(u64, other.values, group.values)) continue;
                    for (other.mask, group.mask) |*into, bit| into.* |= bit;
                    merged = true;
                    break;
                }
                if (!merged) {
                    index += 1;
                    continue;
                }
            }
            var dead = list.swapRemove(index);
            dead.deinit(state.gpa);
        }
    }

    /// One cycle of the threaded discipline. The header's RUNNING section documents
    /// the order and the reasons; the deviations from the ring are steps 3, 6, 7 and
    /// 8, which is exactly what the groups replace.
    fn stepThreaded(state: *State, root_bits: []const Word) Allocator.Error!Report {
        const monitor = state.monitor;
        @memcpy(state.root_bits, root_bits);

        if (Nfa.bitIsSet(root_bits, monitor.reset)) {
            state.clear();
            return .{ .reset = true };
        }
        // An unbounded obligation has no ring for its attempts to live in, which is
        // the one shape a backend refuses rather than tracks; a run of it would need
        // dynamic attempts too.
        std.debug.assert(monitor.depth > 0);

        const obligation = &monitor.obligation;
        var report: Report = .{};

        // The antecedent, when it assigns: its own groups carry the values, and the
        // group whose match ends hands them to the attempt it seeds (LRM 16.10,
        // F.5.5). Otherwise the plain existential vector is all it needs.
        var fires: u64 = 0;
        if (monitor.trigger) |*trigger| {
            if (state.trigger_groups.items.len != 0 or triggerAssigns(monitor)) {
                fires = try state.stepTriggerGroups(trigger);
            } else {
                trigger.satisfy(&monitor.alphabet, root_bits, state.trigger_satisfied);
                const next = state.scratch[0..trigger.words];
                trigger.advance(state.trigger_current, state.trigger_satisfied, next);
                @memcpy(state.trigger_current, next);
                trigger.seed(state.trigger_satisfied, state.trigger_current);
                fires = if (trigger.matched(state.trigger_current)) 1 else 0;
            }
        }

        // Advance every group: its own roots, its own satisfied mask, the same
        // constant follow rows the ring walks, then the split on assignment.
        const roots = try state.gpa.alloc(Word, monitor.alphabet.words());
        defer state.gpa.free(roots);
        const next = try state.gpa.alloc(Word, obligation.words);
        defer state.gpa.free(next);
        const satisfied = try state.gpa.alloc(Word, obligation.words);
        defer state.gpa.free(satisfied);

        const live = state.groups.items.len;
        for (0..live) |index| {
            const group = &state.groups.items[index];
            state.groupRoots(group, roots);
            obligation.satisfy(&monitor.alphabet, roots, satisfied);
            obligation.advance(group.mask, satisfied, next);
            @memcpy(group.mask, next);
            // The values are copied out first: the split may reallocate the list and
            // move the group this pointer names.
            const values = try state.gpa.dupe(u64, group.values);
            defer state.gpa.free(values);
            const owner = group.owner;
            const mask = try state.gpa.dupe(Word, group.mask);
            defer state.gpa.free(mask);
            try state.splitOnAssignment(&state.groups, obligation, owner, mask, values);
            @memcpy(state.groups.items[index].mask, mask);
        }

        // Seed. `|->` seeds in the cycle the antecedent match ends, `|=>` the cycle
        // after, and a bare sequence every cycle.
        const seeding: u64 = if (monitor.trigger == null)
            1
        else if (monitor.overlap)
            fires
        else
            state.fired;
        if (monitor.trigger != null) state.fired = fires;

        if (seeding != 0) {
            const slot = state.freeSlot() orelse @panic("no free attempt slot for a threaded seed");
            state.attempts[slot] = seeding;
            const values = try state.gpa.alloc(u64, monitor.locals.len);
            defer state.gpa.free(values);
            if (state.trigger_groups.items.len != 0 or triggerAssigns(monitor)) {
                @memcpy(values, state.fired_values);
            } else {
                for (monitor.locals, 0..) |local, slot_index| {
                    values[slot_index] = if (local.init == Ir.local_no_init)
                        0
                    else
                        state.locals.?.assign(state.locals.?.user, local.init, values.ptr);
                }
            }
            @memset(satisfied, 0);
            state.groupRootsFor(values, roots);
            obligation.satisfy(&monitor.alphabet, roots, satisfied);
            @memset(next, 0);
            obligation.seed(satisfied, next);
            try state.addGroup(&state.groups, @intCast(slot), obligation.words, next, values);
            const seeded = state.groups.items.len - 1;
            const mask = try state.gpa.dupe(Word, state.groups.items[seeded].mask);
            defer state.gpa.free(mask);
            try state.splitOnAssignment(&state.groups, obligation, @intCast(slot), mask, values);
            @memcpy(state.groups.items[seeded].mask, mask);
        }

        // Verdicts, per attempt: the union of its groups' masks is its state vector,
        // and the lattice is the ring's.
        for (0..monitor.depth) |slot| {
            if (state.attempts[slot] == 0) continue;
            const vector = state.pending[slot * obligation.words ..][0..obligation.words];
            @memset(vector, 0);
            for (state.groups.items) |group| {
                if (group.owner == slot) Nfa.bitUnion(vector, group.mask);
            }
            const held = state.attempts[slot];
            if (obligation.matched(vector)) {
                report.matches += held;
                state.attempts[slot] = 0;
            } else if (!Nfa.bitAny(vector)) {
                report.failures += held;
                state.attempts[slot] = 0;
            }
        }

        // A resolved attempt's groups go with it, then what is left is compacted and
        // merged.
        for (state.groups.items) |*group| {
            if (state.attempts[group.owner] == 0) @memset(group.mask, 0);
        }
        state.compactGroups(&state.groups, true);
        return report;
    }

    /// The roots as a valuation sees them, for a frame that is not a group's yet.
    fn groupRootsFor(state: *State, values: []const u64, into: []Word) void {
        const monitor = state.monitor;
        @memcpy(into, state.root_bits);
        const locals = state.locals.?;
        for (0..monitor.alphabet.roots.len) |index| {
            if (!Nfa.bitIsSet(monitor.root_local, @intCast(index))) continue;
            const holds = locals.root(locals.user, @intCast(index), values.ptr);
            if (holds) Nfa.bitSet(into, @intCast(index)) else Nfa.bitClear(into, @intCast(index));
        }
    }

    /// The trigger's own groups: advance, seed one per cycle, and report whether a
    /// match ended -- latching its values for the attempt that match seeds.
    fn stepTriggerGroups(state: *State, trigger: *const Nfa) Allocator.Error!u64 {
        const monitor = state.monitor;
        const roots = try state.gpa.alloc(Word, monitor.alphabet.words());
        defer state.gpa.free(roots);
        const satisfied = try state.gpa.alloc(Word, trigger.words);
        defer state.gpa.free(satisfied);
        const next = try state.gpa.alloc(Word, trigger.words);
        defer state.gpa.free(next);

        const live = state.trigger_groups.items.len;
        for (0..live) |index| {
            const group = &state.trigger_groups.items[index];
            state.groupRoots(group, roots);
            trigger.satisfy(&monitor.alphabet, roots, satisfied);
            trigger.advance(group.mask, satisfied, next);
            @memcpy(group.mask, next);
            const values = try state.gpa.dupe(u64, group.values);
            defer state.gpa.free(values);
            const mask = try state.gpa.dupe(Word, group.mask);
            defer state.gpa.free(mask);
            try state.splitOnAssignment(&state.trigger_groups, trigger, 0, mask, values);
            @memcpy(state.trigger_groups.items[index].mask, mask);
        }

        // An attempt of the assertion starts every cycle, and 16.10 creates the local
        // variables when the attempt does -- so the antecedent's own thread runs the
        // declaration initializers.
        {
            const values = try state.gpa.alloc(u64, monitor.locals.len);
            defer state.gpa.free(values);
            for (monitor.locals, 0..) |local, slot_index| {
                values[slot_index] = if (local.init == Ir.local_no_init)
                    0
                else
                    state.locals.?.assign(state.locals.?.user, local.init, values.ptr);
            }
            state.groupRootsFor(values, roots);
            trigger.satisfy(&monitor.alphabet, roots, satisfied);
            @memset(next, 0);
            trigger.seed(satisfied, next);
            try state.addGroup(&state.trigger_groups, 0, trigger.words, next, values);
            const seeded = state.trigger_groups.items.len - 1;
            const mask = try state.gpa.dupe(Word, state.trigger_groups.items[seeded].mask);
            defer state.gpa.free(mask);
            try state.splitOnAssignment(&state.trigger_groups, trigger, 0, mask, values);
            @memcpy(state.trigger_groups.items[seeded].mask, mask);
        }

        // Whose match ends now, and with what values. At most one: an antecedent that
        // can match twice in one attempt is refused alongside local variables.
        var fires: u64 = 0;
        for (state.trigger_groups.items) |group| {
            if (!trigger.matched(group.mask)) continue;
            @memcpy(state.fired_values, group.values);
            fires = 1;
        }
        state.compactGroups(&state.trigger_groups, false);
        return fires;
    }

    pub fn step(state: *State, root_bits: []const Word) Report {
        if (state.monitor.mode == .threaded) {
            // Stepping a monitor with local variable slots needs the two questions
            // libsva cannot answer about a value; see `sva_run_bind`.
            std.debug.assert(state.locals != null);
            return state.stepThreaded(root_bits) catch @panic("out of memory stepping a threaded monitor");
        }
        const monitor = state.monitor;

        if (Nfa.bitIsSet(root_bits, monitor.reset)) {
            state.clear();
            return .{ .reset = true };
        }

        monitor.obligation.satisfy(&monitor.alphabet, root_bits, state.obligation_satisfied);

        var fires: u64 = 0;
        var in_flight: u64 = 0;
        if (monitor.trigger) |*trigger| {
            trigger.satisfy(&monitor.alphabet, root_bits, state.trigger_satisfied);
            // The ids ride the transitions the vector rides, so they move off
            // the same pre-advance set and have to move first.
            state.advanceIds(trigger);
            const next = state.scratch[0..trigger.words];
            trigger.advance(state.trigger_current, state.trigger_satisfied, next);
            @memcpy(state.trigger_current, next);
            trigger.seed(state.trigger_satisfied, state.trigger_current);
            state.seedId(trigger);
            fires = state.matchCount(trigger);
            in_flight = state.inFlightCount(fires);
        }

        // An abort above the implication covers the antecedent too, so the
        // attempts still matching one are resolved by it rather than left to
        // seed obligations. This is the cycle the pushed-down spelling cannot
        // see at all.
        //
        // Every attempt with a live antecedent thread is resolved, not only the
        // ones whose match ends now: this cycle is inside the evaluation of each
        // of them, which is what arms an abort. They are counted rather than
        // collapsed — `in_flight` of them, each owing its own verdict — and
        // their threads are cleared out of the trigger, because an antecedent
        // completing later would impose an obligation on an attempt this verdict
        // has already resolved. Clearing the whole vector is exactly clearing
        // each of them, an abort above the trigger covering every attempt of it.
        var trigger_verdict: ?Abort.Effect = null;
        if (in_flight != 0) {
            for (monitor.aborts) |abort| {
                if (!abort.covers_trigger) continue;
                if (!Nfa.bitIsSet(root_bits, abort.cond)) continue;
                // Last wins. Lowering appends outward, so the last entry is the
                // outermost abort, and an outer abort beats an inner verdict
                // landing in the same cycle.
                trigger_verdict = abort.effect;
            }
            if (trigger_verdict != null) {
                fires = 0;
                @memset(state.trigger_current, 0);
                @memset(state.trigger_ids, 0);
            }
        }

        var report: Report = .{};
        if (trigger_verdict) |effect| switch (effect) {
            .accept => report.matches += in_flight,
            .reject => report.failures += in_flight,
        };

        // The two disciplines part company here. Everything above is shared —
        // the reset, the satisfied masks, the trigger, and the verdict an abort
        // covering the trigger produces — and everything below is only about how
        // the pending obligation attempts are held.
        if (monitor.mode == .counted) return state.stepCounted(root_bits, fires, report);

        // Which aborts each attempt is inside for this cycle, guards aside.
        //
        // An abort covers every cycle its subtree could hold a thread in, which
        // is the reachable set rather than the live one. Both ends of that
        // matter and neither is the live set: the cycle a subtree is *entered*
        // in is one its guard may already fail in, and the cycle its last thread
        // dies in is one it was still being evaluated in. LRM 16.12.7 arms an
        // abort at any point during the evaluation attempt, and both are such a
        // point — `nexttime[2] (accept_on(b) p)` over a `p` that fails
        // immediately is the case where they coincide and the live set is empty
        // either side of the advance.
        const depth = monitor.depth;
        @memset(state.abort_entered, false);
        if (monitor.aborts.len != 0) {
            for (state.attempts, 0..) |count, slot| {
                if (count == 0) continue;
                @memset(state.abort_reach, 0);
                var live: Nfa.BitIterator = .init(state.slotOf(@intCast(slot)));
                while (live.next()) |from| {
                    Nfa.bitUnion(state.abort_reach, monitor.obligation.followOf(from));
                }
                for (monitor.aborts, 0..) |_, index| {
                    state.abort_entered[index * depth + slot] =
                        Nfa.bitOverlaps(state.abort_reach, monitor.maskOf(index));
                }
            }
        }

        const next = state.scratch[0..monitor.obligation.words];
        for (state.attempts, 0..) |count, slot| {
            if (count == 0) continue;
            const current = state.slotOf(@intCast(slot));
            monitor.obligation.advance(current, state.obligation_satisfied, next);
            @memcpy(current, next);
        }

        // How many attempts a seeding stands for. One per cycle without a
        // trigger; otherwise as many antecedent matches as ended in the cycle
        // the obligation starts from, which is this one under `|->` and the
        // previous one under `|=>`. Zero is no seeding at all.
        const born: u64 = if (monitor.trigger == null)
            1
        else if (monitor.overlap)
            fires
        else
            state.fired;
        if (born != 0) {
            // A slot is always free here: an attempt is resolved by the time it
            // is `depth - 1` cycles old, so at most `depth - 1` of them survived
            // into this cycle. Several matches ending together still take one
            // slot — they impose the same obligation from the same cycle, so one
            // state vector answers for all of them and the count says how many.
            const slot = state.freeSlot() orelse unreachable;
            const current = state.slotOf(slot);
            @memset(current, 0);
            monitor.obligation.seed(state.obligation_satisfied, current);
            state.attempts[slot] = born;
            // A fresh attempt reaches the entry states, guards aside.
            for (monitor.aborts, 0..) |_, index| {
                if (Nfa.bitOverlaps(monitor.obligation.first, monitor.maskOf(index))) {
                    state.abort_entered[index * depth + slot] = true;
                }
            }
        }
        state.fired = fires;

        // Aborts resolve an attempt before it would have resolved itself. Each
        // is checked against every attempt separately, because being armed is a
        // per-attempt question: the subtree the abort guards has to still be
        // under evaluation *in this attempt* for aborting it to mean anything.
        //
        // This runs after seeding for the same reason step 7 does — an abort
        // whose condition holds in the cycle an attempt starts applies to that
        // attempt — and before verdicts, because resolving early is the whole
        // point.
        //
        // Outward, so the scan runs backwards: lowering appends aborts from the
        // inside out, and when two of them resolve one attempt in one cycle the
        // OUTER one wins. `accept_on(a) (reject_on(b) p)` with both holding is
        // TRUE, not FALSE — the reject settles the inner evaluation as false,
        // and the accept's condition is then true during an evaluation that is
        // still under way, which LRM 16.12.14 reads as accepting. Step 5 takes
        // the same rule the same way round.
        var index = monitor.aborts.len;
        while (index > 0) {
            index -= 1;
            const abort = monitor.aborts[index];
            if (!Nfa.bitIsSet(root_bits, abort.cond)) continue;
            const mask = monitor.maskOf(index);
            for (state.attempts, 0..) |count, slot| {
                if (count == 0) continue;
                // Armed if the abort's subtree could hold a thread of this
                // attempt this cycle. That is the reachable set, computed above:
                // the live set would miss both the cycle the subtree is entered
                // in and the cycle its last thread dies in.
                if (!state.abort_entered[index * depth + slot]) continue;
                const current = state.slotOf(@intCast(slot));
                switch (abort.scope) {
                    .attempt => {
                        // One verdict per attempt the slot stands for. The abort
                        // is armed by the slot's state vector, which every one of
                        // them shares, so it fires for all or for none.
                        switch (abort.effect.verdict()) {
                            .holds => report.matches += count,
                            .fails => report.failures += count,
                            .pending => unreachable, // an abort always resolves
                        }
                        @memset(current, 0);
                        state.attempts[slot] = 0;
                    },
                    // The guarded threads die; whether that kills the attempt
                    // is the ordinary verdict below deciding it has no live
                    // states left.
                    .branch => Nfa.bitSubtract(current, mask),
                }
            }
        }

        for (state.attempts, 0..) |count, slot| {
            if (count == 0) continue;
            const current = state.slotOf(@intCast(slot));
            if (monitor.obligation.matched(current)) {
                report.matches += count;
                state.attempts[slot] = 0;
            } else if (!Nfa.bitAny(current)) {
                report.failures += count;
                state.attempts[slot] = 0;
            }
        }
        return report;
    }

    /// `step`'s second half for `.counted` mode: a `u64` per obligation state
    /// where the ring has a vector per attempt.
    ///
    /// The order is `step`'s, with one deliberate difference. The ring applies
    /// aborts *after* the advance, which it can afford because an
    /// `.attempt`-scoped abort discards the post-advance vector anyway and the
    /// arming flag was read off the pre-advance one. Here they have to be applied
    /// before, and not for tidiness: two states can share a successor —
    /// everything that has just died shares the trap — and once two counts have
    /// merged there is no telling which of them was armed. Retiring before the
    /// merge keeps it exact, and lands the same verdict in the same cycle as the
    /// ring, because either way a retired attempt is never looked at again.
    fn stepCounted(
        state: *State,
        root_bits: []const Word,
        fires: u64,
        partial: Report,
    ) Report {
        const monitor = state.monitor;
        const obligation = &monitor.obligation;
        const satisfied = state.obligation_satisfied;

        // Whatever step 5 already resolved on the trigger, which is the same
        // either side of the split.
        var report = partial;

        // Aborts, outermost first, against the counts as they stand. Being armed
        // is a question about the *state* — could the guarded subtree hold a
        // thread of an attempt sitting here this cycle — so one answer serves
        // every attempt counted there, and retiring the whole count is exact.
        // The reachable set is the same `⋃ follow` the ring computes, over a live
        // set that happens to be a single state.
        for (state.counts, 0..) |*count, index| {
            if (count.* == 0) continue;
            const from: u32 = @intCast(index);
            var at = monitor.aborts.len;
            while (at > 0) {
                at -= 1;
                const abort = monitor.aborts[at];
                if (!Nfa.bitIsSet(root_bits, abort.cond)) continue;
                if (!Nfa.bitOverlaps(obligation.followOf(from), monitor.maskOf(at))) continue;
                // `.branch` is refused for a counted obligation: there is no
                // vector to subtract a mask from. See `Abort.Scope`.
                std.debug.assert(abort.scope == .attempt);
                switch (abort.effect.verdict()) {
                    .holds => report.matches += count.*,
                    .fails => report.failures += count.*,
                    .pending => unreachable,
                }
                count.* = 0;
                break;
            }
        }

        // The advance. One successor per state and no more — see `chosenFollow`
        // for why several can be enabled and why the lowest is the right one.
        @memset(state.next_counts, 0);
        for (state.counts, 0..) |count, index| {
            if (count == 0) continue;
            const target = obligation.chosenFollow(@intCast(index), satisfied) orelse
                unreachable; // total: some cell holds this cycle's valuation
            state.next_counts[target] += count;
        }

        const born: u64 = if (monitor.trigger == null)
            1
        else if (monitor.overlap)
            fires
        else
            state.fired;
        if (born != 0) {
            const entry = obligation.chosenSeed(satisfied) orelse unreachable;
            // A fresh attempt is armed by `first` rather than by a follow row,
            // and its fate is settled before its count is added: afterwards it is
            // indistinguishable from whatever else landed in that state. All
            // `born` of them share that fate — one state, one arming, and the
            // count is what says how many verdicts it is worth.
            var seeded: u64 = born;
            var at = monitor.aborts.len;
            while (at > 0) {
                at -= 1;
                const abort = monitor.aborts[at];
                if (!Nfa.bitIsSet(root_bits, abort.cond)) continue;
                if (!Nfa.bitOverlaps(obligation.first, monitor.maskOf(at))) continue;
                std.debug.assert(abort.scope == .attempt);
                switch (abort.effect.verdict()) {
                    .holds => report.matches += born,
                    .fails => report.failures += born,
                    .pending => unreachable,
                }
                seeded = 0;
                break;
            }
            state.next_counts[entry] += seeded;
        }
        state.fired = fires;

        // Verdicts. An exit and a `dead` state cannot be the same state — dead
        // stands for the empty subset and an exit for a subset holding one of the
        // operand's — so testing the exit first mirrors the ring rather than
        // imposing a precedence.
        for (state.next_counts, 0..) |*count, index| {
            if (count.* == 0) continue;
            const at: u32 = @intCast(index);
            if (Nfa.bitIsSet(obligation.last, at)) {
                report.matches += count.*;
                count.* = 0;
            } else if (obligation.isDead(at)) {
                report.failures += count.*;
                count.* = 0;
            }
        }
        @memcpy(state.counts, state.next_counts);
        return report;
    }

    /// Evaluate the roots from an atom valuation and step. A convenience for an
    /// interpreter or a test: a compiling backend emits root evaluation itself
    /// and calls `step`.
    pub fn stepAtoms(state: *State, ir: *const Ir, atoms: []const Word) Report {
        state.monitor.alphabet.evaluate(ir, atoms, state.root_bits);
        return state.step(state.root_bits);
    }

    /// Resolve what is still pending because the trace ended.
    ///
    /// Every attempt reaching here is `.pending` by definition — anything the
    /// trace decided was reported in the cycle it was decided — so the whole
    /// question is what the strength turns that into, which is `Strength.atEnd`.
    pub fn finish(state: *State) Report {
        var report: Report = .{};
        const at_end = state.monitor.strength.atEnd();
        switch (state.monitor.mode) {
            // Every attempt a slot stands for is still pending, exactly as every
            // attempt sharing a counted state is.
            .ring => for (state.attempts) |count| {
                switch (at_end) {
                    .fails => report.failures += count,
                    .holds => {},
                    .pending => unreachable, // a trace that ended decides every attempt
                }
            },
            // Every attempt still counted anywhere is still pending, however
            // many share a state.
            .counted => for (state.counts) |count| {
                switch (at_end) {
                    .fails => report.failures += count,
                    .holds => {},
                    .pending => unreachable,
                }
            },
            // Every group of a still-pending attempt counts once for its attempt,
            // whatever valuation it holds: the verdict is the attempt's, and a
            // trace that has ended decides it (LRM 16.10 changes which values an
            // attempt was reasoning about, not how many verdicts it owes).
            .threaded => {
                for (0..state.monitor.depth) |slot| {
                    const count = state.attempts[slot];
                    if (count == 0) continue;
                    switch (at_end) {
                        .fails => report.failures += count,
                        .holds => {},
                        .pending => unreachable,
                    }
                }
            },
        }
        state.clear();
        return report;
    }
};
