//! A compiled assertion: what a backend runs, once per cycle.
//!
//! # Structure
//!
//! An assertion is one *obligation* automaton plus, for an implication, a
//! *trigger* automaton that says when to start one.
//!
//! The two are tracked differently, and the asymmetry is the whole design:
//!
//!   - The **trigger** is existential. All we ask of the antecedent is whether
//!     *some* match ends this cycle, so the union of every antecedent attempt is
//!     a sound representation and one state vector suffices. Antecedent matches
//!     ending in the same cycle impose the same obligation, so they collapse.
//!   - The **obligation** is per-attempt. Merging attempts into one state vector
//!     is *unsound*: one attempt's threads can die while another's live, and the
//!     union hides the failure. Each pending attempt therefore gets its own
//!     state vector.
//!
//! `depth` bounds how many can be pending at once. It is the obligation's
//! longest match: an attempt seeded in cycle `t` has either matched or run out
//! of live states by cycle `t + depth - 1`, so `depth` vectors in a ring is
//! exact rather than a heuristic. An obligation with no bounded longest match is
//! rejected at lowering time — tracking it soundly needs the obligation
//! determinised, so that one state fully characterises an attempt and merging
//! becomes sound again.
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
//!     like any assertion. Note whether a match now ends.
//!  5. If a match ends now and an `Abort` with `covers_trigger` holds, resolve
//!     it with that abort's verdict and let it seed nothing.
//!  6. Advance every pending obligation attempt.
//!  7. Seed a new attempt: every cycle when there is no trigger, in the cycle
//!     the trigger fires for `|->`, and the cycle after for `|=>`.
//!  8. Apply every armed `Abort` whose condition holds. An abort is armed for
//!     an attempt when that attempt has a live state in the abort's mask.
//!  9. Take verdicts. An attempt that reached an accepting state is discharged;
//!     one with no live states failed. Both leave the ring.
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
/// The property's sequence. One attempt per seeding.
obligation: Nfa,
/// How many attempts can be pending at once — the obligation's longest match,
/// and so the number of state vectors a backend allocates.
depth: u32,
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

pub const Strength = enum(u8) {
    /// A pending attempt holds when the trace ends.
    weak,
    /// A pending attempt fails when the trace ends.
    strong,
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
    /// A trigger is the union of every attempt's antecedent, so resolving one
    /// attempt of it and leaving the others matching is not expressible. It
    /// never has to be. A monitor has exactly one trigger — every operator that
    /// could put an abort beside a second one refuses first — so an abort above
    /// it covers all of its attempts uniformly, and one verdict says everything
    /// there is to say. `step` resolves the antecedent match ending this cycle
    /// with `effect` instead of seeding an obligation from it.
    ///
    /// Only ever set on a trigger whose every antecedent match is one cycle
    /// long; `lowerAbort` refuses the rest, because an attempt halfway through a
    /// longer antecedent owes a verdict of its own that the union cannot count.
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
    };

    pub const Scope = enum(u8) {
        /// Firing resolves the whole attempt. An accept always does — a
        /// disjunct holding makes the disjunction hold — and so does a reject
        /// with nothing but conjunctions and shifts above it.
        attempt,
        /// Firing only kills the threads in the mask, leaving the rest of the
        /// attempt to run on. A reject under a disjunction: that disjunct is
        /// dead, the other may still hold.
        branch,
    };
};

pub fn deinit(monitor: *Monitor, gpa: Allocator) void {
    monitor.alphabet.deinit(gpa);
    if (monitor.trigger) |*trigger| trigger.deinit(gpa);
    monitor.obligation.deinit(gpa);
    gpa.free(monitor.aborts);
    gpa.free(monitor.abort_masks);
}

/// The obligation states the abort at `index` guards.
pub fn maskOf(monitor: *const Monitor, index: usize) []const Word {
    const words = monitor.obligation.words;
    return monitor.abort_masks[index * words ..][0..words];
}

/// What one cycle did. A `cover` counts `matches`; an `assert` reports
/// `failures`.
pub const Report = struct {
    /// Obligation attempts that ran out of live states.
    failures: u32 = 0,
    /// Obligation attempts discharged by a completed match.
    matches: u32 = 0,
    /// The reset condition held, so everything in flight was abandoned.
    reset: bool = false,
};

/// A reference run of the protocol in the file comment. This is the executable
/// form of the contract a backend has to reproduce, not something a backend
/// would call.
pub const State = struct {
    monitor: *const Monitor,
    /// A bit per entry of `alphabet.roots`.
    root_bits: []Word,
    /// Trigger and obligation states whose guard holds this cycle.
    trigger_satisfied: []Word,
    obligation_satisfied: []Word,
    /// The union of every trigger attempt.
    trigger_current: []Word,
    /// `depth` obligation state vectors, each either free or holding one
    /// pending attempt. The order is deliberately meaningless: what makes a
    /// fixed pool work is that `depth` is the obligation's longest match, so an
    /// attempt is always resolved before `depth` more have started and a free
    /// slot always exists when one is needed.
    pending: []Word,
    occupied: []bool,
    /// Whether the trigger matched in the previous cycle, for `|=>`.
    fired: bool,
    /// `aborts.len * depth` flags: whether an abort's subtree could hold a
    /// thread of that attempt this cycle. Set before the advance from the
    /// reachable set rather than the live one — see `step`. Empty when there are
    /// no aborts.
    abort_entered: []bool,
    /// One obligation mask of scratch for computing that reachable set.
    abort_reach: []Word,
    scratch: []Word,

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
        const occupied = try gpa.alloc(bool, monitor.depth);
        errdefer gpa.free(occupied);
        const abort_entered = try gpa.alloc(bool, monitor.aborts.len * monitor.depth);
        errdefer gpa.free(abort_entered);
        const abort_reach = try gpa.alloc(Word, obligation_words);
        errdefer gpa.free(abort_reach);
        const scratch = try gpa.alloc(Word, @max(trigger_words, obligation_words));

        var state: State = .{
            .monitor = monitor,
            .root_bits = root_bits,
            .trigger_satisfied = trigger_satisfied,
            .obligation_satisfied = obligation_satisfied,
            .trigger_current = trigger_current,
            .pending = pending,
            .occupied = occupied,
            .fired = false,
            .abort_entered = abort_entered,
            .abort_reach = abort_reach,
            .scratch = scratch,
        };
        state.clear();
        return state;
    }

    pub fn deinit(state: *State, gpa: Allocator) void {
        gpa.free(state.root_bits);
        gpa.free(state.trigger_satisfied);
        gpa.free(state.obligation_satisfied);
        gpa.free(state.trigger_current);
        gpa.free(state.pending);
        gpa.free(state.occupied);
        gpa.free(state.abort_entered);
        gpa.free(state.abort_reach);
        gpa.free(state.scratch);
    }

    /// Abandon everything in flight.
    pub fn clear(state: *State) void {
        @memset(state.trigger_current, 0);
        @memset(state.pending, 0);
        @memset(state.occupied, false);
        @memset(state.abort_entered, false);
        state.fired = false;
    }

    /// The state vector of the attempt living in `slot`.
    fn slotOf(state: *State, slot: u32) []Word {
        const words = state.monitor.obligation.words;
        return state.pending[@as(usize, slot) * words ..][0..words];
    }

    fn freeSlot(state: *const State) ?u32 {
        for (state.occupied, 0..) |busy, slot| {
            if (!busy) return @intCast(slot);
        }
        return null;
    }

    /// Advance one cycle, given a bit per `alphabet.roots`.
    ///
    /// This is the protocol proper. Producing the root bits is the caller's
    /// half of the contract — it is the only part that needs to know what an
    /// atom means — so a backend that has evaluated them itself steps from
    /// here and never touches an `Ir`.
    pub fn step(state: *State, root_bits: []const Word) Report {
        const monitor = state.monitor;

        if (Nfa.bitIsSet(root_bits, monitor.reset)) {
            state.clear();
            return .{ .reset = true };
        }

        monitor.obligation.satisfy(&monitor.alphabet, root_bits, state.obligation_satisfied);

        var fires = false;
        if (monitor.trigger) |*trigger| {
            trigger.satisfy(&monitor.alphabet, root_bits, state.trigger_satisfied);
            const next = state.scratch[0..trigger.words];
            trigger.advance(state.trigger_current, state.trigger_satisfied, next);
            @memcpy(state.trigger_current, next);
            trigger.seed(state.trigger_satisfied, state.trigger_current);
            fires = trigger.matched(state.trigger_current);
        }

        // An abort above the implication covers the antecedent too, so a match
        // ending now is resolved by it rather than seeding an obligation. This
        // is the cycle the pushed-down spelling cannot see at all.
        //
        // Nothing here tells one attempt from another, and nothing needs to:
        // the trigger is the union of every attempt's antecedent, but the abort
        // covers all of them, so one verdict discharges what the union stands
        // for. It is also why the trigger vector is left alone rather than
        // cleared. `lowerAbort` only sets `covers_trigger` when every follow row
        // of the trigger is empty, so no partially matched antecedent survives
        // this cycle to complete later and impose an obligation on an attempt
        // the abort has already resolved. Lift that restriction and this is the
        // first thing that has to change.
        var trigger_verdict: ?Abort.Effect = null;
        if (fires) {
            for (monitor.aborts) |abort| {
                if (!abort.covers_trigger) continue;
                if (!Nfa.bitIsSet(root_bits, abort.cond)) continue;
                // Last wins. Lowering appends outward, so the last entry is the
                // outermost abort, and an outer abort beats an inner verdict
                // landing in the same cycle.
                trigger_verdict = abort.effect;
            }
            if (trigger_verdict != null) fires = false;
        }

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
            for (state.occupied, 0..) |busy, slot| {
                if (!busy) continue;
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
        for (state.occupied, 0..) |busy, slot| {
            if (!busy) continue;
            const current = state.slotOf(@intCast(slot));
            monitor.obligation.advance(current, state.obligation_satisfied, next);
            @memcpy(current, next);
        }

        const seeding = if (monitor.trigger == null)
            true
        else if (monitor.overlap)
            fires
        else
            state.fired;
        if (seeding) {
            // A slot is always free here: an attempt is resolved by the time it
            // is `depth - 1` cycles old, so at most `depth - 1` of them survived
            // into this cycle.
            const slot = state.freeSlot() orelse unreachable;
            const current = state.slotOf(slot);
            @memset(current, 0);
            monitor.obligation.seed(state.obligation_satisfied, current);
            state.occupied[slot] = true;
            // A fresh attempt reaches the entry states, guards aside.
            for (monitor.aborts, 0..) |_, index| {
                if (Nfa.bitOverlaps(monitor.obligation.first, monitor.maskOf(index))) {
                    state.abort_entered[index * depth + slot] = true;
                }
            }
        }
        state.fired = fires;

        var report: Report = .{};
        if (trigger_verdict) |effect| {
            switch (effect) {
                .accept => report.matches += 1,
                .reject => report.failures += 1,
            }
        }

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
            for (state.occupied, 0..) |busy, slot| {
                if (!busy) continue;
                // Armed if the abort's subtree could hold a thread of this
                // attempt this cycle. That is the reachable set, computed above:
                // the live set would miss both the cycle the subtree is entered
                // in and the cycle its last thread dies in.
                if (!state.abort_entered[index * depth + slot]) continue;
                const current = state.slotOf(@intCast(slot));
                switch (abort.scope) {
                    .attempt => {
                        switch (abort.effect) {
                            .accept => report.matches += 1,
                            .reject => report.failures += 1,
                        }
                        @memset(current, 0);
                        state.occupied[slot] = false;
                    },
                    // The guarded threads die; whether that kills the attempt
                    // is the ordinary verdict below deciding it has no live
                    // states left.
                    .branch => Nfa.bitSubtract(current, mask),
                }
            }
        }

        for (state.occupied, 0..) |busy, slot| {
            if (!busy) continue;
            const current = state.slotOf(@intCast(slot));
            if (monitor.obligation.matched(current)) {
                report.matches += 1;
                state.occupied[slot] = false;
            } else if (!Nfa.bitAny(current)) {
                report.failures += 1;
                state.occupied[slot] = false;
            }
        }
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
    pub fn finish(state: *State) Report {
        var report: Report = .{};
        for (state.occupied) |occupied| {
            if (!occupied) continue;
            if (state.monitor.strength == .strong) report.failures += 1;
        }
        state.clear();
        return report;
    }
};
