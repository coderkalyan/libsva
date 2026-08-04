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
//!  5. Advance every pending obligation attempt.
//!  6. Seed a new attempt: every cycle when there is no trigger, in the cycle
//!     the trigger fires for `|->`, and the cycle after for `|=>`.
//!  7. Take verdicts. An attempt that reached an accepting state is discharged;
//!     one with no live states failed. Both leave the ring.
//!  8. When a finite trace ends, a still-pending attempt holds if the property
//!     is weak and fails if it is strong.
//!
//! Seeding before taking verdicts is not incidental: under `|->` the obligation
//! starts on the trigger's last cycle, so a one-cycle consequent has to be able
//! to discharge in the cycle it was created.

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

pub const Strength = enum(u8) {
    /// A pending attempt holds when the trace ends.
    weak,
    /// A pending attempt fails when the trace ends.
    strong,
};

pub fn deinit(monitor: *Monitor, gpa: Allocator) void {
    monitor.alphabet.deinit(gpa);
    if (monitor.trigger) |*trigger| trigger.deinit(gpa);
    monitor.obligation.deinit(gpa);
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
        gpa.free(state.scratch);
    }

    /// Abandon everything in flight.
    pub fn clear(state: *State) void {
        @memset(state.trigger_current, 0);
        @memset(state.pending, 0);
        @memset(state.occupied, false);
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
        }
        state.fired = fires;

        var report: Report = .{};
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
