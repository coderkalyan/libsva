//! P0: is one merged state vector a sound tracker for an unbounded obligation?
//!
//! `Monitor` tracks an obligation per attempt, in a ring of `depth` state
//! vectors, and refuses an obligation whose longest match is unbounded because
//! the ring cannot be sized (`Glushkov.ObligationNotBounded`). The property
//! layer's phase 3 -- `always`, `s_eventually`, the four `until` forms -- turns
//! on a claim made in athena's `docs/design/sva-property-layer.md` §3: that for
//! an unbounded property, merging every pending attempt into ONE state vector is
//! sound, because two attempts still pending are in the same state.
//!
//! `Monitor.zig:14-17` says the opposite for the general case ("one attempt's
//! threads can die while another's live, and the union hides the failure"), and
//! `Monitor.zig:22-25` says what closes the gap ("tracking it soundly needs the
//! obligation determinised"). Both cannot be right about the same automata, and
//! which one is right decides whether phase 3 is a verdict-rule change or a
//! construction. §9 of the same document asks for this measurement BEFORE the
//! mode is built. This is it.
//!
//! # What is compared
//!
//! For one obligation automaton and one trace, three trackers:
//!
//!   - SEPARATE, the ground truth: one state vector per start cycle, resolved
//!     at the first cycle it matches (a match) or runs out of live states (a
//!     failure) -- exactly `Monitor.State.step`'s per-slot rule, minus the ring
//!     bound.
//!   - MERGED: one vector, advanced once, seeded every cycle, matching when any
//!     state completes and failing when the vector empties. This is the
//!     discipline §3 proposed.
//!   - SUBSETS: a total subset construction where every live subset carries a
//!     count of the attempts sitting in it. The empty subset is a state -- the
//!     trap -- rather than an absence, which is the difference that matters.
//!
//! A discipline is SOUND here when it reports a failure in exactly the cycles
//! some attempt fails in, and likewise for matches. Counts are scored
//! separately: one vector can carry at most one verdict per cycle, and several
//! attempts can resolve in one cycle, so a discipline can preserve the cycle set
//! and still lose the count. That distinction is not academic -- VCS and
//! QuestaSim report per attempt (athena's `9b69e4a`).
//!
//! Each shape is also run wrapped in `first_match`, which is the determinisation
//! libsva has today. `first_match` is verdict-preserving for an obligation --
//! the monitor resolves an attempt at its FIRST match and drops it, so the
//! longer matches it removes are matches no tracker would have reported -- and
//! the run checks that rather than assuming it. That makes the wrapped arm a
//! direct test of `Monitor.zig:22`.
//!
//! # What it found
//!
//! Merging is unsound for a bounded `a ##1 b` in two cycles; it preserves the
//! cycle sets but loses the counts for `always` / `until` / `s_eventually`; and
//! `first_match` rescues none of it, because it is a PARTIAL construction and a
//! dead attempt still has no state to be seen in. The counted subsets are exact
//! everywhere, with a slot count that does not grow with the trace and a
//! multiplicity that does. Full write-up in athena's
//! `docs/design/sva-property-layer.md` §3.
//!
//! Run with `zig build merge`.

const std = @import("std");
const sva = @import("sva");

const Ir = sva.Ir;
const Nfa = sva.Nfa;
const glushkov = sva.glushkov;
const Word = Nfa.Word;
const Allocator = std.mem.Allocator;

/// What a tracker saw over one trace: a bit per cycle carrying a verdict, and
/// how many attempts resolved in each cycle.
const Seen = struct {
    fail_cycles: u64 = 0,
    match_cycles: u64 = 0,
    fails: [16]u8 = @splat(0),
    matches: [16]u8 = @splat(0),
    /// Attempts still live when the trace ended. Weak and strong properties
    /// differ on exactly these, so a tracker that loses them loses the
    /// end-of-trace verdict.
    pending: u32 = 0,
};

const Runner = struct {
    alphabet: *const Nfa.Alphabet,
    nfa: *const Nfa,
    ir: *const Ir,
    current: []Word,
    next: []Word,
    satisfied: []Word,
    root_bits: []Word,

    fn init(gpa: Allocator, automaton: *const glushkov.Automaton, ir: *const Ir) !Runner {
        return .{
            .alphabet = &automaton.alphabet,
            .nfa = &automaton.nfa,
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

    fn evaluate(runner: *Runner, cycle: Word) void {
        const atoms = [_]Word{cycle};
        runner.alphabet.evaluate(runner.ir, &atoms, runner.root_bits);
        runner.nfa.satisfy(runner.alphabet, runner.root_bits, runner.satisfied);
    }

    /// Ground truth: every attempt gets its own vector, and resolves on its own.
    fn separate(runner: *Runner, trace: []const Word) Seen {
        var seen: Seen = .{};
        for (0..trace.len) |index| {
            const start: u32 = @intCast(index);
            var resolved = false;
            var cycle = start;
            while (cycle < trace.len) : (cycle += 1) {
                runner.evaluate(trace[cycle]);
                if (cycle == start) {
                    @memset(runner.current, 0);
                    runner.nfa.seed(runner.satisfied, runner.current);
                } else {
                    runner.nfa.advance(runner.current, runner.satisfied, runner.next);
                    @memcpy(runner.current, runner.next);
                }
                // Seed-then-verdict, and match before failure: `Monitor.State`
                // resolves an attempt in the cycle it is created when the first
                // letter already decides it.
                if (runner.nfa.matched(runner.current)) {
                    seen.match_cycles |= @as(u64, 1) << @intCast(cycle);
                    seen.matches[cycle] += 1;
                    resolved = true;
                    break;
                }
                if (!Nfa.bitAny(runner.current)) {
                    seen.fail_cycles |= @as(u64, 1) << @intCast(cycle);
                    seen.fails[cycle] += 1;
                    resolved = true;
                    break;
                }
            }
            if (!resolved) seen.pending += 1;
        }
        return seen;
    }

    /// The candidate: one vector for every attempt at once.
    fn merged(runner: *Runner, trace: []const Word) Seen {
        var seen: Seen = .{};
        @memset(runner.current, 0);
        for (trace, 0..) |atoms, index| {
            runner.evaluate(atoms);
            runner.nfa.advance(runner.current, runner.satisfied, runner.next);
            runner.nfa.seed(runner.satisfied, runner.next);
            @memcpy(runner.current, runner.next);

            if (runner.nfa.matched(runner.current)) {
                seen.match_cycles |= @as(u64, 1) << @intCast(index);
                seen.matches[index] = 1;
            }
            if (!Nfa.bitAny(runner.current)) {
                seen.fail_cycles |= @as(u64, 1) << @intCast(index);
                seen.fails[index] = 1;
            }
        }
        if (Nfa.bitAny(runner.current)) seen.pending = 1;
        return seen;
    }
};

/// The third discipline, and the one phase 3 would actually build: track the
/// obligation determinised, and carry a COUNT with each subset.
///
/// The naive merged vector loses a failure because a dead attempt has no state
/// to be seen in — the union of the live sets cannot distinguish "no attempt is
/// in these states" from "one attempt died here". Determinising fixes exactly
/// that, but only if the construction is TOTAL: the empty subset has to be a
/// state (the trap) rather than an absence. Two attempts in the same subset have
/// the same future by construction, so they may share a slot — but they are
/// still two attempts, and the verdict has to say so, which is what the count is
/// for.
///
/// This is a lazy subset construction over concrete atom valuations rather than
/// over guard cells, which is sound here because the harness enumerates
/// valuations anyway; libsva's `firstMatchCells` cuts cells precisely to avoid
/// depending on the valuation space, and it would be reused for the real thing.
const Subsets = struct {
    /// A live subset and how many attempts sit in it.
    const Slot = struct { subset: u64, count: u32 };

    nfa: *const Nfa,
    /// `follow[s]` and the state masks, flattened into single words. The
    /// experiment caps the automaton at 64 states, which every shape here is
    /// far below.
    follow: [64]u64 = @splat(0),
    first: u64 = 0,
    last: u64 = 0,
    /// The satisfied-state mask per atom valuation, precomputed once.
    satisfied: [8]u64 = @splat(0),

    live: [64]Slot = undefined,
    live_count: usize = 0,
    /// The most subsets live at once, over every trace: the number of counters
    /// a monitor in this discipline would have to carry.
    peak: usize = 0,
    /// The largest multiplicity any one of them carried, which is what decides
    /// whether a counter can be a bit.
    max_count: u32 = 0,

    fn init(runner: *Runner, atom_count: u5) Subsets {
        std.debug.assert(runner.nfa.state_count <= 64);
        std.debug.assert(runner.nfa.words == 1);

        var subsets: Subsets = .{ .nfa = runner.nfa };
        for (0..runner.nfa.state_count) |state| {
            subsets.follow[state] = runner.nfa.followOf(@intCast(state))[0];
        }
        subsets.first = runner.nfa.first[0];
        subsets.last = runner.nfa.last[0];
        for (0..@as(u32, 1) << atom_count) |letter| {
            runner.evaluate(@intCast(letter));
            subsets.satisfied[letter] = runner.satisfied[0];
        }
        return subsets;
    }

    fn step(subsets: *const Subsets, subset: u64, letter: usize) u64 {
        var reached: u64 = 0;
        var rest = subset;
        while (rest != 0) {
            const state = @ctz(rest);
            rest &= rest - 1;
            reached |= subsets.follow[state];
        }
        return reached & subsets.satisfied[letter];
    }

    fn add(subsets: *Subsets, subset: u64, count: u32) void {
        for (subsets.live[0..subsets.live_count]) |*slot| {
            if (slot.subset == subset) {
                slot.count += count;
                return;
            }
        }
        subsets.live[subsets.live_count] = .{ .subset = subset, .count = count };
        subsets.live_count += 1;
    }

    fn run(subsets: *Subsets, trace: []const Word) Seen {
        var seen: Seen = .{};
        subsets.live_count = 0;

        for (trace, 0..) |atoms, index| {
            const letter: usize = @intCast(atoms);
            var carried: [64]Slot = undefined;
            const carried_count = subsets.live_count;
            @memcpy(carried[0..carried_count], subsets.live[0..carried_count]);
            subsets.live_count = 0;

            var fails: u32 = 0;
            var matches: u32 = 0;

            for (carried[0..carried_count]) |slot| {
                const next = subsets.step(slot.subset, letter);
                if (next & subsets.last != 0) {
                    matches += slot.count;
                } else if (next == 0) {
                    // The trap. Reached as a state, not as an absence — which
                    // is the whole difference from the merged vector.
                    fails += slot.count;
                } else {
                    subsets.add(next, slot.count);
                }
            }

            const seeded = subsets.first & subsets.satisfied[letter];
            if (seeded & subsets.last != 0) {
                matches += 1;
            } else if (seeded == 0) {
                fails += 1;
            } else {
                subsets.add(seeded, 1);
            }

            if (fails != 0) {
                seen.fail_cycles |= @as(u64, 1) << @intCast(index);
                seen.fails[index] = @intCast(fails);
            }
            if (matches != 0) {
                seen.match_cycles |= @as(u64, 1) << @intCast(index);
                seen.matches[index] = @intCast(matches);
            }
            subsets.peak = @max(subsets.peak, subsets.live_count);
            for (subsets.live[0..subsets.live_count]) |slot| {
                subsets.max_count = @max(subsets.max_count, slot.count);
            }
        }

        for (subsets.live[0..subsets.live_count]) |slot| seen.pending += slot.count;
        return seen;
    }
};

/// Every trace of `length` cycles over `atom_count` atoms.
const Traces = struct {
    atom_count: u5,
    length: u32,
    combinations: u64,
    assignment: u64 = 0,
    bits: u64 = 0,
    buffer: [16]Word = undefined,

    fn init(atom_count: u5, length: u32) Traces {
        return .{
            .atom_count = atom_count,
            .length = length,
            .combinations = @as(u64, 1) << @intCast(@as(u32, atom_count) * length),
        };
    }

    fn next(traces: *Traces) ?[]const Word {
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

const Outcome = struct {
    states: u32,
    /// The merged tracker reports a failure in exactly the cycles an attempt
    /// fails in.
    fail_cycles_agree: bool = true,
    match_cycles_agree: bool = true,
    /// ... and as many of them.
    counts_agree: bool = true,
    /// ... and knows something is still owed at the end of the trace.
    pending_agrees: bool = true,
    /// The first trace that separated them, for a hand check.
    counterexample: ?u64 = null,
    /// What that trace produced, as cycle bitmaps.
    want_fail: u64 = 0,
    got_fail: u64 = 0,
    want_match: u64 = 0,
    got_match: u64 = 0,

    /// The determinised-with-counters arm: exact on every count, or not.
    subsets_exact: bool = true,
    /// How many counters that arm needed at once, at its worst, and the largest
    /// multiplicity one of them carried.
    subsets_peak: usize = 0,
    subsets_max_count: u32 = 0,
    /// The most attempts a single cycle resolved, which is what a tracker
    /// carrying one verdict per cycle cannot report.
    peak_resolved: u32 = 0,
};

fn measure(
    gpa: Allocator,
    ir: *const Ir,
    node: Ir.Node.Index,
    atom_count: u5,
    length: u32,
) !Outcome {
    var automaton = try glushkov.sequence(gpa, ir, node, .{}, null);
    defer automaton.deinit(gpa);

    var runner = try Runner.init(gpa, &automaton, ir);
    defer runner.deinit(gpa);

    var outcome: Outcome = .{ .states = automaton.nfa.state_count };
    var subsets: Subsets = .init(&runner, atom_count);
    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |trace| {
        const want = runner.separate(trace);
        const got = runner.merged(trace);
        const det = subsets.run(trace);

        const fails_ok = want.fail_cycles == got.fail_cycles;
        const matches_ok = want.match_cycles == got.match_cycles;
        var counts_ok = true;
        var det_ok = want.fail_cycles == det.fail_cycles and
            want.match_cycles == det.match_cycles and
            want.pending == det.pending;
        for (0..length) |cycle| {
            counts_ok = counts_ok and want.fails[cycle] == got.fails[cycle];
            counts_ok = counts_ok and want.matches[cycle] == got.matches[cycle];
            det_ok = det_ok and want.fails[cycle] == det.fails[cycle];
            det_ok = det_ok and want.matches[cycle] == det.matches[cycle];
            const resolved = @as(u32, want.fails[cycle]) + want.matches[cycle];
            outcome.peak_resolved = @max(outcome.peak_resolved, resolved);
        }
        const pending_ok = (want.pending != 0) == (got.pending != 0);

        outcome.fail_cycles_agree = outcome.fail_cycles_agree and fails_ok;
        outcome.match_cycles_agree = outcome.match_cycles_agree and matches_ok;
        outcome.counts_agree = outcome.counts_agree and counts_ok;
        outcome.pending_agrees = outcome.pending_agrees and pending_ok;
        outcome.subsets_exact = outcome.subsets_exact and det_ok;

        if (outcome.counterexample == null and !(fails_ok and matches_ok)) {
            outcome.counterexample = traces.bits;
            outcome.want_fail = want.fail_cycles;
            outcome.got_fail = got.fail_cycles;
            outcome.want_match = want.match_cycles;
            outcome.got_match = got.match_cycles;
        }
    }
    outcome.subsets_peak = subsets.peak;
    outcome.subsets_max_count = subsets.max_count;
    return outcome;
}

/// Whether wrapping an obligation in `first_match` changes any verdict.
///
/// The claim this checks is what makes `first_match` a legitimate stand-in for
/// "the determinised obligation": a monitor resolves an attempt at its FIRST
/// match and drops it, so the longer matches `first_match` removes are matches
/// no tracker would have reported anyway. If that is wrong, the second arm of
/// the table above measures the wrong thing.
fn preserves(
    gpa: Allocator,
    ir: *const Ir,
    plain: Ir.Node.Index,
    wrapped: Ir.Node.Index,
    atom_count: u5,
    length: u32,
) !bool {
    var bare = try glushkov.sequence(gpa, ir, plain, .{}, null);
    defer bare.deinit(gpa);
    var det = try glushkov.sequence(gpa, ir, wrapped, .{}, null);
    defer det.deinit(gpa);

    var bare_runner = try Runner.init(gpa, &bare, ir);
    defer bare_runner.deinit(gpa);
    var det_runner = try Runner.init(gpa, &det, ir);
    defer det_runner.deinit(gpa);

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |trace| {
        const left = bare_runner.separate(trace);
        const right = det_runner.separate(trace);
        if (left.fail_cycles != right.fail_cycles) return false;
        if (left.match_cycles != right.match_cycles) return false;
        if (left.pending != right.pending) return false;
        if (!std.mem.eql(u8, &left.fails, &right.fails)) return false;
        if (!std.mem.eql(u8, &left.matches, &right.matches)) return false;
    }
    return true;
}

/// A one-cycle sequence over atom `handle`.
fn atomCycle(ir: *Ir, gpa: Allocator, handle: u32) !Ir.Node.Index {
    return ir.seqBoolean(gpa, try ir.atom(gpa, @enumFromInt(handle)));
}

const Shape = struct {
    name: []const u8,
    /// Whether `assertion()` refuses this obligation today, i.e. whether it is
    /// one of the shapes phase 3 exists to support.
    unbounded: bool,
    build: *const fn (ir: *Ir, gpa: Allocator) anyerror!Ir.Node.Index,
};

const shapes = [_]Shape{
    .{ .name = "a", .unbounded = false, .build = buildBoolean },
    .{ .name = "a ##1 b", .unbounded = false, .build = buildConcat },
    .{ .name = "a ##[1:2] b", .unbounded = false, .build = buildRange },
    .{ .name = "a ##[1:$] b", .unbounded = true, .build = buildOpenRange },
    .{ .name = "a[*1:$]  (always a)", .unbounded = true, .build = buildAlways },
    .{ .name = "a throughout ##[0:$] b  (a until_with b)", .unbounded = true, .build = buildUntil },
    .{ .name = "##[1:$] b  (s_eventually b)", .unbounded = true, .build = buildEventually },
    .{ .name = "b[->1]", .unbounded = true, .build = buildGoto },
};

fn buildBoolean(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return atomCycle(ir, gpa, 0);
}

fn buildConcat(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 1, try atomCycle(ir, gpa, 1));
}

fn buildRange(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, 2, try atomCycle(ir, gpa, 1));
}

fn buildOpenRange(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return ir.seqDelay(gpa, try atomCycle(ir, gpa, 0), 1, Ir.unbounded, try atomCycle(ir, gpa, 1));
}

fn buildAlways(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return ir.seqRepeat(gpa, .seq_repeat_consec, try atomCycle(ir, gpa, 0), 1, Ir.unbounded);
}

/// `a until_with b`: `b` eventually matches and `a` holds over the whole span.
/// The `until` obligation, written with the tags that exist today.
fn buildUntil(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    const cycle = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    const reach = try ir.seqDelay(gpa, cycle, 0, Ir.unbounded, try atomCycle(ir, gpa, 1));
    return ir.seqThroughout(gpa, try ir.atom(gpa, @enumFromInt(0)), reach);
}

fn buildEventually(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    const cycle = try ir.seqBoolean(gpa, try ir.boolConst(gpa, true));
    return ir.seqDelay(gpa, cycle, 1, Ir.unbounded, try atomCycle(ir, gpa, 1));
}

fn buildGoto(ir: *Ir, gpa: Allocator) !Ir.Node.Index {
    return ir.seqRepeat(gpa, .seq_repeat_goto, try atomCycle(ir, gpa, 1), 1, 1);
}

fn verdict(outcome: Outcome) []const u8 {
    if (!outcome.fail_cycles_agree and !outcome.match_cycles_agree) return "UNSOUND (fails+matches)";
    if (!outcome.fail_cycles_agree) return "UNSOUND (loses failures)";
    if (!outcome.match_cycles_agree) return "UNSOUND (loses matches)";
    if (!outcome.counts_agree) return "cycles agree, COUNTS differ";
    return "sound";
}

pub fn main() !void {
    var debug: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug.deinit();
    const gpa = debug.allocator();

    const atom_count: u5 = 2;
    const short: u32 = 6;
    const long: u32 = 8;

    const p = std.debug.print;
    p("obligation tracking disciplines, every trace of {d} and {d} cycles over {d} atoms\n\n", .{
        short,
        long,
        atom_count,
    });

    for (shapes) |shape| {
        var ir: Ir = .init();
        defer ir.deinit(gpa);
        const node = try shape.build(&ir, gpa);
        const det_node = try ir.seqFirstMatch(gpa, try shape.build(&ir, gpa));

        const bare = try measure(gpa, &ir, node, atom_count, short);
        const det = try measure(gpa, &ir, det_node, atom_count, short);
        const wide = try measure(gpa, &ir, node, atom_count, long);
        const kept = try preserves(gpa, &ir, node, det_node, atom_count, short);

        p("{s}{s}\n", .{ shape.name, if (shape.unbounded) "   [refused today]" else "" });
        p("  merged vector        {d:>3} states    {s}\n", .{ bare.states, verdict(bare) });
        p("  merged, first_match  {d:>3} states    {s}{s}\n", .{
            det.states,
            verdict(det),
            if (kept) "" else "   [!! first_match changed the verdicts]",
        });
        p("  subsets + counters   {d:>3} counters  {s}   ({d} counters at {d} cycles)\n", .{
            bare.subsets_peak,
            if (bare.subsets_exact and wide.subsets_exact) "EXACT, counts included" else "INEXACT",
            wide.subsets_peak,
            long,
        });
        p("    multiplicity       {d:>3} at {d} cycles, {d} at {d} cycles" ++
            "   (up to {d} attempts resolve in one cycle)\n", .{
            bare.subsets_max_count,
            short,
            wide.subsets_max_count,
            long,
            wide.peak_resolved,
        });
        if (bare.counterexample) |bits| {
            p("    counterexample     trace {b:0>12}: attempts fail at {b:0>6}, merged says {b:0>6}\n", .{
                bits,
                @bitReverse(bare.want_fail) >> 58,
                @bitReverse(bare.got_fail) >> 58,
            });
        }
        p("\n", .{});
    }
}
