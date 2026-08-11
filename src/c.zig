//! The C API: a thin wrapper, one export per Zig entry point.
//!
//! # Objects
//!
//! An `sva_ir`, an `sva_monitor` and an `sva_run` are the Zig objects sitting in
//! the caller's own storage — declared on the stack, initialised in place, and
//! torn down there. Only what they grow internally is allocated. The header
//! spells their sizes as byte arrays whose contents are nobody's business, and a
//! `comptime` check here fails the build if one of them outgrows its constant.
//!
//! An `sva_automaton` is only ever reached by pointer, so it stays an incomplete
//! type.
//!
//! Allocation goes to `std.heap.smp_allocator`, which is stateless and
//! thread-safe. There is nothing to configure and nothing to pass.
//!
//! # Failure
//!
//! A function that can fail returns a status and delivers its result through an
//! out parameter, which is written only on `SVA_OK`. The only thing a builder can
//! report is an allocation failure; there is no accumulated state, so a caller
//! that ignores a return leaves its handle unwritten and will hand garbage to the
//! next call.
//!
//! Nothing else is checked here. A malformed tree — an operand from the wrong
//! layer, a reversed range, goto over a compound sequence — is a bug in the
//! caller's lowering rather than a property of what it is compiling, so there is
//! no source location to report and nothing to hand back. Those keep the Zig
//! assertions and abort. Legal input this library cannot lower yet is a different
//! thing entirely, and comes back from `sva_lower` as a status with the node that
//! caused it.
//!
//! Pointer arguments must not be null.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");
const Nfa = @import("Nfa.zig");
const Monitor = @import("Monitor.zig");
const glushkov = @import("Glushkov.zig");

const Word = Nfa.Word;

/// Stateless and thread-safe, so there is no allocator to hand in or hold.
const gpa = std.heap.smp_allocator;

const Status = enum(c_int) {
    ok = 0,
    out_of_memory = 1,
    state_limit = 2,
    depth_limit = 3,
    unbounded_obligation = 4,
    unsupported_consequent = 5,
    unsupported_repeat_operand = 6,
    mixed_strength = 7,
    unsupported_abort_scope = 8,

    fn of(err: glushkov.Error) Status {
        return switch (err) {
            error.OutOfMemory => .out_of_memory,
            error.StateLimitExceeded => .state_limit,
            error.DepthLimitExceeded => .depth_limit,
            error.ObligationNotBounded => .unbounded_obligation,
            error.UnsupportedConsequent => .unsupported_consequent,
            error.UnsupportedRepeatOperand => .unsupported_repeat_operand,
            error.MixedStrength => .mixed_strength,
            error.UnsupportedAbortScope => .unsupported_abort_scope,
        };
    }
};

/// Deliver a builder's result, or report the one failure it has.
fn deliver(out: *u32, result: Allocator.Error!Ir.Node.Index) Status {
    out.* = @intFromEnum(result catch return .out_of_memory);
    return .ok;
}

// -- The IR ------------------------------------------------------------------

export fn sva_ir_init(ir: *Ir) void {
    ir.* = .init();
}

export fn sva_ir_deinit(ir: *Ir) void {
    ir.deinit(gpa);
}

export fn sva_ir_node_count(ir: *const Ir) u32 {
    return @intCast(ir.nodes.len);
}

/// One past the highest atom handle mentioned: the width of the per-cycle
/// valuation a backend allocates.
export fn sva_ir_atom_count(ir: *const Ir) u32 {
    return ir.atom_count;
}

// -- Boolean layer -----------------------------------------------------------

export fn sva_bool_const(ir: *Ir, value: bool, out: *u32) Status {
    return deliver(out, ir.boolConst(gpa, value));
}

export fn sva_bool_atom(ir: *Ir, atom: u32, out: *u32) Status {
    return deliver(out, ir.atom(gpa, @enumFromInt(atom)));
}

export fn sva_bool_not(ir: *Ir, operand: u32, out: *u32) Status {
    return deliver(out, ir.bNot(gpa, @enumFromInt(operand)));
}

export fn sva_bool_and(ir: *Ir, lhs: u32, rhs: u32, out: *u32) Status {
    return deliver(out, ir.bBinary(gpa, .band, @enumFromInt(lhs), @enumFromInt(rhs)));
}

export fn sva_bool_or(ir: *Ir, lhs: u32, rhs: u32, out: *u32) Status {
    return deliver(out, ir.bBinary(gpa, .bor, @enumFromInt(lhs), @enumFromInt(rhs)));
}

/// How a boolean node is shaped, numbered as `Ir.Node.Tag` orders the layer.
const BoolOp = enum(c_int) { constant, atom, not, conjunction, disjunction };

const BoolView = extern struct { op: BoolOp, lhs: u32, rhs: u32 };

/// Read a boolean node back — the other half of the alphabet contract, since a
/// monitor's roots are boolean nodes a backend has to walk to emit.
export fn sva_bool_read(ir: *const Ir, node: u32, out: *BoolView) void {
    const tag = ir.tagOf(@enumFromInt(node));
    std.debug.assert(tag.layer() == .boolean);
    const data = ir.dataOf(@enumFromInt(node));
    out.* = .{ .op = @enumFromInt(@intFromEnum(tag)), .lhs = data.lhs, .rhs = data.rhs };
}

// -- Sequence layer ----------------------------------------------------------

export fn sva_seq_boolean(ir: *Ir, boolean: u32, out: *u32) Status {
    return deliver(out, ir.seqBoolean(gpa, @enumFromInt(boolean)));
}

export fn sva_seq_delay(ir: *Ir, left: u32, lo: u32, hi: u32, right: u32, out: *u32) Status {
    return deliver(out, ir.seqDelay(gpa, @enumFromInt(left), lo, hi, @enumFromInt(right)));
}

export fn sva_seq_or(ir: *Ir, lhs: u32, rhs: u32, out: *u32) Status {
    return deliver(out, ir.seqOr(gpa, @enumFromInt(lhs), @enumFromInt(rhs)));
}

export fn sva_seq_and(ir: *Ir, lhs: u32, rhs: u32, out: *u32) Status {
    return deliver(out, ir.seqAnd(gpa, @enumFromInt(lhs), @enumFromInt(rhs)));
}

export fn sva_seq_intersect(ir: *Ir, lhs: u32, rhs: u32, out: *u32) Status {
    return deliver(out, ir.seqIntersect(gpa, @enumFromInt(lhs), @enumFromInt(rhs)));
}

/// `exp` is a *boolean* node, not a sequence.
export fn sva_seq_throughout(ir: *Ir, exp: u32, sequence: u32, out: *u32) Status {
    return deliver(out, ir.seqThroughout(gpa, @enumFromInt(exp), @enumFromInt(sequence)));
}

/// Sugar: builds `(true ##[0:$] inner ##[0:$] true) intersect outer` out of the
/// tags the IR already has, rather than being one of its own.
export fn sva_seq_within(ir: *Ir, inner: u32, outer: u32, out: *u32) Status {
    return deliver(out, ir.seqWithin(gpa, @enumFromInt(inner), @enumFromInt(outer)));
}

/// Determinises its operand, so it is the one sequence constructor whose
/// lowering can be exponential in the size of what it is given.
export fn sva_seq_first_match(ir: *Ir, sequence: u32, out: *u32) Status {
    return deliver(out, ir.seqFirstMatch(gpa, @enumFromInt(sequence)));
}

const Repeat = enum(c_int) {
    consecutive,
    goto,
    nonconsecutive,

    fn tag(kind: Repeat) Ir.Node.Tag {
        return switch (kind) {
            .consecutive => .seq_repeat_consec,
            .goto => .seq_repeat_goto,
            .nonconsecutive => .seq_repeat_nonconsec,
        };
    }
};

export fn sva_seq_repeat(ir: *Ir, kind: Repeat, sequence: u32, lo: u32, hi: u32, out: *u32) Status {
    return deliver(out, ir.seqRepeat(gpa, kind.tag(), @enumFromInt(sequence), lo, hi));
}

// -- Property layer ----------------------------------------------------------

const Strength = enum(c_int) { weak, strong };
const Implication = enum(c_int) { overlapping, nonoverlapping };

export fn sva_prop_sequence(ir: *Ir, strength: Strength, sequence: u32, out: *u32) Status {
    const tag: Ir.Node.Tag = switch (strength) {
        .weak => .prop_seq_weak,
        .strong => .prop_seq_strong,
    };
    return deliver(out, ir.propSeq(gpa, tag, @enumFromInt(sequence)));
}

export fn sva_prop_implies(
    ir: *Ir,
    kind: Implication,
    antecedent: u32,
    consequent: u32,
    out: *u32,
) Status {
    const tag: Ir.Node.Tag = switch (kind) {
        .overlapping => .prop_implies_overlap,
        .nonoverlapping => .prop_implies_nonoverlap,
    };
    return deliver(out, ir.propImplies(gpa, tag, @enumFromInt(antecedent), @enumFromInt(consequent)));
}

const Junction = enum(c_int) { conjunction, disjunction };

/// `p and q` / `p or q` over two *properties*. Both operands must carry the same
/// strength; `sva_lower` reports `SVA_ERROR_MIXED_STRENGTH` if they do not.
export fn sva_prop_junction(ir: *Ir, kind: Junction, lhs: u32, rhs: u32, out: *u32) Status {
    const tag: Ir.Node.Tag = switch (kind) {
        .conjunction => .prop_and,
        .disjunction => .prop_or,
    };
    return deliver(out, ir.propJunction(gpa, tag, @enumFromInt(lhs), @enumFromInt(rhs)));
}

const Shift = enum(c_int) {
    /// The range read disjunctively: `nexttime[n]` at `lo == hi`, and
    /// `eventually [m:n]` wider.
    some,
    /// The range read conjunctively: `always [m:n]`.
    every,
};

/// A property shifted by `lo..hi` cycles. `SVA_UNBOUNDED` as `hi` spells the
/// unbounded `always p` / `s_eventually p`, which builds but does not lower.
export fn sva_prop_shift(ir: *Ir, kind: Shift, lo: u32, hi: u32, property: u32, out: *u32) Status {
    const tag: Ir.Node.Tag = switch (kind) {
        .some => .prop_delay,
        .every => .prop_always,
    };
    return deliver(out, ir.propRange(gpa, tag, lo, hi, @enumFromInt(property)));
}

/// `s #-# p` / `s #=# p`: *some* match of `antecedent` followed by `consequent`
/// holding. The existential dual of `sva_prop_implies` — an antecedent that never
/// matches makes an implication hold and makes this fail.
export fn sva_prop_followed_by(
    ir: *Ir,
    kind: Implication,
    antecedent: u32,
    consequent: u32,
    out: *u32,
) Status {
    const tag: Ir.Node.Tag = switch (kind) {
        .overlapping => .prop_followed_by_overlap,
        .nonoverlapping => .prop_followed_by_nonoverlap,
    };
    return deliver(out, ir.propFollowedBy(gpa, tag, @enumFromInt(antecedent), @enumFromInt(consequent)));
}

const Abort = enum(c_int) {
    /// `accept_on`: the condition holding discharges the property — it holds.
    accept,
    /// `reject_on`: the condition holding fails the property.
    reject,
    /// `sync_accept_on`: as `accept`, condition read from sampled values.
    sync_accept,
    /// `sync_reject_on`: as `reject`, condition read from sampled values.
    sync_reject,
};

/// An abort scoped to `property`. Unlike a directive's reset, which abandons an
/// attempt and leaves no verdict, an accept discharges it and a reject fails it.
///
/// The `sync_` kinds read `cond` from sampled values and the others from live
/// values, exactly as a reset does. Passing a term that matches the kind is the
/// caller's job; `sva_monitor_abort_sampled` reports which was asked for.
export fn sva_prop_abort(ir: *Ir, kind: Abort, cond: u32, property: u32, out: *u32) Status {
    const tag: Ir.Node.Tag = switch (kind) {
        .accept => .prop_accept_on,
        .reject => .prop_reject_on,
        .sync_accept => .prop_sync_accept_on,
        .sync_reject => .prop_sync_reject_on,
    };
    return deliver(out, ir.propAbort(gpa, tag, @enumFromInt(cond), @enumFromInt(property)));
}

// -- Directives --------------------------------------------------------------

const Kind = enum(c_int) { assert, assume, cover, restrict };

const Directive = extern struct {
    id: u32,
    kind: Kind,
    /// `SVA_CLOCK_NONE` when unclocked.
    clock: u32,
    /// A boolean node. An assertion that never resets uses a constant false.
    reset: u32,
    property: u32,
};

/// `out` receives the directive's index, for `sva_lower`.
export fn sva_ir_add(ir: *Ir, directive: *const Directive, out: *u32) Status {
    const index = ir.addAssertion(gpa, .{
        .id = @enumFromInt(directive.id),
        .kind = @enumFromInt(@intFromEnum(directive.kind)),
        .clock = @enumFromInt(directive.clock),
        .reset = @enumFromInt(directive.reset),
        .prop = @enumFromInt(directive.property),
    }) catch return .out_of_memory;
    out.* = @intFromEnum(index);
    return .ok;
}

// -- Monitors ----------------------------------------------------------------

const Options = extern struct {
    /// `0` selects the library default.
    state_limit: u32,
    depth_limit: u32,
};

const Diagnostic = extern struct { node: u32, size: u64 };

/// Lower one directive into `out`. `options` and `diagnostic` may be null, and
/// neither `out` nor `diagnostic` is written unless the call fails or succeeds
/// respectively.
export fn sva_lower(
    out: *Monitor,
    ir: *const Ir,
    directive: u32,
    options: ?*const Options,
    diagnostic: ?*Diagnostic,
) Status {
    const limits: Options = if (options) |given| given.* else .{ .state_limit = 0, .depth_limit = 0 };
    var reason: glushkov.Diagnostic = undefined;
    out.* = glushkov.assertion(gpa, ir, @enumFromInt(directive), .{
        .limit = if (limits.state_limit == 0) 4096 else limits.state_limit,
        .depth_limit = if (limits.depth_limit == 0) 256 else limits.depth_limit,
    }, &reason) catch |err| {
        if (diagnostic) |where| {
            where.* = .{ .node = @intFromEnum(reason.node), .size = reason.size };
        }
        return Status.of(err);
    };
    return .ok;
}

export fn sva_monitor_deinit(monitor: *Monitor) void {
    monitor.deinit(gpa);
}

export fn sva_monitor_id(monitor: *const Monitor) u32 {
    return @intFromEnum(monitor.id);
}

export fn sva_monitor_kind(monitor: *const Monitor) Kind {
    return @enumFromInt(@intFromEnum(monitor.kind));
}

export fn sva_monitor_clock(monitor: *const Monitor) u32 {
    return @intFromEnum(monitor.clock);
}

/// Obligation attempts that can be pending at once: the number of state vectors
/// a backend allocates.
export fn sva_monitor_depth(monitor: *const Monitor) u32 {
    return monitor.depth;
}

export fn sva_monitor_strength(monitor: *const Monitor) Strength {
    return @enumFromInt(@intFromEnum(monitor.strength));
}

/// Meaningless without a trigger.
export fn sva_monitor_implication(monitor: *const Monitor) Implication {
    return if (monitor.overlap) .overlapping else .nonoverlapping;
}

/// Root index of the reset condition. Always a real root: an assertion that never
/// resets has a constant false there.
export fn sva_monitor_reset(monitor: *const Monitor) u32 {
    return monitor.reset;
}

/// Aborts scoped to part of the obligation, and to the trigger when
/// `sva_monitor_abort_covers_trigger`. Zero for a property with none.
export fn sva_monitor_abort_count(monitor: *const Monitor) u32 {
    return @intCast(monitor.aborts.len);
}

/// Root index of one abort's condition.
export fn sva_monitor_abort_root(monitor: *const Monitor, abort: u32) u32 {
    return monitor.aborts[abort].cond;
}

/// What firing does to an attempt: `SVA_ABORT_ACCEPT` discharges it, which is a
/// pass a `cover` counts, and `SVA_ABORT_REJECT` fails it.
export fn sva_monitor_abort_effect(monitor: *const Monitor, abort: u32) Abort {
    return switch (monitor.aborts[abort].effect) {
        .accept => .accept,
        .reject => .reject,
    };
}

/// Whether firing resolves the whole attempt, or only kills the threads in the
/// abort's mask and leaves the rest of the attempt running.
export fn sva_monitor_abort_resolves_attempt(monitor: *const Monitor, abort: u32) bool {
    return monitor.aborts[abort].scope == .attempt;
}

/// Whether the abort covers the trigger as well as the obligation: it was
/// written *above* the implication rather than inside its consequent, so it also
/// covers the cycles the antecedent is being matched in. Always false for a
/// property with no trigger.
///
/// When it holds in a cycle an antecedent match ends, resolve that match with
/// the abort's effect and seed no attempt from it. That is a verdict the same
/// abort written inside the consequent never produces, because there it arms
/// only once the consequent starts — a cycle later, for `|=>`.
export fn sva_monitor_abort_covers_trigger(monitor: *const Monitor, abort: u32) bool {
    return monitor.aborts[abort].covers_trigger;
}

/// Whether the condition wants the *sampled* valuation rather than the live one:
/// true for `sync_accept_on`/`sync_reject_on`. Nothing in this library can check
/// that a backend honours it.
export fn sva_monitor_abort_sampled(monitor: *const Monitor, abort: u32) bool {
    return monitor.aborts[abort].sampled;
}

/// The obligation states an abort guards: `sva_automaton_words` of the obligation
/// long. An abort is armed for an attempt only while that attempt has a live
/// state in here.
export fn sva_monitor_abort_mask(monitor: *const Monitor, abort: u32) [*]const Word {
    return monitor.maskOf(abort).ptr;
}

/// The boolean nodes to evaluate once per cycle, one bit each, in this order.
export fn sva_monitor_root_count(monitor: *const Monitor) u32 {
    return @intCast(monitor.alphabet.roots.len);
}

export fn sva_monitor_roots(monitor: *const Monitor) [*]const u32 {
    return @ptrCast(monitor.alphabet.roots.ptr);
}

/// Interned guard cubes. Guard `0` is the empty cube, which always holds.
export fn sva_monitor_guard_count(monitor: *const Monitor) u32 {
    return @intCast(monitor.alphabet.guards.len);
}

/// The literals of one guard: a root index above bit 0, negated in bit 0. Writes
/// the count, and returns null for the empty cube.
export fn sva_monitor_guard(monitor: *const Monitor, guard: u32, count: *u32) ?[*]const u32 {
    const literals = monitor.alphabet.literalsOf(@enumFromInt(guard));
    count.* = @intCast(literals.len);
    return if (literals.len == 0) null else @ptrCast(literals.ptr);
}

export fn sva_monitor_obligation(monitor: *const Monitor) *const Nfa {
    return &monitor.obligation;
}

/// Null when the property is a bare sequence and an obligation starts every
/// cycle.
export fn sva_monitor_trigger(monitor: *const Monitor) ?*const Nfa {
    if (monitor.trigger) |*trigger| return trigger;
    return null;
}

// -- Automata ----------------------------------------------------------------

export fn sva_automaton_state_count(automaton: *const Nfa) u32 {
    return automaton.state_count;
}

/// Words in one state mask, and the row stride of `follow` and `guard_states`.
export fn sva_automaton_words(automaton: *const Nfa) u32 {
    return automaton.words;
}

/// One guard index per state.
export fn sva_automaton_guards(automaton: *const Nfa) [*]const u32 {
    return @ptrCast(automaton.guard_of.ptr);
}

/// The states enterable one cycle after `state`.
export fn sva_automaton_follow(automaton: *const Nfa, state: u32) [*]const Word {
    return automaton.followOf(state).ptr;
}

/// States a match may begin in.
export fn sva_automaton_first(automaton: *const Nfa) [*]const Word {
    return automaton.first.ptr;
}

/// States that complete a match.
export fn sva_automaton_last(automaton: *const Nfa) [*]const Word {
    return automaton.last.ptr;
}

/// The states carrying `guard`, so building the satisfied mask folds over guards
/// rather than gathering per state.
export fn sva_automaton_guard_states(automaton: *const Nfa, guard: u32) [*]const Word {
    return automaton.statesOf(@enumFromInt(guard)).ptr;
}

// -- Running -----------------------------------------------------------------

const Report = extern struct { failures: u32, matches: u32, reset: bool };

/// `monitor` must outlive the run and must not move.
export fn sva_run_init(run: *Monitor.State, monitor: *const Monitor) Status {
    run.* = Monitor.State.init(gpa, monitor) catch return .out_of_memory;
    return .ok;
}

export fn sva_run_deinit(run: *Monitor.State) void {
    run.deinit(gpa);
}

/// Abandon everything in flight, as a reset does.
export fn sva_run_clear(run: *Monitor.State) void {
    run.clear();
}

/// Advance one cycle. `root_bits` holds one bit per root, in the monitor's order,
/// packed into 64-bit words.
export fn sva_run_step(run: *Monitor.State, root_bits: [*]const Word) Report {
    const report = run.step(root_bits[0..run.monitor.alphabet.words()]);
    return .{ .failures = report.failures, .matches = report.matches, .reset = report.reset };
}

/// Resolve what is still pending because the trace ended.
export fn sva_run_finish(run: *Monitor.State) Report {
    const report = run.finish();
    return .{ .failures = report.failures, .matches = report.matches, .reset = report.reset };
}

// -- Contract ----------------------------------------------------------------

/// Sizes the header declares, with a little headroom so a field can be added
/// without breaking every caller's build. A struct that outgrows one fails the
/// build here, naming the size it needs, rather than corrupting a caller's stack.
const ir_size = 96;
const monitor_size = 288;
const run_size = 160;
const object_align = 8;

comptime {
    for ([_]struct { []const u8, usize, usize }{
        .{ "SVA_IR_SIZE", ir_size, @sizeOf(Ir) },
        .{ "SVA_MONITOR_SIZE", monitor_size, @sizeOf(Monitor) },
        .{ "SVA_RUN_SIZE", run_size, @sizeOf(Monitor.State) },
    }) |entry| {
        const name, const declared, const actual = entry;
        if (actual > declared) @compileError(std.fmt.comptimePrint(
            "{s} is {d} but {d} is needed",
            .{ name, declared, actual },
        ));
    }
    for ([_]type{ Ir, Monitor, Monitor.State }) |T| {
        std.debug.assert(@alignOf(T) <= object_align);
    }

    // The C enums are the Zig ones renumbered from zero, so a cast stands in for
    // a mapping — and a reordering breaks the build here instead of silently
    // relabelling a directive or a verdict.
    for (@typeInfo(Kind).@"enum".fields, @typeInfo(Ir.Assertion.Kind).@"enum".fields) |c, zig| {
        std.debug.assert(std.mem.eql(u8, c.name, zig.name) and c.value == zig.value);
    }
    for (@typeInfo(Strength).@"enum".fields, @typeInfo(Monitor.Strength).@"enum".fields) |c, zig| {
        std.debug.assert(std.mem.eql(u8, c.name, zig.name) and c.value == zig.value);
    }
    // `BoolOp` mirrors the boolean prefix of the tag enum, which is what lets
    // `sva_bool_read` cast a tag straight across.
    for (@typeInfo(BoolOp).@"enum".fields, 0..) |field, index| {
        const tag: Ir.Node.Tag = @enumFromInt(index);
        std.debug.assert(tag.layer() == .boolean and field.value == index);
    }
    std.debug.assert(@sizeOf(Nfa.Literal) == @sizeOf(u32));
    std.debug.assert(@sizeOf(Word) == 8);
}

// -- Tests -------------------------------------------------------------------

const testing = std.testing;

const clock_none: u32 = std.math.maxInt(u32);

/// Every builder can fail, so a test that is not about failure says so once.
fn must(status: Status) !void {
    try testing.expectEqual(Status.ok, status);
}

/// `##[lo:hi] right` from the start of an attempt: a one-cycle `true` to the
/// left, which is the only way to spell it now that no edge is optional.
fn leadingDelay(ir: *Ir, lo: u32, hi: u32, right: u32) !u32 {
    var always: u32 = undefined;
    var head: u32 = undefined;
    try must(sva_bool_const(ir, true, &always));
    try must(sva_seq_boolean(ir, always, &head));
    var node: u32 = undefined;
    try must(sva_seq_delay(ir, head, lo, hi, right, &node));
    return node;
}

/// A one-cycle sequence over an atom.
fn atomCycle(ir: *Ir, atom: u32) !u32 {
    var boolean: u32 = undefined;
    try must(sva_bool_atom(ir, atom, &boolean));
    var node: u32 = undefined;
    try must(sva_seq_boolean(ir, boolean, &node));
    return node;
}

/// An assertion that never resets, which is a constant false rather than an
/// absent edge.
fn addDirective(ir: *Ir, property: u32) !u32 {
    var never: u32 = undefined;
    try must(sva_bool_const(ir, false, &never));
    var index: u32 = undefined;
    try must(sva_ir_add(ir, &.{
        .id = 42,
        .kind = .assert,
        .clock = clock_none,
        .reset = never,
        .property = property,
    }, &index));
    return index;
}

/// The root index a monitor gave `atom`. A backend records this when it emits the
/// per-cycle root evaluation; a test needs it because the reset condition is a
/// root too, so bit positions are not the caller's to assume.
fn rootOfAtom(ir: *const Ir, monitor: *const Monitor, atom: u32) ?u32 {
    const roots = sva_monitor_roots(monitor);
    for (0..sva_monitor_root_count(monitor)) |index| {
        var view: BoolView = undefined;
        sva_bool_read(ir, roots[index], &view);
        if (view.op == .atom and view.lhs == atom) return @intCast(index);
    }
    return null;
}

test "the C API builds, lowers and runs an implication" {
    var ir: Ir = undefined;
    sva_ir_init(&ir);
    defer sva_ir_deinit(&ir);

    // req |=> ack
    const req = try atomCycle(&ir, 0);
    const ack = try atomCycle(&ir, 1);
    var consequent: u32 = undefined;
    try must(sva_prop_sequence(&ir, .weak, ack, &consequent));
    var property: u32 = undefined;
    try must(sva_prop_implies(&ir, .nonoverlapping, req, consequent, &property));
    const directive = try addDirective(&ir, property);

    var monitor: Monitor = undefined;
    try must(sva_lower(&monitor, &ir, directive, null, null));
    defer sva_monitor_deinit(&monitor);

    try testing.expectEqual(@as(u32, 42), sva_monitor_id(&monitor));
    try testing.expectEqual(Kind.assert, sva_monitor_kind(&monitor));
    try testing.expectEqual(Implication.nonoverlapping, sva_monitor_implication(&monitor));
    try testing.expectEqual(Strength.weak, sva_monitor_strength(&monitor));
    try testing.expectEqual(@as(u32, 1), sva_monitor_depth(&monitor));
    // Two atoms plus the constant false that says the assertion never resets.
    try testing.expectEqual(@as(u32, 3), sva_monitor_root_count(&monitor));
    try testing.expect(sva_monitor_reset(&monitor) < sva_monitor_root_count(&monitor));
    try testing.expectEqual(@as(u32, 1), sva_automaton_state_count(sva_monitor_obligation(&monitor)));
    try testing.expect(sva_monitor_trigger(&monitor) != null);

    // A root is a boolean expression, so not every one is an atom — the reset
    // here is a constant, and it occupies a bit like any other root.
    const order = [_]u32{
        rootOfAtom(&ir, &monitor, 0).?,
        rootOfAtom(&ir, &monitor, 1).?,
    };

    var run: Monitor.State = undefined;
    try must(sva_run_init(&run, &monitor));
    defer sva_run_deinit(&run);

    const trace = [_]struct { req: bool, ack: bool, failures: u32, matches: u32 }{
        .{ .req = true, .ack = false, .failures = 0, .matches = 0 },
        .{ .req = false, .ack = true, .failures = 0, .matches = 1 },
        .{ .req = true, .ack = false, .failures = 0, .matches = 0 },
        .{ .req = false, .ack = false, .failures = 1, .matches = 0 },
    };
    for (trace) |cycle| {
        var bits = [_]Word{0};
        if (cycle.req) bits[0] |= @as(Word, 1) << @intCast(order[0]);
        if (cycle.ack) bits[0] |= @as(Word, 1) << @intCast(order[1]);
        const report = sva_run_step(&run, &bits);
        try testing.expectEqual(cycle.failures, report.failures);
        try testing.expectEqual(cycle.matches, report.matches);
    }
    try testing.expectEqual(@as(u32, 0), sva_run_finish(&run).failures);
}

test "lowering failures name the node" {
    var ir: Ir = undefined;
    sva_ir_init(&ir);
    defer sva_ir_deinit(&ir);

    // req |=> ##[1:$] ack, whose obligation has no bounded longest match.
    const ack = try atomCycle(&ir, 1);
    const body = try leadingDelay(&ir, 1, std.math.maxInt(u32), ack);
    const req = try atomCycle(&ir, 0);
    var weak: u32 = undefined;
    try must(sva_prop_sequence(&ir, .weak, body, &weak));
    var unbounded: u32 = undefined;
    try must(sva_prop_implies(&ir, .nonoverlapping, req, weak, &unbounded));
    const first = try addDirective(&ir, unbounded);

    // And a wide delay range: many filler states, but a longest match under the
    // depth cap, which is checked first and would otherwise mask the state cap.
    const tail = try atomCycle(&ir, 1);
    var wide: u32 = undefined;
    try must(sva_seq_delay(&ir, req, 1, 100, tail, &wide));
    var bare: u32 = undefined;
    try must(sva_prop_sequence(&ir, .weak, wide, &bare));
    const second = try addDirective(&ir, bare);

    var monitor: Monitor = undefined;
    var reason: Diagnostic = undefined;
    try testing.expectEqual(
        Status.unbounded_obligation,
        sva_lower(&monitor, &ir, first, null, &reason),
    );
    try testing.expectEqual(body, reason.node);

    const limits: Options = .{ .state_limit = 64, .depth_limit = 0 };
    try testing.expectEqual(Status.state_limit, sva_lower(&monitor, &ir, second, &limits, &reason));
    try testing.expectEqual(wide, reason.node);
    // One state per cycle the delay may skip, plus the two matches.
    try testing.expectEqual(@as(u64, 101), reason.size);
}

test "guards are readable as cubes over the roots" {
    var ir: Ir = undefined;
    sva_ir_init(&ir);
    defer sva_ir_deinit(&ir);

    // `(a && b) ##1 a`, so one guard has two literals and the other has one.
    var a: u32 = undefined;
    var b: u32 = undefined;
    try must(sva_bool_atom(&ir, 0, &a));
    try must(sva_bool_atom(&ir, 1, &b));
    var conjunction: u32 = undefined;
    try must(sva_bool_and(&ir, a, b, &conjunction));
    var both: u32 = undefined;
    try must(sva_seq_boolean(&ir, conjunction, &both));
    const tail = try atomCycle(&ir, 0);
    var pattern: u32 = undefined;
    try must(sva_seq_delay(&ir, both, 1, 1, tail, &pattern));
    var property: u32 = undefined;
    try must(sva_prop_sequence(&ir, .weak, pattern, &property));
    const directive = try addDirective(&ir, property);

    var monitor: Monitor = undefined;
    try must(sva_lower(&monitor, &ir, directive, null, null));
    defer sva_monitor_deinit(&monitor);

    const automaton = sva_monitor_obligation(&monitor);
    try testing.expectEqual(@as(u32, 2), sva_automaton_state_count(automaton));

    const guards = sva_automaton_guards(automaton);
    var total: u32 = 0;
    for (0..sva_automaton_state_count(automaton)) |state| {
        var count: u32 = 0;
        const literals = sva_monitor_guard(&monitor, guards[state], &count);
        total += count;
        for (0..count) |index| {
            try testing.expect(literals.?[index] >> 1 < sva_monitor_root_count(&monitor));
        }
    }
    try testing.expectEqual(@as(u32, 3), total);

    // Guard zero is the empty cube.
    var count: u32 = 7;
    try testing.expect(sva_monitor_guard(&monitor, 0, &count) == null);
    try testing.expectEqual(@as(u32, 0), count);
}

test "a bare sequence property has no trigger" {
    var ir: Ir = undefined;
    sva_ir_init(&ir);
    defer sva_ir_deinit(&ir);

    const a = try atomCycle(&ir, 0);
    const b = try atomCycle(&ir, 0);
    var pattern: u32 = undefined;
    try must(sva_seq_delay(&ir, a, 1, 1, b, &pattern));
    var property: u32 = undefined;
    try must(sva_prop_sequence(&ir, .strong, pattern, &property));
    const directive = try addDirective(&ir, property);

    var monitor: Monitor = undefined;
    try must(sva_lower(&monitor, &ir, directive, null, null));
    defer sva_monitor_deinit(&monitor);
    try testing.expect(sva_monitor_trigger(&monitor) == null);
    try testing.expectEqual(Strength.strong, sva_monitor_strength(&monitor));
    try testing.expectEqual(@as(u32, 2), sva_monitor_depth(&monitor));
    // One atom plus the constant false reset.
    try testing.expectEqual(@as(u32, 2), sva_monitor_root_count(&monitor));

    var run: Monitor.State = undefined;
    try must(sva_run_init(&run, &monitor));
    defer sva_run_deinit(&run);

    // `a ##1 a` over a held `a`: an attempt completes every cycle from the second
    // on, and the newest is still pending at the end. Being strong, it fails. The
    // atom's bit has to be looked up, not assumed.
    const held = [_]Word{@as(Word, 1) << @intCast(rootOfAtom(&ir, &monitor, 0).?)};
    try testing.expectEqual(@as(u32, 0), sva_run_step(&run, &held).matches);
    try testing.expectEqual(@as(u32, 1), sva_run_step(&run, &held).matches);
    try testing.expectEqual(@as(u32, 1), sva_run_step(&run, &held).matches);
    try testing.expectEqual(@as(u32, 1), sva_run_finish(&run).failures);
}
