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
    /// The obligation's matches have no bounded length, so the number of
    /// attempts pending at once is unbounded too and no fixed ring of state
    /// vectors tracks them. Sound tracking needs the obligation determinised,
    /// which is not implemented.
    ObligationNotBounded,
    /// The consequent of an implication is itself an implication, so the
    /// obligation would have to spawn obligations of its own.
    UnsupportedConsequent,
    /// The obligation is bounded but deeper than `Options.depth_limit`.
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
    const alphabet = try builder.materialize(&.{&fragment}, &.{}, &automata);
    return .{ .alphabet = alphabet, .nfa = automata[0] };
}

/// Lower a directive to a `Monitor`.
///
/// The property layer is thin: an implication contributes a trigger automaton,
/// and the sequence underneath contributes the obligation. Both are built
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

    // Peel the implication, if there is one. What is left is the property whose
    // sequence becomes the obligation.
    var property = directive.prop;
    var trigger_node: ?Ir.Node.Index = null;
    var overlap = false;
    switch (ir.tagOf(property)) {
        .prop_implies_overlap, .prop_implies_nonoverlap => {
            const data = ir.dataOf(property);
            trigger_node = @as(Ir.Node.Index, @enumFromInt(data.lhs));
            overlap = ir.tagOf(property) == .prop_implies_overlap;
            property = @enumFromInt(data.rhs);
        },
        else => {},
    }
    const strength: Monitor.Strength = switch (ir.tagOf(property)) {
        .prop_seq_weak => .weak,
        .prop_seq_strong => .strong,
        else => {
            if (diagnostic) |into| into.* = .{ .node = property };
            return error.UnsupportedConsequent;
        },
    };
    const obligation_node: Ir.Node.Index = @enumFromInt(ir.dataOf(property).lhs);

    var builder: Builder = .{ .gpa = gpa, .ir = ir, .options = options, .diagnostic = diagnostic };
    defer builder.deinit();
    try builder.reserveCubes();

    // How many attempts can be in flight at once is the obligation's longest
    // match, so an obligation without one cannot be tracked by a fixed ring.
    const longest = maxLength(ir, obligation_node) orelse {
        if (diagnostic) |into| into.* = .{ .node = obligation_node };
        return error.ObligationNotBounded;
    };
    const depth = try builder.checkDepth(obligation_node, longest);

    var obligation = try builder.lower(obligation_node);
    defer obligation.deinit(gpa);

    var trigger: ?Fragment = null;
    defer if (trigger) |*fragment| fragment.deinit(gpa);
    if (trigger_node) |node| trigger = try builder.lower(node);

    // The reset condition is a root in its own right: a backend evaluates it
    // alongside the guards. It is interned last, so it lands past every root a
    // guard mentions, and pinned so materialisation keeps it there even though
    // no guard mentions it. An assertion that never resets has a constant false
    // here, which folds to nothing in any backend.
    var pinned = [_]u32{try builder.rootOf(directive.reset)};

    var fragments: [2]*const Fragment = undefined;
    var count: usize = 1;
    fragments[0] = &obligation;
    if (trigger) |*fragment| {
        fragments[1] = fragment;
        count = 2;
    }

    var automata: [2]Nfa = undefined;
    var alphabet = try builder.materialize(fragments[0..count], &pinned, automata[0..count]);
    errdefer alphabet.deinit(gpa);

    return .{
        .id = directive.id,
        .kind = directive.kind,
        .clock = directive.clock,
        .alphabet = alphabet,
        .trigger = if (count == 2) automata[1] else null,
        .overlap = overlap,
        .obligation = automata[0],
        .depth = depth,
        .strength = strength,
        .reset = pinned[0],
    };
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

        @memset(guard, .always);
        @memset(follow, 0);
        @memset(first, 0);
        @memset(last, 0);
        return .{
            .state_count = state_count,
            .words = words,
            .guard = guard,
            .follow = follow,
            .first = first,
            .last = last,
        };
    }

    fn deinit(fragment: *Fragment, gpa: Allocator) void {
        gpa.free(fragment.guard);
        gpa.free(fragment.follow);
        gpa.free(fragment.first);
        gpa.free(fragment.last);
    }

    fn row(fragment: *Fragment, state: u32) []Word {
        return fragment.follow[@as(usize, state) * fragment.words ..][0..fragment.words];
    }

    fn rowConst(fragment: *const Fragment, state: u32) []const Word {
        return fragment.follow[@as(usize, state) * fragment.words ..][0..fragment.words];
    }

    /// Copy `source`'s states and internal edges in starting at `base`. Entry
    /// and exit sets are the caller's business: a combinator rarely wants them
    /// verbatim.
    fn blit(fragment: *Fragment, source: *const Fragment, base: u32) void {
        for (0..source.state_count) |index| {
            const state: u32 = @intCast(index);
            fragment.guard[base + state] = source.guard[state];
            blitMask(fragment.row(base + state), source.rowConst(state), base);
        }
    }
};

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
            else => unreachable, // not a sequence-layer tag
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

    /// The determinised state one cell leads to, discovered if it is new.
    ///
    /// Null for a cell that leaves the operand nowhere to be: that is an
    /// attempt dying, which a position automaton spells as having no state to
    /// enter rather than as a state of its own.
    fn reach(
        builder: *Builder,
        node: Ir.Node.Index,
        operand: *const Fragment,
        into: *Determinisation,
        candidates: []const Word,
        cell: Guard.Index,
    ) Error!?u32 {
        const gpa = builder.gpa;

        @memset(into.subset, 0);
        var candidate: BitIterator = .init(candidates);
        while (candidate.next()) |state| {
            if (builder.entails(cell, operand.guard[state])) Nfa.bitSet(into.subset, state);
        }
        if (!Nfa.bitAny(into.subset)) return null;

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
    fn prune(builder: *Builder, fragment: *const Fragment) Allocator.Error!Surviving {
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
        for (live, fragment.last, enterable) |*word, exit, ok| word.* = exit & ok;
        try builder.closure(live, incoming, words, enterable);
        Nfa.bitIntersect(live, reachable);

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

    /// Turn one or more fragments into automata over a single alphabet.
    ///
    /// The guard table and the root set are compacted across *all* of them at
    /// once, which is the point: a monitor's two automata have to name the same
    /// term by the same index, or a backend evaluates it twice. `pinned` names
    /// root indices to keep even if no surviving guard mentions them, and is
    /// rewritten in place with their new indices.
    fn materialize(
        builder: *Builder,
        fragments: []const *const Fragment,
        pinned: []u32,
        out: []Nfa,
    ) Error!Nfa.Alphabet {
        const gpa = builder.gpa;
        std.debug.assert(fragments.len == out.len);
        std.debug.assert(fragments.len <= 2);

        var surviving: [2]Surviving = undefined;
        var pruned: usize = 0;
        defer for (surviving[0..pruned]) |*entry| entry.deinit(gpa);
        while (pruned < fragments.len) : (pruned += 1) {
            surviving[pruned] = try builder.prune(fragments[pruned]);
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
        const guard_states = try gpa.alloc(Word, @as(usize, guard_count) * words);
        @memset(follow, 0);
        @memset(first, 0);
        @memset(last, 0);
        @memset(guard_states, 0);

        for (0..fragment.state_count) |index| {
            const state = kept.map[index];
            if (state == dropped) continue;
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

        return .{
            .state_count = state_count,
            .words = words,
            .guard_of = guard_of,
            .follow = follow,
            .first = first,
            .last = last,
            .guard_states = guard_states,
        };
    }
};

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

/// Whether a match of `pattern` starting at `start` can still end in `cycle` or
/// later, over the trace so far plus some continuation. This is the definition
/// the automaton's "state set is non-empty" is supposed to implement, arrived at
/// by enumerating continuations instead.
fn canStillMatch(
    ir: *const Ir,
    pattern: Ir.Node.Index,
    trace: []const u64,
    start: u32,
    cycle: u32,
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
        // A match ending before `cycle` discharged the attempt already; only an
        // end in `cycle` or later keeps it alive.
        ends &= ~((@as(u64, 1) << @intCast(cycle)) - 1);
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
        if (!canStillMatch(ir, pattern, trace, start, cycle, atom_count, bound)) {
            return .{ .failed_at = cycle };
        }
    }
    return .{}; // still pending when the trace ran out
}

/// Check a monitor against the semantics over every trace of `length` cycles.
fn differentialMonitor(
    gpa: Allocator,
    ir: *const Ir,
    index: Ir.Assertion.Index,
    atom_count: u5,
    length: u32,
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
    const obligation: Ir.Node.Index = @enumFromInt(ir.dataOf(property).lhs);
    const strong = ir.tagOf(property) == .prop_seq_strong;
    const bound: u32 = @intCast(maxLength(ir, obligation).?);

    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    const cycles_mask = (@as(u64, 1) << @intCast(length)) - 1;

    var traces: Traces = .init(atom_count, length);
    while (traces.next()) |window| {

        // Which cycles an obligation starts in: every cycle without a trigger,
        // and otherwise wherever some antecedent match ends, shifted by the
        // implication's delay.
        var starts = cycles_mask;
        if (antecedent) |node| {
            var ends: u64 = 0;
            for (0..length) |start| {
                ends |= bruteForceMatches(ir, node, window, @intCast(start));
            }
            starts = (ends << @intCast(delay)) & cycles_mask;
        }

        var want_failures = [_]u32{0} ** 8;
        var want_matches = [_]u32{0} ** 8;
        var want_pending: u32 = 0;
        var remaining = starts;
        while (remaining != 0) {
            const start: u32 = @ctz(remaining);
            remaining &= remaining - 1;
            const fate = bruteForceFate(ir, obligation, window, start, atom_count, bound);
            if (fate.matched_at) |at| {
                want_matches[at] += 1;
            } else if (fate.failed_at) |at| {
                want_failures[at] += 1;
            } else want_pending += 1;
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
) !u32 {
    var monitor = try assertion(gpa, ir, index, .{}, null);
    defer monitor.deinit(gpa);
    var state = try Monitor.State.init(gpa, &monitor);
    defer state.deinit(gpa);

    state.clear();
    var failures: u32 = 0;
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
                try differentialMonitor(gpa, &ir, index, 2, 4);
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
                try differentialMonitor(gpa, &ir, index, 2, 4);
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
                try differentialMonitor(gpa, &ir, index, 2, 4);
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
                try differentialMonitor(gpa, &ir, index, 2, 5);
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
                try differentialMonitor(gpa, &ir, index, 2, 5);
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
                try differentialMonitor(gpa, &ir, index, 2, 5);
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
                try differentialMonitor(gpa, &ir, index, 2, 4);
            }
        }
    }
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
            try differentialMonitor(gpa, &ir, index, 2, 4);
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
            try differentialMonitor(gpa, &ir, index, 1, 5);
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
        try differentialMonitor(gpa, &ir, index, 2, 4);
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
        try differentialMonitor(gpa, &ir, index, 2, 4);
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
        try differentialMonitor(gpa, &ir, index, 2, 4);
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
        var verdict: [2]u32 = undefined;
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
    var raised: [2]u32 = undefined;
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
        // does not bound how late it arrives, so this is still unbounded and
        // still rejected.
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
        var diagnostic: Diagnostic = undefined;
        try testing.expectError(
            error.ObligationNotBounded,
            assertion(gpa, &ir, try addDirective(&ir, gpa, property, null), .{}, &diagnostic),
        );
        try testing.expectEqual(body, diagnostic.node);
    }
    {
        // The conservative side of inheriting, pinned so that tightening it
        // is a visible change: `first_match(a[*1:$])` is just `a`, and a
        // longest path over the automaton would say so — but `maxLength` reads
        // the tree, so the answer is a refusal rather than a depth of one.
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
        try testing.expectError(
            error.ObligationNotBounded,
            assertion(gpa, &ir, try addDirective(&ir, gpa, property, null), .{}, null),
        );

        // The sequence itself lowers fine, and to exactly `a`: one state,
        // which is the shortening the bound declines to notice.
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
            try differentialMonitor(gpa, &ir, index, 2, 4);
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
            try differentialMonitor(gpa, &ir, index, 2, 4);
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

test "an obligation without a bounded length is rejected" {
    const gpa = testing.allocator;

    // `req |=> ##[1:$] ack`. However many attempts are pending, one more can
    // always arrive, so no fixed ring of state vectors tracks them and the
    // sound alternative is a determinised obligation.
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

    var diagnostic: Diagnostic = undefined;
    try testing.expectError(
        error.ObligationNotBounded,
        assertion(gpa, &ir, index, .{}, &diagnostic),
    );
    try testing.expectEqual(body, diagnostic.node);

    // Goto repetition is unbounded for the same reason: the gaps between
    // occurrences are, however few occurrences are asked for.
    var counted: Ir = .init();
    defer counted.deinit(gpa);
    const marks = try counted.seqRepeat(
        gpa,
        .seq_repeat_goto,
        try atomCycle(&counted, gpa, 1),
        2,
        2,
    );
    const marked = try counted.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&counted, gpa, 0),
        try counted.propSeq(gpa, .prop_seq_weak, marks),
    );
    try testing.expectError(
        error.ObligationNotBounded,
        assertion(gpa, &counted, try addDirective(&counted, gpa, marked, null), .{}, null),
    );

    // An `and` is unbounded as soon as *either* side is, where an intersection
    // survives one unbounded side: an intersection's two ends coincide, so the
    // bounded side bounds the pair, but an `and` runs on to the later end.
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
    try testing.expectError(
        error.ObligationNotBounded,
        assertion(gpa, &waited, try addDirective(&waited, gpa, waits, null), .{}, null),
    );

    // The contrast, so that asymmetry is pinned rather than assumed.
    const met = try waited.seqIntersect(gpa, open, try atomCycle(&waited, gpa, 2));
    const meets = try waited.propImplies(
        gpa,
        .prop_implies_nonoverlap,
        try atomCycle(&waited, gpa, 0),
        try waited.propSeq(gpa, .prop_seq_weak, met),
    );
    var monitor = try assertion(gpa, &waited, try addDirective(&waited, gpa, meets, null), .{}, null);
    defer monitor.deinit(gpa);
    try testing.expectEqual(@as(u32, 1), monitor.depth);
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

test {
    testing.refAllDecls(@This());
}
