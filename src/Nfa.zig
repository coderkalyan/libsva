//! The automaton a sequence lowers to, and the primitives that run it.
//!
//! # Shape
//!
//! A **position automaton**: one state per letter occurrence in the sequence,
//! no ε-transitions, and *homogeneous* — every transition entering a state
//! carries the same guard, so a guard belongs to the state rather than to an
//! edge. There is no start state either; `first` is the mask of states a match
//! may begin in, and `last` the mask of states that complete one.
//!
//! That shape exists to make the step cheap. A cycle is:
//!
//! ```
//! next = (⋃ follow[state] for state in current) & satisfied   // advance
//! next |= first & satisfied                                   // seed
//! matched = next & last
//! ```
//!
//! Three bit-set passes and no edge iteration. Seeding an attempt every cycle —
//! the implicit quantification every assertion carries — is one `or`, and a
//! reset is `current = 0`.
//!
//! # The alphabet
//!
//! Guards live in an `Alphabet`, not in the automaton, because a `Monitor` runs
//! two automata over one set of terms: `req |=> ack` would otherwise evaluate
//! `req` once for the antecedent and again for the consequent. An `Nfa` names
//! guards by index and is only meaningful against the alphabet it was built
//! with.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ir = @import("Ir.zig");

const Nfa = @This();

/// How many states there are. State indices are `0..state_count - 1`.
state_count: u32,
/// Words in one state mask, `wordsFor(state_count)`. The row stride of `follow`
/// and of `guard_states`.
words: u32,
/// The guard that must hold for a state to be entered. Indexed by state.
guard_of: []Guard.Index,
/// `follow[state]` is the mask of states enterable one cycle after `state`. A
/// flat `state_count * words` matrix; use `followOf`.
follow: []Word,
/// States a match may begin in.
first: []Word,
/// States that complete a match.
last: []Word,
/// States an obligation attempt has *failed* in, for a monitor that tracks its
/// attempts by counter rather than by state vector. Empty — no bit set — for
/// every other automaton, where failure is the state vector running empty.
///
/// Only a determinised obligation has one. There, an attempt is a single state
/// rather than a set, so "no live states" is not available as a failure signal:
/// the construction is total and every valuation leads somewhere. What it leads
/// to when nothing is possible any more is one of these. See
/// `Glushkov.determinise`.
dead: []Word,
/// Per guard of the owning alphabet, the mask of states carrying it. A flat
/// `alphabet.guards.len * words` matrix; use `statesOf`. This is what lets
/// `satisfy` fold over guards instead of gathering per state.
guard_states: []Word,

pub fn deinit(nfa: *Nfa, gpa: Allocator) void {
    gpa.free(nfa.guard_of);
    gpa.free(nfa.follow);
    gpa.free(nfa.first);
    gpa.free(nfa.last);
    gpa.free(nfa.guard_states);
    gpa.free(nfa.dead);
}

// -- The alphabet ------------------------------------------------------------

/// The terms an automaton's guards are written over, and the interned guards
/// themselves. Shared by every automaton of one monitor.
pub const Alphabet = struct {
    /// The boolean-layer nodes a backend evaluates once per cycle. A
    /// `Literal.root` indexes this.
    roots: []Ir.Node.Index,
    /// Interned cubes. Slots `always` and `never` are always present.
    guards: []Guard,
    /// Backing pool the cubes slice into.
    literals: []Literal,

    pub fn deinit(alphabet: *Alphabet, gpa: Allocator) void {
        gpa.free(alphabet.roots);
        gpa.free(alphabet.guards);
        gpa.free(alphabet.literals);
    }

    /// The cube `guard` stands for.
    pub fn literalsOf(alphabet: *const Alphabet, guard: Guard.Index) []const Literal {
        const cube = alphabet.guards[@intFromEnum(guard)];
        return alphabet.literals[cube.start..][0..cube.len];
    }

    /// Whether `guard` holds, given a bit per entry of `roots`.
    pub fn holds(alphabet: *const Alphabet, guard: Guard.Index, root_bits: []const Word) bool {
        if (guard == .never) return false;
        for (alphabet.literalsOf(guard)) |literal| {
            if (bitIsSet(root_bits, literal.root) == literal.negated) return false;
        }
        return true;
    }

    /// Words in a `roots` valuation.
    pub fn words(alphabet: *const Alphabet) u32 {
        return wordsFor(@intCast(alphabet.roots.len));
    }

    /// Evaluate every root against an atom valuation. `out` is `words()` long
    /// and is overwritten. A backend emits this itself — that is the whole point
    /// of the atom being opaque — so this exists to run an automaton in a test.
    pub fn evaluate(
        alphabet: *const Alphabet,
        ir: *const Ir,
        atoms: []const Word,
        out: []Word,
    ) void {
        @memset(out, 0);
        for (alphabet.roots, 0..) |root, index| {
            if (evaluateBoolean(ir, root, atoms)) bitSet(out, @intCast(index));
        }
    }
};

/// Evaluate a boolean-layer node against an atom valuation.
pub fn evaluateBoolean(ir: *const Ir, node: Ir.Node.Index, atoms: []const Word) bool {
    const data = ir.dataOf(node);
    return switch (ir.tagOf(node)) {
        .bool_const => data.lhs != 0,
        .atom => bitIsSet(atoms, data.lhs),
        .bnot => !evaluateBoolean(ir, @enumFromInt(data.lhs), atoms),
        .band => evaluateBoolean(ir, @enumFromInt(data.lhs), atoms) and
            evaluateBoolean(ir, @enumFromInt(data.rhs), atoms),
        .bor => evaluateBoolean(ir, @enumFromInt(data.lhs), atoms) or
            evaluateBoolean(ir, @enumFromInt(data.rhs), atoms),
        else => unreachable, // not a boolean-layer tag
    };
}

// -- Guards ------------------------------------------------------------------

/// One conjunct of a guard: a boolean root, possibly complemented.
pub const Literal = packed struct(u32) {
    /// Whether the root is complemented.
    negated: bool,
    /// Index into `Alphabet.roots`.
    root: u31,
};

/// A cube: the conjunction of `Alphabet.literals[start..start + len]`. The empty
/// conjunction is `true`.
pub const Guard = extern struct {
    /// Offset into `Alphabet.literals`.
    start: u32,
    /// How many literals the cube has.
    len: u32,

    pub const Index = enum(u32) {
        /// The empty cube — holds every cycle. Filler states carry it.
        always = 0,
        /// The unsatisfiable cube. No surviving state carries it; the slot is
        /// reserved so the construction can name a dead guard.
        never = 1,
        _,
    };
};

// -- Stepping ----------------------------------------------------------------

/// The states enterable one cycle after `state`.
pub fn followOf(nfa: *const Nfa, state: u32) []const Word {
    return nfa.follow[@as(usize, state) * nfa.words ..][0..nfa.words];
}

/// The states carrying `guard`.
pub fn statesOf(nfa: *const Nfa, guard: Guard.Index) []const Word {
    return nfa.guard_states[@as(usize, @intFromEnum(guard)) * nfa.words ..][0..nfa.words];
}

/// The states whose guard holds this cycle, given a `roots` valuation. `out` is
/// `words` long and is overwritten.
pub fn satisfy(
    nfa: *const Nfa,
    alphabet: *const Alphabet,
    root_bits: []const Word,
    out: []Word,
) void {
    @memset(out, 0);
    for (0..alphabet.guards.len) |index| {
        const guard: Guard.Index = @enumFromInt(index);
        if (!alphabet.holds(guard, root_bits)) continue;
        bitUnion(out, nfa.statesOf(guard));
    }
}

/// `next = (⋃ follow[state] for state in current) & satisfied`. `next` is
/// overwritten and may not alias `current`.
pub fn advance(nfa: *const Nfa, current: []const Word, satisfied: []const Word, next: []Word) void {
    @memset(next, 0);
    var states: BitIterator = .init(current);
    while (states.next()) |state| bitUnion(next, nfa.followOf(state));
    bitIntersect(next, satisfied);
}

/// Start a fresh attempt in this cycle: `next |= first & satisfied`.
pub fn seed(nfa: *const Nfa, satisfied: []const Word, next: []Word) void {
    for (next, nfa.first, satisfied) |*word, entry, holding| word.* |= entry & holding;
}

/// Whether `current` holds a state that completes a match.
pub fn matched(nfa: *const Nfa, current: []const Word) bool {
    return bitOverlaps(current, nfa.last);
}

/// The lowest index set in both masks, which is how a *counted* monitor picks
/// the one state an attempt moves into. See `chosenFollow`.
pub fn lowestBoth(left: []const Word, right: []const Word) ?u32 {
    for (left, right, 0..) |x, y, index| {
        const both = x & y;
        if (both != 0) return @intCast(index * word_bits + @ctz(both));
    }
    return null;
}

/// The single state a counted attempt sitting in `state` moves to: the
/// lowest-indexed successor whose guard holds this cycle.
///
/// A counted attempt is *one* state, not a set, so exactly one successor may
/// take it. Several can be enabled at once even though the obligation is
/// determinised — a determinised state is a (subset, cell) pair and two cells
/// may share a valuation — but such siblings carry the same subset, and so the
/// same exits and the same successors. They are bisimilar, which is why taking
/// the lowest is a choice and not an approximation. Handing the count to every
/// enabled sibling instead would manufacture attempts out of nothing.
///
/// Never null for a determinised obligation: the construction is total, so some
/// cell holds this cycle's valuation. Null is reachable only if a caller runs
/// this against an automaton that is not one.
pub fn chosenFollow(nfa: *const Nfa, state: u32, satisfied: []const Word) ?u32 {
    return lowestBoth(nfa.followOf(state), satisfied);
}

/// The single state a counted attempt seeded this cycle starts in, by the same
/// rule as `chosenFollow`.
pub fn chosenSeed(nfa: *const Nfa, satisfied: []const Word) ?u32 {
    return lowestBoth(nfa.first, satisfied);
}

/// Whether an attempt sitting in `state` has failed: a counted monitor's
/// replacement for the state vector running empty.
pub fn isDead(nfa: *const Nfa, state: u32) bool {
    return bitIsSet(nfa.dead, state);
}

// -- Bit sets ----------------------------------------------------------------
//
// A mask is a `[]Word` of `wordsFor(bits)` words, least significant bit of word
// zero first. Shared by the automaton and by the fragments it is built from, so
// there is one representation from construction through to the backend.

/// Backing word of every mask.
pub const Word = u64;
pub const word_bits: u32 = @bitSizeOf(Word);

/// Words needed to hold `bits` bits.
pub fn wordsFor(bits: u32) u32 {
    return (bits + word_bits - 1) / word_bits;
}

pub fn bitSet(bits: []Word, index: u32) void {
    bits[index / word_bits] |= @as(Word, 1) << @intCast(index % word_bits);
}

pub fn bitIsSet(bits: []const Word, index: u32) bool {
    return bits[index / word_bits] & (@as(Word, 1) << @intCast(index % word_bits)) != 0;
}

pub fn bitUnion(destination: []Word, source: []const Word) void {
    for (destination, source) |*word, add| word.* |= add;
}

pub fn bitIntersect(destination: []Word, source: []const Word) void {
    for (destination, source) |*word, keep| word.* &= keep;
}

/// Clear every bit of `destination` that is set in `source`.
pub fn bitSubtract(destination: []Word, source: []const Word) void {
    for (destination, source) |*word, drop| word.* &= ~drop;
}

pub fn bitAny(bits: []const Word) bool {
    for (bits) |word| if (word != 0) return true;
    return false;
}

/// How many bits are set: the size of the set a mask stands for. What a monitor
/// counting attempts by identity reads a count out of — see
/// `Monitor.State.idCount`.
pub fn bitCount(bits: []const Word) u64 {
    var total: u64 = 0;
    for (bits) |word| total += @popCount(word);
    return total;
}

pub fn bitOverlaps(left: []const Word, right: []const Word) bool {
    for (left, right) |x, y| if (x & y != 0) return true;
    return false;
}

/// Iterates the set bits of a mask, lowest index first.
pub const BitIterator = struct {
    bits: []const Word,
    word: usize,
    remaining: Word,

    pub fn init(bits: []const Word) BitIterator {
        return .{ .bits = bits, .word = 0, .remaining = if (bits.len > 0) bits[0] else 0 };
    }

    pub fn next(iterator: *BitIterator) ?u32 {
        while (iterator.remaining == 0) {
            iterator.word += 1;
            if (iterator.word >= iterator.bits.len) return null;
            iterator.remaining = iterator.bits[iterator.word];
        }
        const bit = @ctz(iterator.remaining);
        iterator.remaining &= iterator.remaining - 1;
        return @intCast(iterator.word * word_bits + bit);
    }
};

comptime {
    std.debug.assert(@sizeOf(Literal) == 4);
    std.debug.assert(@sizeOf(Guard) == 8);
}

test "bit iteration walks set bits in order" {
    var bits = [_]Word{ 0, 0 };
    bitSet(&bits, 0);
    bitSet(&bits, 63);
    bitSet(&bits, 64);
    bitSet(&bits, 127);

    var across: BitIterator = .init(&bits);
    try std.testing.expectEqual(@as(?u32, 0), across.next());
    try std.testing.expectEqual(@as(?u32, 63), across.next());
    try std.testing.expectEqual(@as(?u32, 64), across.next());
    try std.testing.expectEqual(@as(?u32, 127), across.next());
    try std.testing.expectEqual(@as(?u32, null), across.next());

    var empty = [_]Word{0};
    var over_zero: BitIterator = .init(&empty);
    try std.testing.expectEqual(@as(?u32, null), over_zero.next());

    var over_nothing: BitIterator = .init(&.{});
    try std.testing.expectEqual(@as(?u32, null), over_nothing.next());
}
