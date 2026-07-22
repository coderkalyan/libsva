//! libsva intermediate representation.
//!
//! The IR is a *flattened tree*, the same data-oriented layout `libcrv` uses
//! for constraints: instead of allocating expression nodes on the heap and
//! linking them with pointers, every node lives in one contiguous array and
//! refers to other nodes by index. It buys the rest of the library three
//! things:
//!
//!   * **Cache locality** — lowering an assertion to an automaton is a linear
//!     scan over packed arrays, not a pointer chase.
//!   * **Cheap content hashing** — the whole IR is plain POD arrays, so a hash
//!     is just Blake3 over their bytes (see `hash`). Two structurally identical
//!     assertion sets hash identically, which is what lets the NFA/backend
//!     pipeline cache compiled monitors keyed on the query.
//!   * **Cheap serialization** — `serialize`/`deserialize` blit the arrays to
//!     and from a byte buffer behind a small versioned, checksummed header. No
//!     pointer fix-ups.
//!
//! Where `libcrv` models a single layer (scalar constraint expressions), an
//! assertion language is layered, and the node array holds all three layers
//! back to back, distinguished by `Node.Tag`:
//!
//!   1. **Boolean layer** — the "letters" of the automaton alphabet: sampled
//!      boolean expressions over declared signals (`signal_ref`, `b_and`,
//!      `b_or`, `b_not`, `b_xor`, `bool_const`). Each maximal boolean node is
//!      the guard on an NFA transition.
//!   2. **Sequence layer** (SEREs) — regular temporal patterns over cycles:
//!      cycle-delay concatenation (`##[lo:hi]`), `or`, `intersect`, and
//!      consecutive/goto/non-consecutive repetition.
//!   3. **Property layer** — implication (`|->`, `|=>`), boolean property
//!      combinators, and the temporal operators (`always`, `s_eventually`,
//!      `until`, `nexttime`).
//!
//! Storage mirrors `libcrv`: a few flat side tables.
//!
//!   * `nodes`      — the boolean/sequence/property tree (struct-of-arrays).
//!   * `signals`    — declared signals the boolean layer references.
//!   * `assertions` — top-level directives (`assert`/`assume`/`cover`/
//!                    `restrict`), each naming a clock, an optional
//!                    `disable iff`, and a root property node.
//!   * `extra`      — `u32` payload pool for nodes with a variable or
//!                    multi-field shape (delay ranges, repetition counts).
//!
//! Indices into these tables are distinct `enum(u32)` types (`Node.Index`,
//! `Signal.Index`, ...) so the compiler catches an index used against the
//! wrong table. Where a node edge may be absent (e.g. a leading `##1 s` with no
//! left operand, or a missing clock) `Node.Index` carries a `null` sentinel
//! variant (`maxInt(u32)`).
//!
//! Signals and assertions carry an opaque `u32` id (`Signal.Id`,
//! `Assertion.Id`), not stored text: the IR holds no string bytes at all. An id
//! is only meaningful within the IR instance that produced it — resolve it
//! against whatever symbol table the front-end keeps on the side. This keeps
//! the IR purely numeric, so hashing and caching never have to canonicalize
//! strings.
//!
//! Scope: this is an *architectural* skeleton aimed at the concurrent-assertion
//! subset. The node tags stake out the full language, but the NFA lowering and
//! backends implement a core subset and reject the rest cleanly (see
//! `Nfa.zig`); the roadmap is called out per tag below.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

const Ir = @This();

/// Boolean/sequence/property nodes, stored struct-of-arrays for locality.
nodes: std.MultiArrayList(Node) = .{},
/// Declared signals, referenced by `.signal_ref` boolean nodes.
signals: std.MultiArrayList(Signal) = .{},
/// Top-level assertion directives; each names a root property.
assertions: std.MultiArrayList(Assertion) = .{},
/// Side pool of `u32` payloads for nodes with variable/multi-field shape.
extra: std.ArrayListUnmanaged(u32) = .empty,

pub fn deinit(ir: *Ir, gpa: Allocator) void {
    ir.nodes.deinit(gpa);
    ir.signals.deinit(gpa);
    ir.assertions.deinit(gpa);
    ir.extra.deinit(gpa);
    ir.* = undefined;
}

// -- Nodes -------------------------------------------------------------------

/// One entry in the flattened tree. `tag` selects how `data` is interpreted;
/// see `Tag` for the per-tag encoding. Kept to `tag` + two `u32`s so the hot
/// path stays small and the whole thing is bit-blittable.
pub const Node = struct {
    tag: Tag,
    data: Data,

    /// Index into `Ir.nodes`. The `null` variant is a sentinel for an absent
    /// edge (e.g. a leading cycle delay with no left operand).
    pub const Index = enum(u32) {
        null = std.math.maxInt(u32),
        _,

        pub fn unwrap(i: Index) ?Index {
            return if (i == .null) null else i;
        }
    };

    /// `extern` so the field order/layout is fixed for hashing and on-disk use.
    /// Fields are a raw payload whose meaning depends on `tag`.
    pub const Data = extern struct {
        lhs: u32 = 0,
        rhs: u32 = 0,
    };

    pub const Tag = enum(u8) {
        // == Boolean layer =====================================================
        // The alphabet. These evaluate to a single bit under a signal
        // valuation; a maximal boolean node is what labels an NFA transition.

        /// Boolean constant. `lhs` is 0 (false) or 1 (true).
        bool_const,
        /// Reference to a declared signal, read as a 1-bit sampled value.
        /// `lhs` = `Signal.Index`.
        signal_ref,
        /// Logical negation (`!a`). `lhs` = operand boolean node.
        b_not,
        /// Logical and (`a && b`). `lhs`, `rhs` = boolean nodes.
        b_and,
        /// Logical or (`a || b`). `lhs`, `rhs` = boolean nodes.
        b_or,
        /// Logical xor (`a ^ b`). `lhs`, `rhs` = boolean nodes.
        b_xor,

        // == Sequence layer (SEREs) ===========================================
        // Regular patterns over a finite run of cycles. A boolean is lifted
        // into a one-cycle sequence with `seq_boolean`.

        /// A boolean used as a length-1 sequence: matches a single cycle in
        /// which the boolean holds. `lhs` = boolean node.
        seq_boolean,
        /// Cycle-delay concatenation `a ##[lo:hi] b`. `lhs` = left sequence
        /// (`.null` for a leading `##[lo:hi] b`), `rhs` = `extra` index of
        /// `[right, lo, hi]`: `right` is the right `Node.Index`, `lo`/`hi` the
        /// delay bounds. `hi == unbounded` encodes `$` (`##[lo:$]`). `lo == 0`
        /// is overlap (`##0`, the left and right cycles coincide).
        seq_delay,
        /// Sequence disjunction `a or b`: matches if either matches. `lhs`,
        /// `rhs` = sequence nodes.
        seq_or,
        /// Sequence conjunction `a and b`: both match from the same start; the
        /// match ends at the later of the two end points. `lhs`, `rhs` =
        /// sequence nodes. Roadmap in the NFA lowering.
        seq_and,
        /// Length-matching intersection `a intersect b`: both match from the
        /// same start *and* the same end. `lhs`, `rhs` = sequence nodes.
        seq_intersect,
        /// Repetition `a[*lo:hi]` / `a[->lo:hi]` / `a[=lo:hi]`. `lhs` = the
        /// repeated sequence, `rhs` = `extra` index of `[lo, hi, mode]` where
        /// `mode` is a `Repeat.Mode` and `hi == unbounded` encodes `$`.
        seq_repeat,

        // == Property layer ===================================================
        // Properties may hold over an unbounded trace; only some are regular.

        /// A sequence promoted to a property (matches ⇒ holds). `lhs` =
        /// sequence node.
        prop_seq,
        /// Property negation `not p`. `lhs` = property node. Roadmap.
        prop_not,
        /// Property conjunction `p and q`. `lhs`, `rhs` = property nodes.
        /// Roadmap.
        prop_and,
        /// Property disjunction `p or q`. `lhs`, `rhs` = property nodes.
        /// Roadmap.
        prop_or,
        /// Overlapping implication `s |-> p`: on every match of antecedent
        /// sequence `s`, property `p` must hold starting at the match's last
        /// cycle. `lhs` = antecedent sequence, `rhs` = consequent property.
        prop_implies_overlap,
        /// Non-overlapping implication `s |=> p`: like `|->` but `p` starts the
        /// cycle after the match. `lhs` = antecedent sequence, `rhs` =
        /// consequent property.
        prop_implies_nonoverlap,
        /// `always p`. `lhs` = property node. Roadmap.
        prop_always,
        /// `s_eventually p` (strong). `lhs` = property node. Roadmap.
        prop_s_eventually,
        /// `nexttime p`. `lhs` = property node. Roadmap.
        prop_nexttime,
        /// `p until q` (weak). `lhs`, `rhs` = property nodes. Roadmap.
        prop_until,
        /// `p s_until q` (strong). `lhs`, `rhs` = property nodes. Roadmap.
        prop_s_until,
    };
};

/// Repetition flavor for `seq_repeat`.
pub const Repeat = struct {
    pub const Mode = enum(u32) {
        /// `[*lo:hi]` — consecutive repetition.
        consecutive,
        /// `[->lo:hi]` — goto: the sub-sequence holds `lo..hi` times, not
        /// necessarily consecutively, ending on the last occurrence.
        goto,
        /// `[=lo:hi]` — non-consecutive repetition (like goto but the match may
        /// extend past the last occurrence).
        non_consecutive,
    };
};

/// Sentinel for the open upper bound `$` in a delay or repetition range.
pub const unbounded: u32 = std.math.maxInt(u32);

// -- Signals -----------------------------------------------------------------

/// A declared signal the boolean layer samples. Signedness/width is carried for
/// completeness; the boolean layer reads a signal as a single bit.
pub const Signal = struct {
    /// Opaque caller-assigned id (see `Id`).
    id: Id,
    /// Bit-vector width. `1` for a plain boolean signal.
    ty: Type,

    /// Index into `Ir.signals`.
    pub const Index = enum(u32) { _ };

    /// Opaque per-IR identity handle the caller assigns; resolve it against the
    /// front-end's own symbol table. The IR stores no text.
    pub const Id = enum(u32) { _ };
};

/// A value's type is just its bit-vector width, exactly as in `libcrv`. A
/// `packed struct(u32)` so it is one word and bit-blittable.
pub const Type = packed struct(u32) {
    /// Bit width (`1..=65535`). `0` is reserved for "unspecified".
    width: u16,
    _reserved: u16 = 0,

    /// A vector of the given width.
    pub fn bit(width: u16) Type {
        return .{ .width = width };
    }
};

// -- Assertions --------------------------------------------------------------

/// A top-level assertion directive: a clocked property plus its disposition.
pub const Assertion = struct {
    /// Opaque caller-assigned id (see `Id`).
    id: Id,
    kind: Kind,
    flags: Flags,
    /// Boolean node naming the clock the property samples on (informational for
    /// the runtime backend, which is stepped once per clock; emitted by the
    /// formal backend). `.null` if unclocked.
    clock: Node.Index,
    /// `disable iff (b)` reset condition — a boolean node. When it holds, all
    /// in-flight obligations are abandoned. `.null` if absent.
    disable: Node.Index,
    /// The root property node.
    prop: Node.Index,

    /// Index into `Ir.assertions`.
    pub const Index = enum(u32) { _ };

    /// Opaque per-IR identity handle the caller assigns; resolve it against the
    /// front-end's own symbol table. The IR stores no text.
    pub const Id = enum(u32) { _ };

    pub const Kind = enum(u8) {
        /// `assert property` — the property must hold; a failure is an error.
        assert,
        /// `assume property` — a constraint on the environment (formal).
        assume,
        /// `cover property` — coverage: report when the property matches.
        cover,
        /// `restrict property` — a formal-only environment restriction.
        restrict,
    };

    pub const Flags = packed struct(u8) {
        _reserved: u8 = 0,
    };
};

// -- Index helpers -----------------------------------------------------------

pub const Extra = struct {
    /// Element offset into `Ir.extra`.
    pub const Index = enum(u32) { _ };
};

// -- Construction ------------------------------------------------------------
//
// Low-level builders the front-end sits on top of. They append to the flat
// arrays and hand back typed indices; nothing here inspects or validates the
// tree — that is the job of the lowering passes.

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

pub fn addSignal(ir: *Ir, gpa: Allocator, s: Signal) Allocator.Error!Signal.Index {
    const index: u32 = @intCast(ir.signals.len);
    try ir.signals.append(gpa, s);
    return @enumFromInt(index);
}

pub fn addAssertion(ir: *Ir, gpa: Allocator, a: Assertion) Allocator.Error!Assertion.Index {
    const index: u32 = @intCast(ir.assertions.len);
    try ir.assertions.append(gpa, a);
    return @enumFromInt(index);
}

// -- Convenience node constructors: boolean layer ----------------------------

pub fn boolConst(ir: *Ir, gpa: Allocator, value: bool) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .bool_const, .data = .{ .lhs = @intFromBool(value) } });
}

pub fn signalRef(ir: *Ir, gpa: Allocator, s: Signal.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .signal_ref, .data = .{ .lhs = @intFromEnum(s) } });
}

pub fn bNot(ir: *Ir, gpa: Allocator, operand: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .b_not, .data = .{ .lhs = @intFromEnum(operand) } });
}

pub fn bBinary(ir: *Ir, gpa: Allocator, tag: Node.Tag, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    std.debug.assert(tag == .b_and or tag == .b_or or tag == .b_xor);
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

// -- Convenience node constructors: sequence layer ---------------------------

/// Lift a boolean into a one-cycle sequence.
pub fn seqBoolean(ir: *Ir, gpa: Allocator, b: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .seq_boolean, .data = .{ .lhs = @intFromEnum(b) } });
}

/// Cycle-delay concatenation `left ##[lo:hi] right`. Pass `left = .null` for a
/// leading delay. `hi == Ir.unbounded` encodes `$`.
pub fn seqDelay(
    ir: *Ir,
    gpa: Allocator,
    left: Node.Index,
    lo: u32,
    hi: u32,
    right: Node.Index,
) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.appendSlice(gpa, &.{ @intFromEnum(right), lo, hi });
    return ir.addNode(gpa, .{ .tag = .seq_delay, .data = .{ .lhs = @intFromEnum(left), .rhs = start } });
}

pub fn seqOr(ir: *Ir, gpa: Allocator, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .seq_or, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

pub fn seqIntersect(ir: *Ir, gpa: Allocator, lhs: Node.Index, rhs: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .seq_intersect, .data = .{ .lhs = @intFromEnum(lhs), .rhs = @intFromEnum(rhs) } });
}

/// Repetition `seq[*lo:hi]` (and goto/non-consecutive variants). `hi ==
/// Ir.unbounded` encodes `$`.
pub fn seqRepeat(
    ir: *Ir,
    gpa: Allocator,
    seq: Node.Index,
    mode: Repeat.Mode,
    lo: u32,
    hi: u32,
) Allocator.Error!Node.Index {
    const start: u32 = @intCast(ir.extra.items.len);
    try ir.extra.appendSlice(gpa, &.{ lo, hi, @intFromEnum(mode) });
    return ir.addNode(gpa, .{ .tag = .seq_repeat, .data = .{ .lhs = @intFromEnum(seq), .rhs = start } });
}

// -- Convenience node constructors: property layer ---------------------------

pub fn propSeq(ir: *Ir, gpa: Allocator, seq: Node.Index) Allocator.Error!Node.Index {
    return ir.addNode(gpa, .{ .tag = .prop_seq, .data = .{ .lhs = @intFromEnum(seq) } });
}

/// Implication `antecedent |-> consequent` (overlapping) or `|=>`
/// (non-overlapping), selected by `overlap`.
pub fn propImplies(
    ir: *Ir,
    gpa: Allocator,
    antecedent: Node.Index,
    consequent: Node.Index,
    overlap: bool,
) Allocator.Error!Node.Index {
    const tag: Node.Tag = if (overlap) .prop_implies_overlap else .prop_implies_nonoverlap;
    return ir.addNode(gpa, .{ .tag = tag, .data = .{ .lhs = @intFromEnum(antecedent), .rhs = @intFromEnum(consequent) } });
}

// -- Accessors ---------------------------------------------------------------

pub fn tagOf(ir: *const Ir, node: Node.Index) Node.Tag {
    return ir.nodes.items(.tag)[@intFromEnum(node)];
}

pub fn dataOf(ir: *const Ir, node: Node.Index) Node.Data {
    return ir.nodes.items(.data)[@intFromEnum(node)];
}

/// Decoded payload of a `seq_delay` node.
pub const Delay = struct { left: Node.Index, right: Node.Index, lo: u32, hi: u32 };

pub fn delayOf(ir: *const Ir, node: Node.Index) Delay {
    const d = ir.dataOf(node);
    const p = ir.extra.items[d.rhs..][0..3];
    return .{ .left = @enumFromInt(d.lhs), .right = @enumFromInt(p[0]), .lo = p[1], .hi = p[2] };
}

/// Decoded payload of a `seq_repeat` node.
pub const RepeatInfo = struct { seq: Node.Index, mode: Repeat.Mode, lo: u32, hi: u32 };

pub fn repeatOf(ir: *const Ir, node: Node.Index) RepeatInfo {
    const d = ir.dataOf(node);
    const p = ir.extra.items[d.rhs..][0..3];
    return .{ .seq = @enumFromInt(d.lhs), .lo = p[0], .hi = p[1], .mode = @enumFromInt(p[2]) };
}

// -- Hashing -----------------------------------------------------------------

/// A Blake3 content digest — the full 256-bit output. Its collision probability
/// is astronomically small, so callers may treat the digest as collision-free.
pub const Digest = [Blake3.digest_length]u8;

/// Content hash over every flat array. Structurally identical IRs hash equal,
/// so the pipeline can key a compiled-monitor cache on this. Section lengths
/// are folded in so that shifting a boundary between two arrays cannot collide.
pub fn hash(ir: *const Ir) Digest {
    var st = Blake3.init(.{});
    inline for (.{
        std.mem.sliceAsBytes(ir.nodes.items(.tag)),
        std.mem.sliceAsBytes(ir.nodes.items(.data)),
        std.mem.sliceAsBytes(ir.signals.items(.id)),
        std.mem.sliceAsBytes(ir.signals.items(.ty)),
        std.mem.sliceAsBytes(ir.assertions.items(.id)),
        std.mem.sliceAsBytes(ir.assertions.items(.kind)),
        std.mem.sliceAsBytes(ir.assertions.items(.flags)),
        std.mem.sliceAsBytes(ir.assertions.items(.clock)),
        std.mem.sliceAsBytes(ir.assertions.items(.disable)),
        std.mem.sliceAsBytes(ir.assertions.items(.prop)),
        std.mem.sliceAsBytes(ir.extra.items),
    }) |section| {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(section.len), .little);
        st.update(&len);
        st.update(section);
    }
    var out: Digest = undefined;
    st.final(&out);
    return out;
}

// -- Serialization -----------------------------------------------------------
//
// On-disk form (identical scheme to `libcrv`):
//
//     [magic: 4 bytes]
//     [format_version: u32]
//     [ per section: u32 element count, then the section's raw little-endian
//       bytes ]
//     [checksum: Blake3 digest over everything above]
//
// Every section is a paddingless array of POD, so there are no pointers to
// relocate; the reader resizes and blits straight back in. The trailing Blake3
// checksum is verified *before* any length is trusted, so a corrupt or tampered
// cache file fails cleanly rather than driving an allocation off a garbage
// count. Endianness is not converted (a local build cache is same-endian).

/// Identifies a libsva cache blob.
pub const magic: [4]u8 = .{ 'S', 'V', 'A', 'B' };
/// On-disk format version. Bump on any layout change; readers reject mismatches.
pub const format_version: u32 = 1;

const checksum_len = Blake3.digest_length;
/// Bytes preceding the first section: `magic` + `format_version`.
const header_len = magic.len + @sizeOf(u32);

/// Serialize into a freshly allocated byte buffer owned by the caller.
pub fn serialize(ir: *const Ir, gpa: Allocator) Allocator.Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(gpa);

    try out.appendSlice(gpa, &magic);
    try writeU32(&out, gpa, format_version);

    try writeU32(&out, gpa, @intCast(ir.nodes.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try writeU32(&out, gpa, @intCast(ir.signals.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.signals.items(.id)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.signals.items(.ty)));

    try writeU32(&out, gpa, @intCast(ir.assertions.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.id)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.kind)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.flags)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.clock)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.disable)));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.assertions.items(.prop)));

    try writeU32(&out, gpa, @intCast(ir.extra.items.len));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(ir.extra.items));

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

    var ir: Ir = .{};
    errdefer ir.deinit(gpa);

    // Counts are trusted only now, after the checksum has been verified.
    var cur: Cursor = .{ .bytes = payload, .pos = header_len };

    try ir.nodes.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.tag)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.nodes.items(.data)));

    try ir.signals.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.signals.items(.id)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.signals.items(.ty)));

    try ir.assertions.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.id)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.kind)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.flags)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.clock)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.disable)));
    try cur.copyInto(std.mem.sliceAsBytes(ir.assertions.items(.prop)));

    try ir.extra.resize(gpa, try cur.readU32());
    try cur.copyInto(std.mem.sliceAsBytes(ir.extra.items));

    return ir;
}

fn writeU32(out: *std.ArrayListUnmanaged(u8), gpa: Allocator, v: u32) Allocator.Error!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try out.appendSlice(gpa, &buf);
}

const Cursor = struct {
    bytes: []const u8,
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

    var ir: Ir = .{};
    defer ir.deinit(gpa);

    // Signals a, b, c and clock clk (ids stand in for a symbol table).
    const clk = try ir.addSignal(gpa, .{ .id = @enumFromInt(0), .ty = Type.bit(1) });
    const a = try ir.addSignal(gpa, .{ .id = @enumFromInt(1), .ty = Type.bit(1) });
    const b = try ir.addSignal(gpa, .{ .id = @enumFromInt(2), .ty = Type.bit(1) });
    const c = try ir.addSignal(gpa, .{ .id = @enumFromInt(3), .ty = Type.bit(1) });

    // assert property (@(posedge clk) a ##1 b |-> c);
    const seq_a = try ir.seqBoolean(gpa, try ir.signalRef(gpa, a));
    const seq_b = try ir.seqBoolean(gpa, try ir.signalRef(gpa, b));
    const ante = try ir.seqDelay(gpa, seq_a, 1, 1, seq_b);
    const cons = try ir.propSeq(gpa, try ir.seqBoolean(gpa, try ir.signalRef(gpa, c)));
    const prop = try ir.propImplies(gpa, ante, cons, true);

    _ = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(7),
        .kind = .assert,
        .flags = .{},
        .clock = try ir.signalRef(gpa, clk),
        .disable = .null,
        .prop = prop,
    });

    // Serialize -> deserialize preserves structure and content hash.
    const bytes = try ir.serialize(gpa);
    defer gpa.free(bytes);

    var ir2 = try Ir.deserialize(gpa, bytes);
    defer ir2.deinit(gpa);

    try std.testing.expectEqual(ir.nodes.len, ir2.nodes.len);
    try std.testing.expectEqual(ir.signals.len, ir2.signals.len);
    try std.testing.expectEqual(ir.assertions.len, ir2.assertions.len);

    const h1 = ir.hash();
    const h2 = ir2.hash();
    try std.testing.expectEqualSlices(u8, &h1, &h2);

    // Opaque ids round-trip verbatim.
    try std.testing.expectEqual(@as(Assertion.Id, @enumFromInt(7)), ir2.assertions.items(.id)[0]);

    // The decoded delay payload survives.
    const d = ir2.delayOf(ante);
    try std.testing.expectEqual(@as(u32, 1), d.lo);
    try std.testing.expectEqual(@as(u32, 1), d.hi);
    try std.testing.expectEqual(seq_b, d.right);
}

test "cache format is versioned and checksum-protected" {
    const gpa = std.testing.allocator;

    var ir: Ir = .{};
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
