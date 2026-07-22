# libsva

A Zig library for **SystemVerilog Assertions (SVA)**. This first cut is the
**core IR**: a small, cacheable intermediate representation for concurrent
assertions. It borrows its data-oriented, content-hashable design from
[`libcrv`](https://github.com/coderkalyan/libcrv) and applies it to a different
domain — where libcrv models constrained-random *constraints*, libsva models
temporal *assertions*.

The lowering to automata and the runtime/formal backends that consume them are
staked out below as a roadmap and land in follow-up changes.

## Requirements

- [Zig](https://ziglang.org/) `0.16.0` or newer.

## Building and testing

```sh
zig build          # build the static library into zig-out/
zig build test     # run the unit tests
```

## Using it as a dependency

```sh
zig fetch --save git+https://github.com/coderkalyan/libsva
```

```zig
const sva = b.dependency("libsva", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("sva", sva.module("sva"));
```

```zig
const std = @import("std");
const sva = @import("sva");

pub fn build(gpa: std.mem.Allocator) !sva.Ir {
    var ir: sva.Ir = .{};

    // Signals (ids are yours — e.g. indices into your symbol table).
    const clk = try ir.addSignal(gpa, .{ .id = @enumFromInt(0), .ty = .bit(1) });
    const req = try ir.addSignal(gpa, .{ .id = @enumFromInt(1), .ty = .bit(1) });
    const ack = try ir.addSignal(gpa, .{ .id = @enumFromInt(2), .ty = .bit(1) });

    // assert property (@(posedge clk) req |=> ack);
    const ante = try ir.seqBoolean(gpa, try ir.signalRef(gpa, req));
    const cons = try ir.propSeq(gpa, try ir.seqBoolean(gpa, try ir.signalRef(gpa, ack)));
    const prop = try ir.propImplies(gpa, ante, cons, false); // false = |=>
    _ = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(10),
        .kind = .assert,
        .flags = .{},
        .clock = try ir.signalRef(gpa, clk),
        .disable = .null,
        .prop = prop,
    });

    return ir; // hash with ir.hash(); cache with ir.serialize(gpa)
}
```

## The IR

`sva.Ir` is a **flattened tree**, exactly the layout libcrv uses: expression,
sequence, and property nodes live in one contiguous array and refer to each
other by index rather than by pointer. Those indices are distinct `enum(u32)`
types (`Node.Index`, `Signal.Index`, ...) so the compiler catches an index used
against the wrong table. Signals and assertions carry an opaque per-instance
`u32` id — the IR stores no text; resolve an id against whatever symbol table
your front-end keeps on the side.

An assertion language is layered, so the node array holds three layers back to
back, distinguished by `Node.Tag`:

1. **Boolean layer** — the alphabet: sampled boolean expressions over signals
   (`signal_ref`, `b_and`, `b_or`, `b_not`, `b_xor`, `bool_const`). A maximal
   boolean node is what will guard an automaton transition.
2. **Sequence layer (SEREs)** — regular patterns over cycles: cycle-delay
   concatenation `a ##[lo:hi] b` (including `##0` overlap and the unbounded
   `##[lo:$]`), `or`, `intersect`, and consecutive / goto / non-consecutive
   repetition.
3. **Property layer** — implication (`|->`, `|=>`), boolean property
   combinators, and the temporal operators (`always`, `s_eventually`,
   `nexttime`, `until`).

Because it is all plain POD arrays:

- `ir.hash()` is a **Blake3** content digest over the arrays (256-bit) —
  structurally identical assertion sets hash equal, so it doubles as a
  collision-resistant cache key for the compiled monitor a backend will produce.
- `ir.serialize(gpa)` / `Ir.deserialize(gpa, bytes)` blit the arrays to and from
  a byte buffer behind a `magic` + 32-bit `format_version` header, with a
  trailing Blake3 checksum that is verified *before* any length is trusted — so
  a corrupt or tampered cache file fails cleanly rather than driving an
  allocation off a garbage count.

The `Node.Tag` set stakes out the full concurrent-assertion language up front so
the on-disk format is stable; the lowering (next) implements a core subset and
rejects the rest cleanly, the same way libcrv's evaluator covers a scalar core
and leaves the structural constraints as roadmap.

## Roadmap

The IR is the foundation for an engine that lowers concurrent assertions to
finite automata and runs or compiles them. Landing next, in separate changes:

- **`Nfa`** — a Thompson ε-NFA lowering from sequences (cycle-delay
  concatenation with `##0` fusion, `or`, repetition, `intersect`), plus the
  ε-closure/step simulation primitives.
- **`Backend`** — a swappable runtime-backend interface (type-erased pointer +
  vtable, in the style of `std.mem.Allocator` and libcrv's `Solver`).
- **`Coroutine`** — a per-cycle stepping monitor that seeds a fresh antecedent
  attempt each cycle and spawns a consequent coroutine per match.
- **`Formal`** — a backend that compiles a monitor to a NuSMV/nuXmv symbolic
  transition model for exhaustive checking.

## Layout

```
build.zig                # build graph: module, static library, test step
build.zig.zon            # package manifest (name, version, fingerprint)
src/root.zig             # library root — the public API
src/Ir.zig               # the flattened-tree, cacheable SVA IR
```

## License

MIT — see [LICENSE](LICENSE).
