# libsva

A Zig library for **concurrent assertions**: temporal properties checked over a
trace of discrete cycles. You build assertions into `sva.Ir`, hash or cache it,
and hand it to a backend that lowers it to automata and runs or compiles it.

The IR's leaves are opaque handles, so libsva is tied to no particular assertion
syntax: any front-end that can name its own terms and evaluate them once per
cycle can build against it.

## Requirements

- [Zig](https://ziglang.org/) `0.16.0` or newer.

## Install

```sh
zig fetch --save git+https://github.com/coderkalyan/libsva
```

```zig
// ReleaseSafe, not ReleaseFast: the IR contract is assert-enforced.
const sva = b.dependency("libsva", .{ .target = target, .optimize = .ReleaseSafe });
exe.root_module.addImport("sva", sva.module("sva"));
```

## Usage

```zig
const std = @import("std");
const sva = @import("sva");

pub fn build(gpa: std.mem.Allocator) !sva.Ir {
    var ir: sva.Ir = .init();

    // Atoms are opaque u32 handles you assign, dense from zero — e.g. indices
    // into your own symbol table. libsva never inspects one; evaluating and
    // sampling them is entirely your side of the fence. A clock is a handle in
    // its own space, since it delimits cycles rather than being sampled in one.
    const clk: sva.Ir.Clock = @enumFromInt(0);
    const req = try ir.atom(gpa, @enumFromInt(0));
    const ack = try ir.atom(gpa, @enumFromInt(1));

    // In every cycle in which `req` holds, `ack` must hold the cycle after.
    const ante = try ir.seqBoolean(gpa, req);
    const cons = try ir.propSeq(gpa, .prop_seq_weak, try ir.seqBoolean(gpa, ack));
    const prop = try ir.propImplies(gpa, .prop_implies_nonoverlap, ante, cons);
    _ = try ir.addAssertion(gpa, .{
        .id = @enumFromInt(10),
        .kind = .assert,
        .clock = clk,
        // Never resetting is a constant false, not an absent edge.
        .reset = try ir.boolConst(gpa, false),
        .prop = prop,
    });

    return ir;
}
```

Constructors come in three layers, and each asserts that its operands belong to
the layer it expects:

- **boolean** — `atom`, `boolConst`, `bNot`, `bBinary` (`.band`, `.bor`).
- **sequence** — `seqBoolean`, `seqDelay`, `seqOr`, `seqIntersect`, `seqRepeat`
  (`.seq_repeat_consec`, `.seq_repeat_goto`, `.seq_repeat_nonconsec`).
- **property** — `propSeq` (`.prop_seq_weak`, `.prop_seq_strong`),
  `propImplies` (`.prop_implies_overlap`, `.prop_implies_nonoverlap`).

Read a tree back with `tagOf` for a node's operator, `atomOf`, `delayOf` and
`repeatOf` for its payload, and `operandsOf(node, &buf)` to enumerate the nodes
it refers to.

`ir.hash()` is a Blake3 content digest: structurally identical assertion sets
hash equal, so it works as a cache key for whatever a backend compiles.
`ir.serialize(gpa)` and `Ir.deserialize(gpa, bytes)` move the IR to and from a
byte buffer, versioned and checksummed, so a stale or corrupt blob is rejected
rather than misread.

## Semantics

A **trace** is a sequence of cycles, finite (a terminated simulation) or
infinite (a formal model), each assigning a boolean to every atom. The engine is
stepped once per cycle. `Assertion.clock` names what delimits a cycle; edge
selection is yours, so the rising and falling edge of one signal are two
distinct `Clock` handles.

An assertion is **implicitly quantified over start cycles**: one *attempt* is
seeded every cycle, and the verdict is over all of them — an `assert` holds only
if every attempt holds, a `cover` reports each attempt that matches. There is no
`always` operator because that quantification is always there.

Within an attempt, a sequence property **holds** as soon as its sequence
completes a match and **fails** as soon as the sequence can no longer match. An
implication holds vacuously if its antecedent never completes a match;
otherwise each antecedent match obliges the consequent.

A finite trace ends with attempts still pending, and their **strength** decides
what happens: a weak property discharges the obligation and holds, a strong one
fails. Implication takes its strength from its consequent.

**Empty matches do not exist.** Every sequence matches at least one cycle:
repetition counts are `>= 1` and nothing else can introduce an empty match, so
none of the operators carry empty-match special cases. A source language with a
zero-count repetition expands it away before building the IR, per that
language's own empty-match rules.

An assertion's `reset` is a boolean node: whenever it holds, every in-flight
attempt is abandoned — neither holding nor failing nor matching — and fresh
attempts are seeded again from the next cycle.

**Nothing in the tree is optional.** Every edge is present, so a `Node.Index`
always names a node and there is no sentinel for an absent one. The two things
that used to need one are spelled with values the IR already has. A delay
measured from the start of an attempt is `seqDelay` with a one-cycle `true` to
its left — exactly, with no adjustment to the bounds — so there is one delay
operator rather than two counting differently. And an assertion that never
resets carries a `boolConst` of false, which costs a backend one constant fold
and the monitor one branch fewer.

Atom handles must be **dense** from zero; `ir.atom_count` is the width of the
per-cycle valuation a backend allocates and indexes by handle.

## Automata

`sva.glushkov.sequence(gpa, &ir, node, .{}, null)` lowers a sequence to an
`sva.Nfa` and the alphabet its guards are written over. It is a position
automaton: one state per letter occurrence, no ε-transitions, and *homogeneous*
— every transition entering a state carries the same guard, so a guard belongs
to the state and there is no start state, because `first` is a mask. A cycle is
then three bit-set passes:

```zig
alphabet.evaluate(&ir, atoms, root_bits); // a backend emits this part itself
nfa.satisfy(&alphabet, root_bits, holds); // states whose guard holds this cycle
nfa.advance(current, holds, next);        // ⋃ follow[p] for p in current, & holds
nfa.seed(holds, next);                    // start this cycle's attempt: | first
if (nfa.matched(next)) { ... }            // & last
```

`alphabet.roots` is exactly the set of boolean nodes a backend evaluates once
per cycle; past that it is bit arithmetic over the result. Guards are interned
cubes over those roots, and `guard_states` gives the mask of states carrying
each one, so building the satisfied mask folds over guards rather than gathering
per state. Seeding an attempt every cycle is one `or`; a reset is `current = 0`.

A delay range costs a filler state per cycle it may skip, repetition unrolls,
and `intersect` is a product, so every one of those is a blow-up vector.
`Options.limit` caps the state count, checked before anything is allocated, and
`Diagnostic` names the node that tripped it. Goto and non-consecutive
repetition are rejected over an operand that is not a `seqBoolean`.

## Monitors

`sva.glushkov.assertion(gpa, &ir, index, .{}, null)` lowers a directive to an
`sva.Monitor`: an *obligation* automaton, and for an implication a *trigger*
automaton saying when to start one. Both are built against a single alphabet, so
a term the antecedent and the consequent share is evaluated once rather than
twice.

The two are tracked differently, and that asymmetry is the design. The trigger
is existential — all we ask is whether *some* antecedent match ends this cycle —
so the union of every trigger attempt is sound and one state vector does. The
obligation is per-attempt: merging attempts is unsound, because one attempt's
threads can die while another's live and the union hides the failure. Each
pending attempt therefore gets its own state vector, and `monitor.depth` says
how many a backend must allocate.

`depth` is the obligation's longest match, which is exactly right: an attempt
started in cycle `t` has either matched or run out of live states by cycle
`t + depth - 1`. An obligation with no bounded longest match — `##[1:$] ack`, or
any goto repetition — is rejected with `ObligationNotBounded`, because tracking
it soundly needs the obligation determinised so that one state characterises an
attempt. A consequent that is itself an implication is rejected too.

`Monitor.State` is a reference implementation of the whole per-cycle protocol,
which is the contract a backend reproduces rather than something it calls.

## The C API

`include/sva.h`, linked against `libsva.a`. One export per Zig entry point.

An `sva_ir`, an `sva_monitor` and an `sva_run` are libsva's objects in *your*
storage — declare one, initialise it in place, tear it down there. Only what they
grow internally is allocated, and libsva picks its own allocator, so there is
nothing to configure and nothing to pass. Their contents are opaque; the header
declares their sizes with headroom, and the library fails its own build — naming
the size it needs — if a struct outgrows one.

An `sva_automaton` is only ever reached by pointer, so it has no size at all.
Reading a monitor is direct: an accessor hands back a pointer to a whole array
and you index it — one call per table, not per element.

```c
sva_ir ir;
sva_ir_init(&ir);

sva_node req, ack, consequent, property;
uint32_t index;
if (sva_bool_atom(&ir, 0, &req) != SVA_OK) goto fail;
if (sva_seq_boolean(&ir, req, &req) != SVA_OK) goto fail;
if (sva_bool_atom(&ir, 1, &ack) != SVA_OK) goto fail;
if (sva_seq_boolean(&ir, ack, &ack) != SVA_OK) goto fail;
if (sva_prop_sequence(&ir, SVA_WEAK, ack, &consequent) != SVA_OK) goto fail;
if (sva_prop_implies(&ir, SVA_NONOVERLAPPING, req, consequent, &property) != SVA_OK) goto fail;

sva_directive d = { 0 };
d.kind = SVA_ASSERT;
d.clock = SVA_CLOCK_NONE;
if (sva_bool_const(&ir, false, &d.reset) != SVA_OK) goto fail; /* never resets */
d.property = property;
if (sva_ir_add(&ir, &d, &index) != SVA_OK) goto fail;

sva_monitor monitor;
if (sva_lower(&monitor, &ir, index, NULL, NULL) == SVA_OK) { /* emit code */ }
```

Anything that can fail returns a status and delivers its result through an out
parameter, written only on `SVA_OK`. For a builder the only failure is an
allocation failure, and there is no accumulated state to check later: a caller
that ignores a return leaves its handle unwritten.

Nothing else is checked. A **malformed tree** — an operand from the wrong layer, a
reversed range — is a bug in the caller's lowering, not a property of what it is
compiling; there is no source location to report, because the location is the
caller's own code. The Zig assertions stand and the wrapper does not check twice.
**Legal input libsva cannot lower yet** — an unbounded obligation, a nested
implication, a limit tripped — comes back from `sva_lower` as a status plus the
node that caused it, which is what a front-end turns into a diagnostic.

`sva_bool_read` walks a boolean node back. A monitor's roots *are* boolean nodes,
so this is how a backend discovers what to evaluate each cycle.

### Compiling the protocol rather than interpreting it

`Monitor.State` iterates the set bits of a state vector at run time. A code
generator cannot — it has no loop over an unknown set. It reads every count and
every mask out of the monitor once, then emits a fold over *statically known*
states that selects each state's constant follow row by its own liveness bit:

```c
next = 0;
for (each state s)                           /* unrolled at compile time */
    next |= sext(current >> s) & FOLLOW_s;    /* FOLLOW_s is a constant */
next &= satisfied;
```

`satisfy` transposes the same way, folding over guards instead of gathering per
state — which is what `sva_automaton_guard_states` is for. Both collapse to
branch-free integer arithmetic, and a state vector is one machine integer per
attempt rather than a word array.

`test/backend.c` is that backend, differential-tested against `Monitor.State`
over random traces across eight property shapes. It is a test of the *contract*:
if the exposed tables were not enough to reconstruct the protocol, or the
transposition were unsound, it disagrees.

One trap it pins down: `occupied` is **not** redundant with `pending != 0`. An
attempt seeded into a cycle whose first letter does not hold has an empty state
vector immediately, and that is a *failure* — the most common one there is. A
backend that derives occupancy from emptiness cannot tell it from an absent
attempt and silently reports nothing.

## Building the repository

```sh
zig build          # build the static library into zig-out/
zig build test     # unit tests, the C smoke test, and the backend test
```

**Build it safe.** Everything above — operand layers, repetition and delay
ranges, dense atoms — is enforced entirely by `std.debug.assert`, so
`ReleaseFast` silently accepts a malformed assertion and lowers it to a wrong
monitor. Develop against `Debug` and ship `ReleaseSafe`.

## License

MIT — see [LICENSE](LICENSE).
