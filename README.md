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
- **sequence** — `seqBoolean`, `seqDelay`, `seqOr`, `seqAnd`, `seqIntersect`,
  `seqThroughout`, `seqWithin`, `seqFirstMatch`, `seqRepeat`
  (`.seq_repeat_consec`, `.seq_repeat_goto`, `.seq_repeat_nonconsec`).
- **property** — `propSeq` (`.prop_seq_weak`, `.prop_seq_strong`),
  `propImplies` (`.prop_implies_overlap`, `.prop_implies_nonoverlap`),
  `propJunction` (`.prop_and`, `.prop_or`), `propRange` (`.prop_delay`,
  `.prop_always`), `propFollowedBy` (`.prop_followed_by_overlap`,
  `.prop_followed_by_nonoverlap`), `propAbort` (`.prop_accept_on`,
  `.prop_reject_on`, `.prop_sync_accept_on`, `.prop_sync_reject_on`),
  `propUntil` (`.prop_until`, `.prop_until_with`, `.prop_s_until`,
  `.prop_s_until_with`).

Read a tree back with `tagOf` for a node's operator, `atomOf`, `delayOf`,
`repeatOf` and `propRangeOf` for its payload, and `operandsOf(node, &buf)` to
enumerate the nodes it refers to.

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
fails. Implication takes its strength from its consequent, every operator above
the leaves takes it from its operands, and `propNot` **inverts** it.

`propJunction` constrains the two sides rather than choosing between them, and
which constraint depends on the tag. `.prop_and` and `.prop_or` want them equal:
the pair is pending until both resolve, and a pending pair with one weak and one
strong side has no single disposition to carry. `.prop_implies` wants them to
*differ*, which is the same rule seen through its reduction to `not p or q`.
`.prop_iff` can never satisfy it and is always refused; see "Negation" below.

### The property layer above a sequence

The operators above `propSeq` combine **verdicts**, not matches, and none of them
gets automaton machinery of its own. A sequence property holds exactly when its
sequence has a match, so "both hold" is "both have a match", and the whole layer
collapses onto the sequence layer:

| operator | is | states added |
|---|---|---|
| `propJunction .prop_and` | `seqAnd` of the two obligations | the product's |
| `propJunction .prop_or` | `seqOr` — a disjoint union | 0 |
| `propRange .prop_delay` | a filler chain the operand is entered from | `hi` |
| `propRange .prop_always` | the operand shifted to each offset, conjoined | the product's |
| `propFollowedBy` | the antecedent *concatenated* with the operand | the fusion's |
| `propUntil` | the left operand's run, then the right one — `##1` for the plain forms, a `##0` fusion for `_with` | the delay's |
| `propNot` | the operand determinised, with two masks exchanged | the determinisation's |

`propRange` reads its range **disjunctively** for `.prop_delay` and
**conjunctively** for `.prop_always`, which is the whole difference between
`nexttime[n]` / `eventually [m:n]` and `always [m:n]`. The two coincide at
`lo == hi`. An open `hi` spells the unbounded `always p` and `s_eventually p`,
which leaves the obligation without a longest match and so selects the counted
tracking discipline described further down.

`propUntil` is LRM 16.12.10's four operators, and reading the definition as a set
of matches is what makes them cheap. `p until q` asks that `q` hold at some cycle
`k` with `p` holding over `[t, k)`, which is `q or (p[*1:$] ##1 q)`; `until_with`
asks `p` at `k` too, which is a fusion of the run's last cycle with `q`'s first,
so `p[*1:$] ##0 q` and no disjunct is needed. The `s_` forms are the same
construction — strength is not a property of the match set but of what becomes of
an attempt the trace ends with still pending, and `q` never arriving is exactly
that. Unlike every other property tag, the tag carries the strength rather than
taking it from its leaves.

`propFollowedBy` is the existential dual of `propImplies`, and it is *not* that
construction with the verdict flipped. LRM 16.12.9 defines `s #-# p` as
`not (s |-> not p)`, and negating an implication turns its universal over
antecedent matches into an existential **and** turns vacuity from a pass into a
failure. An existential over "`s` matches, then `p` holds from there" is just a
match of the two concatenated, so followed-by never becomes a trigger at all —
its antecedent joins the obligation. The visible consequence is the one that
matters: where an implication with no antecedent match holds vacuously, a
followed-by asserted that a witness exists and **fails**.

That is also why there is no conditional or `case` operator. `if (c) p else q` is
`(c #-# p) or (!c #-# q)`: exactly one guard matches, the other disjunct has no
witness and so contributes a failure rather than a vacuous pass, and the taken
branch decides the verdict. The `case` form is the same shape with one disjunct
per item. Spelled with implication instead it would always pass.

A shift over an implication moves the **antecedent**, not the obligation:
`nexttime[n] (s |-> p)` obliges `p` for exactly those matches of `s` that start
`n` cycles into the attempt, and since a trigger is seeded every cycle, growing
an `n`-cycle filler prefix in front of it selects exactly those and costs the
ring nothing. A *range* survives that move only for `.prop_always`, because a
conjunction of universals is a universal over the union of their triggers;
`eventually [m:n] (s |-> p)` is a disjunction of separately-vacuous implications
and no single trigger says that, so it is refused.

**Bounded `always` is linear, not a blow-up.** A hundred-way conjunction sounds
like the case for a dedicated construction, and it is not: consecutive conjuncts
only have a reachable product where their copies of the operand are live at the
same time, and that window is as wide as the operand however wide the range is.
So `always [1:100] p` costs `101` states for a one-cycle `p` and `3535` for a
five-state one — linear in the range with a slope set by `p` alone. This lowering
*is* the desugaring, producing exactly the state counts a front-end's own
`seqAnd` spelling would; the operator earns its place by keeping the semantics
and the refusals in one place, not by beating it on size.

### Negation

`propNot` is the one property operator that needs a construction of its own,
because a monitor answers "did this match" and negation asks "did this *not*
match" — a complement, which needs the operand determinised.

The automaton half is two masks exchanged on a **total** determinisation:

- A total determinised state is a subset of the operand's states, and the subset
  is empty exactly where every thread has died — where the operand *fails*, so
  where `not p` **holds**. Those become the complement's exits.
- A state whose subset meets the operand's exits is where the operand completes a
  match, so where `not p` **fails**. A failure here is an empty state vector, so
  those states are given a `never` guard: nothing enters them, the vector empties
  in exactly that cycle, and the ordinary reading reports it.

What comes out is an ordinary fragment. It composes with the disjunction and the
shifts like any other, and — the part that surprises — it does **not** select the
counted discipline. Counted mode exists for obligations with no longest match and
reads failure out of `dead` because its determinisation is left total; a
complement's is not, so a ring holds it. `not p` borrows `p`'s bound exactly:
after `p`'s longest match every thread has matched or died, so `p` has resolved
and so has its complement.

**The verdict lattice is three-valued, and that is what negation costs.**
Complementing an automaton exchanges match for no-match, which is two values;
the third has no automaton to invert it:

```
not holds = fails      not fails = holds      not pending = pending
```

`sva_verdict` names it in the ABI and `sva_verdict_not` is the normative table.
The last row is load-bearing rather than decorative. It says a complement may not
resolve an attempt one cycle before its operand resolves, however certain the
outcome has become — and `p ##1 (b or !b)` is a real case where the outcome is
certain a cycle early. Ordinary pruning deletes states that can reach no exit,
because for a sequence "no exit reachable" and "no continuation can match" are
the same sentence; inside a complement they are not, and pruning there would
report the failure at the cycle the match became *inevitable* instead of the
cycle it arrived. So a complement marks its states as pending-by-construction and
pruning spares them. Without that, `not not p` would drift earlier than `p`.

Strength inverts too (LRM 16.12.2). An **abort beneath a `not` is refused**, and
not for want of a mask — the mask moves through the subsets fine. Arming is what
does not survive: an abort is armed from the *unfiltered* one-step image of
where an attempt is, deliberately, so that it covers the cycle its subtree's
last thread dies in — which is the cycle an `accept_on` exists for, discharging
an attempt the property was about to fail. A determinised successor is already
filtered by the valuation, and the complement then deletes the very state that
image would have been read off, so the abort would read as unarmed exactly when
it should fire.

`p implies q` is `not p or q` and `p iff q` is the conjunction of the two
directions. Both are tags rather than front-end desugarings for one reason —
a refusal has to name the operator the user wrote, and the desugared tree would
put `prop_or` on the diagnostic. They are not cheaper: the `implies` lowering
*is* the desugaring and produces the same automaton state for state.

The strength arithmetic is where this bites. `not` inverts, and the disjunction
underneath wants its operands to agree, so **`implies` lowers only when its two
operands' strengths differ** — `weak implies weak` is refused. That is not
conservatism: with both weak, a trace where `q` has definitively failed while `p`
is still pending ends with `p` holding, so `q` was required and the implication
fails, while a trace where both are pending ends with it holding. One
`Monitor.Strength` cannot carry both. `iff` is worse and it is arithmetic: each
direction forces its operands opposite and comes out carrying its right-hand
one's strength, so the outer conjunction always sees a mixed pair and every `iff`
is refused. Lifting both needs a per-state end-of-trace disposition rather than
one bit per monitor, which would change what `sva_monitor_strength` means.

Determinisation is exponential in the worst case and the cap is the existing one:
`Options.limit`, reported through the same path as every other blow-up, with
`Diagnostic.node` naming the `not`. Measured, complementing costs roughly double:
`not a` is 2 states against 1, `not (a ##1 b)` 4 against 2, `not (a ##[1:3] b)`
6 against 4.

### Aborts

`propAbort` resolves the property beneath it the moment its condition holds,
while that property is still being evaluated: `.prop_accept_on` resolves it as
**holding**, `.prop_reject_on` as **failing**.

This is not `reset` with a different spelling, and the difference is the point: a
reset **abandons** an attempt, leaving no verdict at all, while an accept
**discharges** it, which is a pass a `cover` counts. Counting failures alone
cannot tell the two apart, which is exactly why it is worth stating.

An abort is also *scoped*, where a reset is monitor-wide. It reaches one subtree
of the obligation, named by a state mask on the `Monitor.Abort`, and an abort is
**armed** for a pending attempt only while that attempt could hold a thread
inside the mask. That is what confines it to the cycles its property is really
being evaluated in — `nexttime[2] (reject_on(b) p)` cannot reject during the two
cycles it spends reaching `p` — without anything having to track a window. Arming
is measured over the states the attempt could *reach*, not the ones it ends up
live in, so it covers both the cycle the subtree is entered in and the cycle its
last thread dies in; LRM 16.12.7 arms an abort at any point during the evaluation
attempt, and both are such a point. An abort therefore wins a tie against the
property's own verdict in the cycle they coincide.

Because the scope is the obligation, an abort below an implication leaves the
antecedent alone: matches of it keep raising fresh obligations in the very cycles
the condition holds, which a monitor-wide reset would suppress.

An abort *above* an implication is a different property from one below it, not
another spelling of the same one, so neither can be rewritten into the other. It
covers the antecedent as well, and `Monitor.Abort.covers_trigger` says so: when
the condition holds, every attempt with a live antecedent thread is resolved with
the abort's effect and seeds nothing — the ones whose match ends there and the
ones still matching alike, one verdict each. That is a cycle earlier than the
pushed-down form can manage, since below the implication the abort only arms
once the consequent starts — the cycle after, for `|=>`. A monitor has exactly
one trigger, so an abort above it covers every attempt of it uniformly; how many
attempts that is comes from `Monitor.trigger_window`, and their threads are
cleared so that none completes an antecedent later and imposes an obligation on
an attempt already resolved.

When two aborts resolve one attempt in the same cycle the **outermost** wins.
`accept_on(a) (reject_on(b) p)` with both conditions holding is a pass: the
reject settles the inner evaluation as false, and the accept's condition is then
true during an evaluation still under way, which LRM 16.12.14 reads as
accepting. `monitor.aborts` runs innermost to outermost, so the winner is the
last entry rather than the first.

A reject below a `.prop_or` stops resolving the whole attempt and only clears its
own disjunct — the other side may still hold. An accept keeps resolving the
attempt, since one disjunct holding makes the disjunction hold.

The `.prop_sync_*` tags read the condition from **sampled** values and the plain
ones from **live** values, mirroring `reset` (LRM 16.15). Nothing in the IR can
check that: the request is recorded on `Monitor.Abort.sampled` for a backend to
honour, and honouring it is the front-end's half of the contract.

**Live means every time step**, not merely the ones carrying a clocking event,
so a backend stepped only on the clock edge has to latch the condition: set a
bit whenever it holds at any time slot, then consume and clear that bit at the
next edge. Reading it at the edge alone is not the live valuation, and the
failure mode is silent — it collapses the plain forms onto the `.prop_sync_*`
ones, since a condition pulsing strictly between two ticks is precisely what
separates them.

Four shapes are refused rather than mis-lowered. A property operator over an
operand that raises a trigger of its own — a conjunction of two implications, a
followed-by or a disjunctive shift over one — has nowhere to put the second
trigger, since a monitor has one. An abort below an operator that *renumbers*
its operand's states cannot carry its mask through: a disjunction and a shift
move an operand as a block and are fine, a product (`.prop_and`,
`.prop_always`) and a fusion (`propFollowedBy`) are not. And an implication whose
antecedent has no bounded longest match — `(a ##[1:$] b) |-> c` — is refused with
`AntecedentNotBounded`: an obligation is owed per antecedent *match*, so the cycle
a match ends in has to report how many attempts ended one there, and an unbounded
antecedent has unboundedly many alive at once to tell apart. Counting them
soundly needs the trigger determinised; reporting the number of matching *cycles*
instead is a wrong number rather than a coarse one.

And an abort that only kills its own branch — a reject under a disjunction —
cannot be carried over an obligation that will be counted, because killing part
of an attempt is a mask subtracted from its state vector and a counted attempt
has no vector, only one state standing for a whole subset. An abort that resolves
the whole attempt is carried through the determinisation and is fine: being armed
depends on the state alone, so firing retires every attempt counted there at
once, which is exact rather than an approximation of retiring "the" attempt.

**Empty matches do not exist.** Every sequence matches at least one cycle:
repetition counts are `>= 1` and nothing else can introduce an empty match, so
none of the operators carry empty-match special cases. A source language with a
zero-count repetition expands it away before building the IR, per that
language's own empty-match rules.

That ban is what shapes the three operators SVA writes over whole matches.
`seqAnd` is `seqIntersect` without the requirement that the two matches end in
the same cycle: both start in the cycle the attempt does, and the match ends at
the later of the two ends, so a side that has already finished stops
constraining the trace rather than waiting on it. `seqThroughout` takes a
*boolean* on the left — the only sequence-layer operand that is not a sequence
— and requires it in every cycle the match spans; the LRM writes that as
`(exp[*0:$]) intersect seq`, which cannot be built here, but since no sequence
matches an empty span, `(exp[*1:$]) intersect seq` is the same language, and
that product is a conjunction into every state's guard and nothing else.

`seqWithin` is not an operator of its own. It expands into
`(true ##[0:$] inner ##[0:$] true) intersect outer`, which is the LRM's own
reading with its `1[*0:$]` padding replaced by a `0..$` delay against a
one-cycle `true` — the same padding without the empty match, because a delay
of 0 fuses the padding cycle into the match beside it and so can cost nothing.
A tag is worth adding for a construction the existing ones cannot reach, and
every consumer of the IR carries it forever; `within` is not one.

`seqFirstMatch` keeps only the **shortest** match of a sequence from each start
cycle: as soon as one match completes, every longer match of that attempt is
suppressed. It is the one sequence operator that is not a language operation on
matches taken one at a time — whether a match survives depends on whether a
*shorter* one covers the same cycles, which is the set difference against the
operand's own longer matches, and a difference is a complement. Nothing built
out of unions and products is one, so unlike `within` this does need a tag, and
unlike every other operator it lowers by determinising rather than composing.

It is worth knowing where that is observable, because it is not everywhere. Over
a *property* it is a no-op: a sequence property holds as soon as its sequence
completes a match — the earliest, which is the one `first_match` keeps — and
fails as soon as it can no longer match, which while pending is the same
question for both, since a match having ended would already have made it hold.
Pending-at-end coincides too, so strength does not change it. `first_match`
bites only where the operand's *later* matches are themselves observable: as an
implication antecedent, because every antecedent match raises an obligation of
its own, and inside a larger sequence, because a suppressed match is a
continuation that never happens. A test written over a bare
`assert property (first_match(s))` therefore passes whether the operator is
implemented or ignored.

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

**Atoms are assumed two-valued.** `bNot` is boolean complement — the guard layer
pushes it down to a leaf and evaluates it as `!atom`. A front-end whose own
booleans are *three*-valued must not emit it. SystemVerilog is the motivating
case: LRM 16.6 reads an unknown expression as false, so under `a = 1'bx` both `a`
and `!a` are false and `a || !a` does not hold, whereas complementing the
collapsed bit reports it as true. Such a front-end lowers to negation normal form
and gives each polarity its own atom; De Morgan is sound in three-valued logic,
so nothing else changes. `band` and `bor` need no care — they commute with the
collapse, because an unknown operand cannot turn a known outcome around.

The cost is that two polarities of one term become unrelated atoms, so a cube
holding both is not recognised as unsatisfiable (`Glushkov.zig` spots `x & !x`
only when the negation is a real `bNot`). Teaching the guard layer that a pair of
atoms is mutually exclusive would recover that without this library learning
anything about widths or unknown values, and is the intended move if state counts
ever justify it.

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
and `intersect` and `and` are products — `and` carrying, on top of the pairs, a
copy of each operand to run on alone in once the other has finished — so every
one of those is a blow-up vector. `Options.limit` caps the state count, checked
before anything is allocated, and `Diagnostic` names the node that tripped it.
Goto and non-consecutive repetition are rejected over an operand that is not a
`seqBoolean`.

`first_match` is a blow-up vector of a different kind: it determinises its
operand, so a state is a *subset* of the operand's states rather than a letter
occurrence, paired with the cube that reached it. The pairing is what keeps the
result homogeneous — the guard that says a subset is exactly what is live is
`⋀ g ∧ ⋀ ¬h`, which is not a cube, so the subset is split across the cells its
predecessor's guards cut the valuation space into, one branch per guard holding
and one per *literal* of it failing. Suppression is then that a subset holding
an exit of the operand has no successors at all. Reachability keeps this small
in practice, because a subset is only explored while no match has ended yet,
but the worst case is `|subsets| · |cells|` and the cap is what stands between
that and an allocation.

## Monitors

`sva.glushkov.assertion(gpa, &ir, index, .{}, null)` lowers a directive to an
`sva.Monitor`: an *obligation* automaton, and for an implication a *trigger*
automaton saying when to start one. Both are built against a single alphabet, so
a term the antecedent and the consequent share is evaluated once rather than
twice.

The two are tracked differently, and that asymmetry is the design. The trigger is
asked how many antecedent matches end this cycle, so one state vector holds every
attempt of it and each state carries the *set* of attempts with a thread there —
`Monitor.trigger_window` ids, rotating, and zero of them when no two matches can
end together, which is the common case and costs nothing. The obligation is
per-attempt: merging attempts is unsound, because one attempt's threads can die
while another's live and the union hides the failure. What does collapse is the
obligation itself — matches ending in one cycle impose the identical future, so
one instance of it serves them all and carries the number of attempts it stands
for. Reporting one verdict for three attempts is a wrong answer, not a coarse
one: LRM 16.14.1 starts an evaluation of the consequent per match.

How the pending attempts are held is `monitor.mode`, and there are two
disciplines because there are two kinds of obligation.

**`.ring`.** Each pending attempt gets its own state vector, and `monitor.depth`
says how many a backend must allocate. `depth` is the obligation's longest match,
which is exactly right: an attempt started in cycle `t` has either matched or run
out of live states by cycle `t + depth - 1`.

**`.counted`.** An obligation with no bounded longest match — `##[1:$] ack`, any
goto repetition, `always p`, `s_eventually p`, any `until` — has no such ring.
Its obligation is *determinised*, and totally: the death of an attempt becomes a
state rather than the absence of one, so one state fully characterises an
attempt's future and attempts sharing a state may share it. A backend then
carries one `u64` count per obligation state instead of a vector per attempt.
The count is what the verdict needs — two attempts is two failures, and a merged
vector could only ever report one. `monitor.depth` is zero and meaningless there.

Two rules come with it. A count moves along exactly **one** enabled transition,
the lowest-indexed: several can be enabled at once because a determinised state
is a `(subset, cell)` pair and two cells may share a valuation, and those
siblings are bisimilar. And failure is `automaton.dead`, not an empty vector: the
construction is total, so a count never falls off the automaton.

Two shapes are still refused. A trigger strictly *below* an unbounded operator —
`always (a |=> b)` — wants both disciplines at once and gets
`ObligationModeMixed`; a trigger *above* one is fine, so `req |=> s_eventually
ack` lowers. And the universal operators want a single-cycle operand:
`always p` and an `until`'s **left** operand say `p` holds at every cycle of a
window, which is a statement about overlapping attempts of `p` that one counter
cannot carry. That is `ObligationNotBounded`, naming the operand. Existential
positions — `s_eventually`'s operand, an `until`'s **right** operand — are
unrestricted.

An `and` is unbounded as soon as *either* side is, where an `intersect` needs
only one bounded side: an intersection's two ends coincide, so a bounded side
bounds the pair, but an `and` runs on to the later end. So
`req |=> (ack ##[1:$] done) and rdy` is counted while the same shape under
`intersect` is ringed.

`first_match` keeps a subset of its operand's matches, so it reports the
operand's bound: exact where the earliest match is also the longest on offer, and
an over-estimate otherwise. `first_match(ack[*1:$])` is really just `ack` and is
counted anyway — seeing that needs the longest path over the automaton, and the
bound is read off the tree.

Everything the property layer adds above the sequences lands in those same two
automata, so a backend gains no table and no opcode — with one exception.
`monitor.aborts` is a small array of `Monitor.Abort`, each a root index, an
effect (`accept` discharges the attempt, `reject` fails it), a scope (whether
firing resolves the whole attempt or only clears the mask), a `covers_trigger`
flag, a `sampled` flag, and a state mask read with `monitor.maskOf(i)`. Apply
them outermost first — the array runs the other way — after seeding a new
attempt and before taking verdicts. One with `covers_trigger` also acts earlier,
on the antecedent: it resolves every attempt still matching one, instead of
seeding.

`Monitor.State` is a reference implementation of the whole per-cycle protocol,
which is the contract a backend reproduces rather than something it calls.

## The C API

`include/sva.h`, linked against `libsva.a`. One export per Zig entry point.

An `sva_ir`, an `sva_monitor` and an `sva_run` are incomplete types: libsva
allocates one and hands back a pointer, and you hand that pointer back to be
destroyed. libsva picks its own allocator, so there is nothing to configure and
nothing to pass, and nothing on your side depends on how big any of them is.

They used to be byte arrays in the caller's storage, sized by constants in the
header. That contract could only ever be checked from one side: a header from a
different build compiled fine against the archive and then overran the caller's
stack, silently. `SVA_ABI_VERSION` and `sva_abi_version()` replace it, because a
version can be compared from both sides — check it once at startup.

An `sva_automaton` is reached only by pointer and owned by its monitor. Reading
a monitor is direct: an accessor hands back a pointer to a whole array and you
index it — one call per table, not per element.

```c
if (sva_abi_version() != SVA_ABI_VERSION) return 1; /* stale header or archive */

sva_ir *ir = sva_ir_create();
if (ir == NULL) goto fail;

sva_node req, ack, consequent, property, reset;
if (sva_bool_atom(ir, 0, &req) != SVA_OK) goto fail;
if (sva_seq_boolean(ir, req, &req) != SVA_OK) goto fail;
if (sva_bool_atom(ir, 1, &ack) != SVA_OK) goto fail;
if (sva_seq_boolean(ir, ack, &ack) != SVA_OK) goto fail;
if (sva_prop_sequence(ir, SVA_WEAK, ack, &consequent) != SVA_OK) goto fail;
if (sva_prop_implies(ir, SVA_NONOVERLAPPING, req, consequent, &property) != SVA_OK) goto fail;
if (sva_bool_const(ir, false, &reset) != SVA_OK) goto fail; /* never resets */

const sva_directive d = {
    .id = 0,
    .kind = SVA_ASSERT,
    .clock = SVA_CLOCK_NONE,
    .reset = reset,
    .property = property,
};
uint32_t index;
if (sva_ir_add(ir, &d, &index) != SVA_OK) goto fail;

sva_monitor *monitor;
if (sva_lower(&monitor, ir, index, NULL, NULL) == SVA_OK) { /* emit code */ }
```

Anything that can fail returns a status and delivers its result through an out
parameter, written only on `SVA_OK`. For a builder the only failure is an
allocation failure, and there is no accumulated state to check later: a caller
that ignores a return leaves its handle unwritten.

Nothing else is checked. A **malformed tree** — an operand from the wrong layer, a
reversed range — is a bug in the caller's lowering, not a property of what it is
compiling; there is no source location to report, because the location is the
caller's own code. The Zig assertions stand and the wrapper does not check twice.
**Legal input libsva cannot lower yet** — a mode mix, a multi-cycle operand under
a universal operator, a nested
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
