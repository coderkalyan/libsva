//! libsva — a concurrent assertion engine.
//!
//! Downstream code pulls this in with `@import("sva")`.
//!
//! Concurrent assertions are temporal properties checked over a trace of
//! discrete cycles. `Ir` is the front-end-facing representation of them: a
//! flattened-tree, content-hashable, serializable boolean → sequence →
//! property layering over opaque, front-end-owned atoms, so it is tied to no
//! particular assertion syntax. `Glushkov` lowers a sequence to an `Nfa`, the
//! position automaton a backend runs, and a directive to a `Monitor`, which
//! pairs the automata with the obligation tracking a property needs. The
//! runtime and formal backends land in follow-up changes; see the README.

const std = @import("std");

/// The intermediate representation assertions are built into. See `Ir.zig`.
pub const Ir = @import("Ir.zig");

/// The automaton a sequence lowers to, and the primitives that run it.
pub const Nfa = @import("Nfa.zig");

/// A compiled assertion: the automata plus the obligation tracking around them.
pub const Monitor = @import("Monitor.zig");

/// The lowering from `Ir` to `Nfa` and `Monitor`. See `Glushkov.zig`.
pub const glushkov = @import("Glushkov.zig");

test {
    std.testing.refAllDecls(@This());
}

/// The C API. Referenced so its `export`s are part of the library.
pub const c = @import("c.zig");
