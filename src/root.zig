//! libsva — a SystemVerilog Assertions engine.
//!
//! Downstream code pulls this in with `@import("sva")`.
//!
//! This first cut is the **core IR**: a flattened-tree, content-hashable,
//! serializable representation of the concurrent-assertion subset (a boolean →
//! sequence → property layering). The lowering to NFAs and the coroutine /
//! formal backends that consume them land in follow-up changes; see the
//! roadmap in the README.

const std = @import("std");

/// The intermediate representation: a flattened-tree IR that assertions are
/// built into and that the lowering will consume. See `Ir.zig`.
pub const Ir = @import("Ir.zig");

test {
    std.testing.refAllDecls(@This());
}
