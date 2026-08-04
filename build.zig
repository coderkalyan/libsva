const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // What a Zig consumer imports: `@import("sva")`.
    const mod = b.addModule("sva", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // What a C consumer links. Its root is `src/c.zig` rather than the Zig
    // module, because Zig analyses lazily: behind a `pub const` the `export`
    // declarations are never reached and the archive comes out empty.
    // `include/sva.h` declares them, and the smoke test below is what holds the
    // two together.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "sva",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.installHeader(b.path("include/sva.h"), "sva.h");
    b.installArtifact(lib);

    // `zig build test` — the unit tests, plus the C smoke test.
    const lib_tests = b.addTest(.{ .root_module = mod });
    const run_lib_tests = b.addRunArtifact(lib_tests);

    // Compiles the public header as C and drives the whole API against the
    // library. This is the only thing that proves the header and the exports
    // agree, so it is part of `test` rather than a separate step.
    const smoke = b.addExecutable(.{
        .name = "sva-c-smoke",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    smoke.root_module.addCSourceFile(.{
        .file = b.path("test/smoke.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Wconversion", "-Werror" },
    });
    smoke.root_module.addIncludePath(b.path("include"));
    smoke.root_module.linkLibrary(lib);
    const run_smoke = b.addRunArtifact(smoke);

    const test_step = b.step("test", "Run unit tests and the C smoke test");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_smoke.step);
}
