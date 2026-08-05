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
            // A C consumer's own build is very likely position-independent (a PIE
            // executable is the default on most toolchains), and a non-PIC archive
            // cannot be linked into one.
            .pic = true,
        }),
    });
    // A static library handed to a C or C++ consumer must be self-contained: it is
    // linked by that project's linker, not Zig's, so nothing supplies Zig's
    // compiler-rt for it. Without this, `__zig_probe_stack` (and any wide-integer
    // helper the code happens to need) comes out undefined at the consumer's link
    // step -- which no `zig build test` can catch, because those link with Zig.
    lib.bundle_compiler_rt = true;
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

    // Proves the tables an `sva_monitor` exposes are enough to *compile* the
    // per-cycle protocol, not just interpret it: it reimplements the protocol the
    // way a code generator must - constants baked in, the state vector one machine
    // integer, the bit iteration in `advance` transposed into a fold over
    // statically-known states - and differential-tests that against `Monitor.State`
    // over random traces.
    const backend = b.addExecutable(.{
        .name = "sva-c-backend",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    backend.root_module.addCSourceFile(.{
        .file = b.path("test/backend.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Wconversion", "-Werror" },
    });
    backend.root_module.addIncludePath(b.path("include"));
    backend.root_module.linkLibrary(lib);
    const run_backend = b.addRunArtifact(backend);

    // Compiles the public header as C++ and drives the API from it. The C tests
    // above cannot catch a header that is not *includable* from C++ — `alignas`
    // versus `_Alignas` was one such break, found by a real consumer — and C++ is
    // the expected consumer, not an exotic one.
    const header = b.addExecutable(.{
        .name = "sva-cpp-header",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    header.root_module.addCSourceFile(.{
        .file = b.path("test/header.cpp"),
        .flags = &.{ "-std=c++17", "-Wall", "-Wextra", "-Wconversion", "-Werror" },
    });
    header.root_module.addIncludePath(b.path("include"));
    header.root_module.linkLibrary(lib);
    header.root_module.link_libcpp = true;
    const run_header = b.addRunArtifact(header);

    const test_step = b.step("test", "Run unit tests and the C / C++ tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_smoke.step);
    test_step.dependOn(&run_backend.step);
    test_step.dependOn(&run_header.step);
}
