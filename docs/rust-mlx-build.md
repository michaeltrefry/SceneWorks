# Building the native MLX Rust GPU worker (macOS)

The Rust GPU worker (`crates/sceneworks-worker`) links the [`mlx-gen`](https://github.com/michaeltrefry/mlx-gen)
engine (epic 2337) as a `cfg(target_os = "macos")` Cargo dependency and runs image
(and, later, video) generation **in-process** — no Python adapter, no sidecar venv,
no subprocess. This is the consumer side of epic 3018.

`mlx-gen` and its `mlx-rs` fork are **public**, git-pinned by SHA, so Linux and
Windows builds *resolve* them but never *compile* them (the target gate excludes
them). Only macOS compiles MLX from source.

## Requirements (macOS only)

- **macOS 26.2 or newer** at runtime for the NAX fast path (Apple matrix-unit
  kernels). The app's runtime floor is pinned at cutover (sc-3032).
- **Full Xcode + the Metal Toolchain** (`xcode-select -p` must point at Xcode, not
  the Command Line Tools; `xcrun --find metal` must resolve).
- A recent stable Rust toolchain (`rust-toolchain.toml` pins `stable`).

## The deployment-target build seam

`/.cargo/config.toml` pins `MACOSX_DEPLOYMENT_TARGET = "26.2"`. This **must** live in
the SceneWorks workspace: Cargo does not read a dependency's `.cargo/config.toml`, so
mlx-gen's own 26.2 pin does not travel to this consumer. If the pin is missing, MLX's
`mlx-sys` build.rs floors the target at macOS 14, the NAX kernels compile out
(`-DMLX_METAL_NO_NAX`), and the Mac path regresses ~2.5×. Worse, at 26.0 the 16-bit
kernels miscompile to garbage. The `nax_guard` integration test is the loud tripwire
for a slip:

```sh
cargo test -p sceneworks-worker --test nax_guard -- --nocapture   # needs macOS >= 26.2
```

The pin is **not forced**, so CI (and you, locally) can override it via the
environment — GitHub's hosted macOS runners cap at SDK 15:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build -p sceneworks-worker   # correctness-only, no NAX
```

> Note: `mlx-sys`'s build.rs has no `rerun-if-env-changed`, so changing the
> deployment target needs a clean rebuild of `pmetal-mlx-sys` to take effect.

## Heavy-recompile mitigation (compile MLX once)

Building MLX from source is the slow part of a clean build, and a fresh git worktree
gets its own `target/`, recompiling MLX from scratch. To share compiled artifacts
across worktrees/clean builds, opt in via the environment (not pinned in the committed
config, so machines/CI lanes without the tool are unaffected):

```sh
brew install sccache
export RUSTC_WRAPPER=sccache
export CARGO_TARGET_DIR=~/.cache/sceneworks-target
```

## Local mlx-gen co-development

The dependency is git-pinned by SHA. To iterate against a local checkout without the
push-and-bump cycle, add a workspace-root `[patch]` (do **not** commit it — a path
that is absent on CI breaks resolution):

```toml
# Cargo.toml (workspace root), local only
[patch."https://github.com/michaeltrefry/mlx-gen"]
mlx-gen = { path = "../mlx-gen" }
mlx-gen-z-image = { path = "../mlx-gen/mlx-gen-z-image" }
```

When a co-dev change lands in mlx-gen, bump the `rev` in
`crates/sceneworks-worker/Cargo.toml` (and the matching `mlx-rs` rev in
`[dev-dependencies]`) to the new SHA and drop the patch.
