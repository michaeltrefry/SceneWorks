# syntax=docker/dockerfile:1.7
# Shared multi-target build for the Rust API and Rust worker images. The builder
# stage (base image, workspace COPYs, cargo build) was previously copy-pasted
# between docker/rust-api.Dockerfile and docker/rust-worker.Dockerfile, differing
# only in the `-p` target and the runtime apt packages (sc-4284 / F-INFRA-7).
#
# Build a specific image with `--target` + `--build-arg BIN=…`; docker-compose
# sets both per service:
#   docker build -f docker/rust.Dockerfile --target rust-api   --build-arg BIN=sceneworks-rust-api   .
#   docker build -f docker/rust.Dockerfile --target rust-worker --build-arg BIN=sceneworks-rust-worker .

FROM rust:1-bookworm AS builder
# Which workspace binary to build (sceneworks-rust-api | sceneworks-rust-worker).
ARG BIN
WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates/sceneworks-core/Cargo.toml ./crates/sceneworks-core/Cargo.toml
COPY crates/sceneworks-worker/Cargo.toml ./crates/sceneworks-worker/Cargo.toml
COPY apps/rust-api/Cargo.toml ./apps/rust-api/Cargo.toml
COPY apps/rust-worker/Cargo.toml ./apps/rust-worker/Cargo.toml
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml

RUN mkdir -p \
      apps/desktop/src \
      apps/rust-api/src \
      apps/rust-worker/src \
      crates/sceneworks-core/src \
      crates/sceneworks-worker/src \
    && printf 'fn main() {}\n' > apps/desktop/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-api/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-worker/src/main.rs \
    && touch crates/sceneworks-core/src/lib.rs crates/sceneworks-worker/src/lib.rs

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker
# Copied purely to satisfy workspace membership (the desktop crate is in the
# workspace but is not built into either image).
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
COPY apps/desktop/build.rs ./apps/desktop/build.rs
# The builtin catalog: `sceneworks-core` embeds these manifests via `include_str!`
# so the API can seed an empty config dir, which means they must exist in the
# build context (not just the runtime bind mount) or the compile can't read them.
COPY config ./config

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build -p "${BIN}" --release \
    && mkdir -p /out \
    && cp "target/release/${BIN}" "/out/${BIN}"

# --- Rust API runtime ---------------------------------------------------------
FROM debian:bookworm-slim AS rust-api

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/sceneworks-rust-api /usr/local/bin/sceneworks-rust-api

CMD ["sceneworks-rust-api"]

# --- Rust worker runtime ------------------------------------------------------
FROM debian:bookworm-slim AS rust-worker

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/hf-cli \
    && /opt/hf-cli/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/hf-cli/bin/pip install --no-cache-dir "huggingface_hub[cli]>=0.36,<1"

ENV PATH="/opt/hf-cli/bin:${PATH}"

COPY --from=builder /out/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]

# --- Candle GPU worker build (CUDA; compute_80 PTX → sm_120) ------------------
# Separate builder: the candle backend needs the CUDA toolkit (nvcc) to compile
# candle-kernels, which the stock rust:bookworm builder above lacks. CUDA 12.9 is the
# toolchain the candle lane builds + validates against (server-candle-linux.yml + the
# dev box). CUDA_COMPUTE_CAP=80 emits compute_80 PTX the driver JITs forward to sm_120
# (RTX PRO 6000) — one binary covers Ampere→Blackwell, matching the Windows desktop
# bundle (build-sidecar.mjs) and the Linux candle CI lane. The backend-candle feature
# lives on the sceneworks-worker library crate, enabled through the thin binary
# (epic 5483 Phase 7 / sc-5503 — the Docker torch→candle cutover).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS candle-builder
ENV DEBIAN_FRONTEND=noninteractive
# build-essential + pkg-config for the CUDA/native build scripts; libssl-dev because
# native-tls (pulled transitively by the worker's deps) links system OpenSSL on Linux
# — the Windows host build uses schannel instead, so this only surfaces here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Rust toolchain. The default rustup profile ships rustfmt+clippy, satisfying
# rust-toolchain.toml (stable + those components).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ENV CUDA_PATH=/usr/local/cuda
ARG CUDA_COMPUTE_CAP=80
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
WORKDIR /app

# Dependency-graph layer (mirrors the builder above): COPY the manifests + stub
# entrypoints, then `cargo fetch` so the candle dependency tree (candle-gen +
# candle/cudarc, all public git deps) caches independently of source edits.
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates/sceneworks-core/Cargo.toml ./crates/sceneworks-core/Cargo.toml
COPY crates/sceneworks-worker/Cargo.toml ./crates/sceneworks-worker/Cargo.toml
COPY apps/rust-api/Cargo.toml ./apps/rust-api/Cargo.toml
COPY apps/rust-worker/Cargo.toml ./apps/rust-worker/Cargo.toml
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
RUN mkdir -p \
      apps/desktop/src apps/rust-api/src apps/rust-worker/src \
      crates/sceneworks-core/src crates/sceneworks-worker/src \
    && printf 'fn main() {}\n' > apps/desktop/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-api/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-worker/src/main.rs \
    && touch crates/sceneworks-core/src/lib.rs crates/sceneworks-worker/src/lib.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker
# Workspace-membership only (not built into this image).
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
COPY apps/desktop/build.rs ./apps/desktop/build.rs
# The builtin catalog, embedded via include_str! by sceneworks-core (see above).
COPY config ./config

# nvcc compiles every candle provider's CUDA kernels here (compiling needs no GPU).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build -p sceneworks-rust-worker --release \
        --features sceneworks-worker/backend-candle \
    && mkdir -p /out \
    && cp target/release/sceneworks-rust-worker /out/sceneworks-rust-worker

# --- Candle GPU worker runtime (CUDA-12) -------------------------------------
# The off-Mac torch replacement: Docker GPU inference runs on the native candle/CUDA
# worker, not the Python torch worker. The CUDA-runtime base provides cudart/cublas/
# cublasLt for candle; the `ort` CV-aux lanes (DWPose/YOLO/Real-ESRGAN, sc-6209) get a
# version-matched onnxruntime-gpu + its cuDNN-9 / cuFFT / nvJitLink / nvRTC deps from
# PyPI, dlopened via ORT_DYLIB_PATH (the `ort` crate links load-dynamic).
#
# ubuntu24.04 (Python 3.12) on purpose: onnxruntime-gpu >= 1.24 (the `ort` rc.12 floor
# / ORT API 24) ships no cp310 Linux wheel, so the 22.04 base's Python 3.10 caps at
# 1.23.2 and can't satisfy it; cp312 has 1.26.0. The builder stays on 22.04 (it needs
# no Python) — its older-glibc binary runs fine on 24.04 (glibc is backward-compatible).
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS rust-worker-candle
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: candle video lanes encode mp4. python3/venv: stage onnxruntime-gpu + the hf
# CLI for model downloads. libgomp1: onnxruntime's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu + the CUDA-12 deps its providers_cuda needs that the CUDA-runtime
# base doesn't ship — cuDNN-9 (incl. lazily-loaded sub-engines), cuFFT, nvJitLink,
# nvRTC. onnxruntime-gpu does NOT declare these as hard deps, so request them
# explicitly (sc-6209). 1.26.0 matches the `ort` crate rc.12 (ORT API 24); validated
# on RTX PRO 6000 with cuDNN-cu12 9.23. Also installs the hf CLI for downloads.
ENV ORT_PY_SITE=/opt/ort/lib/python3.12/site-packages
ARG ONNXRUNTIME_GPU_VERSION=1.26.0
RUN python3 -m venv /opt/ort \
    && /opt/ort/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/ort/bin/pip install --no-cache-dir \
        "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}" \
        nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12 \
        "huggingface_hub[cli]>=0.36,<1" \
    && ORT_SO="$(ls ${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so* | head -1)" \
    && test -n "${ORT_SO}" \
    && ln -sf "${ORT_SO}" "${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so"

# Point the `ort` crate (load-dynamic) at the staged onnxruntime, and tell ort_cuda
# where the CUDA-12 runtime (base) + cuDNN-9 (pip wheel) live. LD_LIBRARY_PATH is the
# Linux analogue of the Windows PATH-prepend in ort_cuda::preload_cuda_dylibs — the
# dynamic linker resolves the providers' CUDA/cuDNN deps + cuDNN's lazy sub-engines.
ENV ORT_DYLIB_PATH=${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so
ENV SCENEWORKS_ORT_CUDA_DIR=/usr/local/cuda/lib64
ENV SCENEWORKS_ORT_CUDNN_DIR=${ORT_PY_SITE}/nvidia/cudnn/lib
ENV LD_LIBRARY_PATH=${ORT_PY_SITE}/onnxruntime/capi:${ORT_PY_SITE}/nvidia/cudnn/lib:${ORT_PY_SITE}/nvidia/cufft/lib:${ORT_PY_SITE}/nvidia/nvjitlink/lib:${ORT_PY_SITE}/nvidia/cuda_nvrtc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH="/opt/ort/bin:${PATH}"

COPY --from=candle-builder /out/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]
