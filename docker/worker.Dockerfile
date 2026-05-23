FROM python:3.12-slim AS builder

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTORCH_SPEC=torch>=2.8,<2.9
ARG PYTORCH_AUDIO_SPEC=torchaudio>=2.8,<2.9
# torchvision is required by the vendored sensenova_u1 (SenseNova-U1 T2I). Install
# it from the same index as torch so it's an ABI-matched build — a CPU torchvision
# against a CUDA torch breaks with "operator torchvision::nms does not exist".
ARG PYTORCH_VISION_SPEC=torchvision>=0.23,<0.24
ARG INCLUDE_LTX_PIPELINES=1
# Real person detection/tracking/segmentation backends (ultralytics + SAM2) for
# the Replace Person workflow (epic sc-1090). Opt-in like the LTX pipelines.
ARG INCLUDE_PERSON_BACKENDS=1
# Microsoft Lens runs in a SEPARATE venv (/opt/lens-venv) because it needs
# transformers 5.x + diffusers 0.38, which are incompatible with the main venv's
# transformers 4.x stack that native LTX-2.3 (ltx-core's Gemma-3 integration)
# requires. Built from requirements-lens.txt with its own cu128 torch 2.11 +
# torchvision; invoked out-of-process by scene_worker/lens_runner.py. Opt-in.
ARG INCLUDE_LENS=1
ARG LENS_PYTORCH_SPEC=torch>=2.11,<2.12
ARG LENS_PYTORCH_VISION_SPEC=torchvision>=0.26,<0.27

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY apps/worker/requirements.txt ./requirements.txt
COPY apps/worker/requirements-ltx.txt ./requirements-ltx.txt
COPY apps/worker/requirements-person.txt ./requirements-person.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-compile --index-url "${PYTORCH_INDEX_URL}" "${PYTORCH_SPEC}" "${PYTORCH_AUDIO_SPEC}" "${PYTORCH_VISION_SPEC}" \
    && pip freeze | grep -E '^(torch|torchaudio|torchvision|nvidia-|triton)==.*' > /tmp/torch-constraints.txt \
    && pip install --no-cache-dir --no-compile -c /tmp/torch-constraints.txt -r requirements.txt \
    && if [ "${INCLUDE_LTX_PIPELINES}" = "1" ]; then pip install --no-cache-dir --no-compile -c /tmp/torch-constraints.txt -r requirements-ltx.txt; fi \
    && if [ "${INCLUDE_PERSON_BACKENDS}" = "1" ]; then pip install --no-cache-dir --no-compile -c /tmp/torch-constraints.txt -r requirements-person.txt; fi \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
        /opt/venv/lib/python3.12/site-packages/torch/test \
        /opt/venv/lib/python3.12/site-packages/torch/share \
    && (find /opt/venv -type f -name "*.so*" -exec strip --strip-unneeded {} + || true)

# Isolated Lens sidecar venv. Separate venv + separate RUN so a Lens build
# failure can't poison the main worker venv layer, and so this stays cached when
# only the main requirements change. torchvision is installed from the same
# (cu128) index as torch — a CPU torchvision ABI-mismatches CUDA torch
# ("operator torchvision::nms does not exist"), which breaks diffusers 0.38's
# Flux.2 VAE import. When INCLUDE_LENS=0 an empty dir is created so the runtime
# COPY still succeeds.
COPY apps/worker/requirements-lens.txt ./requirements-lens.txt
RUN if [ "${INCLUDE_LENS}" = "1" ]; then \
        python -m venv /opt/lens-venv \
        && /opt/lens-venv/bin/pip install --no-cache-dir --upgrade pip \
        && /opt/lens-venv/bin/pip install --no-cache-dir --no-compile --index-url "${PYTORCH_INDEX_URL}" "${LENS_PYTORCH_SPEC}" "${LENS_PYTORCH_VISION_SPEC}" \
        && /opt/lens-venv/bin/pip freeze | grep -E '^(torch|torchvision|nvidia-|triton)==.*' > /tmp/lens-torch-constraints.txt \
        && /opt/lens-venv/bin/pip install --no-cache-dir --no-compile -c /tmp/lens-torch-constraints.txt -r requirements-lens.txt \
        && find /opt/lens-venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
        && rm -rf /opt/lens-venv/lib/python3.12/site-packages/torch/include \
            /opt/lens-venv/lib/python3.12/site-packages/torch/test \
            /opt/lens-venv/lib/python3.12/site-packages/torch/share \
        && (find /opt/lens-venv -type f -name "*.so*" -exec strip --strip-unneeded {} + || true); \
    else mkdir -p /opt/lens-venv; fi

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Triton (bundled with the CUDA torch wheels) JIT-compiles GPU kernels at
# runtime — notably the fp8 path used by LTX-2.3 — and needs a host C compiler.
# The slim runtime image ships none, so fp8 jobs failed with
# "Failed to find C compiler. Please specify via CC environment variable...".
# Triton bundles its own ptxas and CUDA headers and Python.h ships with the
# base image, so gcc plus libc headers are all that is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Lens sidecar venv (empty dir when INCLUDE_LENS=0). LensTurboAdapter runs
# /opt/lens-venv/bin/python scene_worker/lens_runner.py out-of-process; override
# the interpreter path with SCENEWORKS_LENS_PYTHON.
COPY --from=builder /opt/lens-venv /opt/lens-venv

COPY apps/worker ./apps/worker
COPY packages/shared ./packages/shared
ENV PYTHONPATH=/app/apps/worker:/app/packages/shared

ENTRYPOINT ["python", "-m", "scene_worker"]
