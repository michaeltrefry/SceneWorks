FROM rust:1-bookworm AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker

RUN cargo build -p sceneworks-rust-worker --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/hf-cli \
    && /opt/hf-cli/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/hf-cli/bin/pip install --no-cache-dir "huggingface_hub[cli]>=0.36,<1"

ENV PATH="/opt/hf-cli/bin:${PATH}"

COPY --from=builder /app/target/release/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]
