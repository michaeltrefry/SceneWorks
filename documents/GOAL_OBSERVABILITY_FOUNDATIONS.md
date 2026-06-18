# GOAL: Observability foundations

> **Type:** Goal brief for an autonomous working thread (`/goal`).
> **Scope:** Foundational only — logging backbone, declared levels, API-500/auth visibility. **No metrics, no new tracing/correlation IDs.**
> **Target:** Desktop (Tauri sidecars) **and** headless server/Docker, equally.
> **Status:** Done. `tracing` + `tracing-subscriber` back all Rust crates via
> `sceneworks_core::observability::init_logging[_with_buffer]`, called from every
> binary `main`; output is format-adaptive (`SCENEWORKS_LOG_FORMAT`, `auto`=JSON when
> non-TTY); levels are declared at the call site and honored verbatim by `session_log`;
> `api_error` (5xx=error / 4xx=debug) and `auth_rejected` (warn) are emitted and
> documented. `cargo fmt --check`, `cargo clippy --workspace --all-targets`, and
> `cargo test --workspace` all pass.
> **Companion doc:** [`docs/observability.md`](../docs/observability.md) — the operator's guide to the existing event vocabulary and Logs screen. This goal must keep every claim in that doc true.

---

## 1. Why this exists

A survey of the codebase found that SceneWorks has **thoughtful failure-path handling** for its job pipeline (panic containment via `catch_unwind` in `generator_cache.rs`, worker-crash attribution in `workers.rs`, user-visible attributed job errors, a redacting ring-buffer Logs screen) but **lacks the foundational logging plumbing** a server deployment expects:

- **No logging framework.** ~200 raw `println!`/`eprintln!` sites; no `tracing`/`log`/`tracing-subscriber` in any of the 5 `Cargo.toml`s. A deliberate JSON `emit_event` convention exists but is used at only ~6 sites.
- **Log levels are *inferred*, not declared.** `sceneworks_core::session_log::infer_level` classifies a line as `error`/`warn`/`info` by string heuristics (does the event name end in `_failed`? is there an `error` field?). A plain `eprintln!` that doesn't match the regex is mis-filed as `info`, so filtering the Logs screen by `level=error` silently drops real errors.
- **API-side 500s are invisible.** `ApiError::internal(...)` returns `{detail}` to the client but logs **nothing**. An untyped internal failure leaves no operator trace.
- **Auth/CORS rejections are invisible.** `access_control` returns 401/403 with no log line.

This goal closes those four gaps and routes the existing event vocabulary through a real logging API — **without regressing** the parts that already work well.

## 2. The goal, in one sentence

Introduce `tracing` + `tracing-subscriber` as the single logging backbone for all Rust crates, emitting **format-adaptive** output (JSON when captured/headless, pretty when at an interactive TTY), so log levels are **declared at the call site** instead of inferred — while preserving the existing event vocabulary, secret redaction, ring-buffer Logs screen, and `job_id` correlation, and adding error-level logs for API 500s and auth rejections.

## 3. Non-goals (do **not** do these — they are deliberately out of scope)

- ❌ **No metrics.** No Prometheus, OpenTelemetry, statsd, `/metrics` endpoint, or failure-rate counters.
- ❌ **No new distributed tracing / spans / correlation IDs.** Keep `job_id`/`worker_id` as the existing cross-process correlation handle; do not add `x-request-id` propagation. (`#[instrument]` may be used *locally* for ergonomics, but adding a request-tracing system is out of scope.)
- ❌ **No deep health/readiness probe rework.** Leave `/api/v1/health` as-is.
- ❌ **No rewrite of the failure-path logic** (panic containment, crash attribution, job-error persistence). It works — route its logging through `tracing`, don't redesign it.
- ❌ **No swallowed-`let _ =`-result cleanup beyond logging.** If a `let _ = fail_job(...)` path is touched, it's only to add an `error!` log when the inner result is `Err`; do not change recovery semantics.

> These are the bounds. If the work seems to require crossing one of them, **stop and flag it** rather than expanding scope.

## 4. Workstreams

### WS1 — `tracing` backbone

- Add `tracing` and `tracing-subscriber` (+ `tracing-subscriber`'s `json`, `env-filter`, `fmt` features) to the workspace. Prefer a single workspace-level dependency set.
- Add **one shared init function** — suggest `sceneworks_core::observability::init_logging()` (new module) — that installs the subscriber. Call it from the `main` of each binary: `apps/rust-api`, `apps/rust-worker`, and any other entrypoint. Idempotent / safe to call once per process.
- Honor `RUST_LOG` via `EnvFilter`, with a sensible default (e.g. `info,sceneworks=debug`).

### WS2 — Format-adaptive output (serves desktop **and** server from one binary)

- **Selection rule:** `SCENEWORKS_LOG_FORMAT = json | pretty | auto` (default `auto`). In `auto`, emit **pretty** when `stdout` is a TTY (interactive `cargo run`), **JSON** otherwise (Tauri sidecar capture, Docker, piped). This makes desktop sidecars and headless servers both emit JSON — which is exactly what the ring buffer and log ingestion want — while developers at a terminal still get readable output.
- **JSON line shape must stay parseable by `session_log`.** The ring buffer ingests sidecar stdout line-by-line and expects one JSON object per line. Preserve the existing envelope **`{ event, level, reportedAt, ...fields }`** (camelCase, matching `docs/observability.md`). Recommended approach: a thin custom `tracing_subscriber` JSON formatter/layer that renders `event` (from a structured `event = "..."` field or the event's target), `level` (from the tracing level — **now authoritative**), `reportedAt` (timestamp), and the remaining fields flattened. Do **not** silently switch to tracing's native JSON envelope without updating both `session_log` parsing and `docs/observability.md`.
- **Secret redaction must survive.** `session_log::redact_secrets` runs on ingestion today; keep it. If any redaction needs to move to the emit side, ensure tokens/api-keys/bearer headers are still scrubbed before they hit stdout or a log file.

### WS3 — Declared levels replace inference

- Migrate the structured-event helpers (`sceneworks_worker::emit_json` / `emit_event`, the API's `mlx_route_decision` emitter) and the **high-value error/warn `eprintln!` sites** to `tracing::{error,warn,info,debug}!` with structured fields. The full ~200 `println!` sweep is **not** required — prioritize: (a) every existing structured event, (b) every `*_failed`/`*_error` site, (c) anything currently relied on by `docs/observability.md`.
- **Preserve event names** as a field/target so the vocabulary in `docs/observability.md` stays valid, e.g.
  `info!(event = "mlx_route_decision", decision = ?d, reason = %r, model = %m, job_id = %id);`
- Update `session_log::infer_level` so that when a line carries a **declared `level`**, that level is used verbatim; fall back to the existing heuristic only for legacy/plain lines that lack one. The Logs-screen `level` filter must become trustworthy.

### WS4 — Make API 500s and auth rejections visible

- `ApiError::internal(...)` (or its `IntoResponse`): emit `error!(event = "api_error", status, detail, ...)` **before** returning the response, so every 5xx leaves a server-side trace. Avoid double-logging typed 4xx domain errors that are expected/normal — log 5xx at `error`, optionally 4xx at `debug`.
- `auth::access_control`: on a rejected request emit `warn!(event = "auth_rejected", path, reason, status)` (401/403). Do **not** log the token/secret (redaction + don't-include).

### WS5 — Keep the surface working & update docs

- The Logs screen (`apps/web/src/screens/LogsScreen.jsx`), `get_session_logs` (desktop), and `GET /api/v1/logs` (`apps/rust-api/src/logs.rs`) must work end-to-end with declared levels, unchanged from the user's perspective except that the `level` filter is now accurate.
- Update `docs/observability.md`: a new "Logging backbone" section describing `init_logging`, `SCENEWORKS_LOG_FORMAT`/`RUST_LOG`, the TTY rule, and that levels are now declared. Add the two new events (`api_error`, `auth_rejected`) to the vocabulary. Update this goal doc's **Status** to reflect completion.

## 5. Invariants (must remain true)

1. Secret redaction still scrubs tokens / api-keys / bearer / authorization before anything is persisted or surfaced.
2. The `LogEntry` shape and the Logs-screen source/level/search filters keep working; expanding a row still shows the raw structured event.
3. Every event documented in `docs/observability.md` is still emitted under its existing name (`mlx_route_decision`, `claim_lock_contention`, `image_inference_*`, `image_pipeline_load_*`, `mlx_generator_cache_idle_evicted`).
4. `job_id` / `worker_id` remain present on job-lifecycle events for cross-process correlation.
5. No new heavyweight runtime dependency beyond the `tracing` ecosystem.
6. `cargo fmt --check`, `cargo clippy`, and the existing test suites pass. (Repo has a rustfmt pre-commit hook — keep it green.)
7. Desktop sidecars and Docker both emit JSON under `auto`; an interactive terminal gets pretty output.

## 6. Acceptance criteria (the definition of done — each is checkable)

- [x] `tracing` + `tracing-subscriber` are workspace dependencies; a single `init_logging()` is called from every Rust binary's `main`.
- [x] `SCENEWORKS_LOG_FORMAT=json|pretty|auto` works; `auto` selects JSON for non-TTY stdout and pretty for a TTY. `RUST_LOG` filtering works.
- [x] Running the API headless and hitting `GET /api/v1/logs` returns entries whose `level` is the **declared** level (not inferred). Forcing an internal error produces an `error`-level `api_error` entry visible in the logs; an unauthorized request produces a `warn`-level `auth_rejected` entry. *(Verified live: a 401 yields `warn` `auth_rejected`; a typed 4xx yields `debug` `api_error` that is **not** re-promoted to error; 5xx logs at `error` via the same `IntoResponse` branch.)*
- [x] In the desktop build, the **System → Logs** screen still tails, filters by source/level (accurately now), searches, and expands raw events — and `mlx_route_decision` / `claim_lock_contention` rows are still highlighted. *(No frontend change; `LogEntry` shape unchanged, levels now declared.)*
- [x] All structured events from `docs/observability.md` still appear under their documented names with their documented fields. Secrets are still redacted.
- [x] `cargo fmt --check`, `cargo clippy`, and `cargo test` (all crates) pass; web tests (`apps/web`) unaffected.
- [x] `docs/observability.md` updated (Logging backbone section + `api_error`/`auth_rejected` events); this doc's Status set to Done.

## 7. Verification commands

```bash
# Build / lint / format / test
cargo build --workspace
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace

# Format selection (server/headless = JSON)
SCENEWORKS_LOG_FORMAT=json cargo run -p sceneworks-api 2>&1 | head   # expect one JSON object per line, with "level"
# Interactive (TTY) auto -> pretty
cargo run -p sceneworks-api                                          # expect human-readable colored output

# 500 visibility: trigger an internal error path, then confirm it logged
curl -s localhost:<port>/api/v1/logs | jq '.[] | select(.event=="api_error")'
# Auth visibility: hit a protected route without a token
curl -s -o /dev/null -w '%{http_code}\n' localhost:<port>/api/v1/<protected>
curl -s localhost:<port>/api/v1/logs | jq '.[] | select(.event=="auth_rejected")'

# Declared-level trust: every error entry should carry an explicit level, none inferred from text
curl -s localhost:<port>/api/v1/logs | jq '.[] | select(.level=="error") | {event,level,message}'
```

Web tests (frontend untouched but verify no regression): `cd apps/web && npm test`.

## 8. Suggested sequencing (small, reviewable PRs)

1. **WS1+WS2** — backbone + format-adaptive subscriber + `init_logging`, wired into binaries. No behavior change to events yet (still `println!`). Verifiable on its own.
2. **WS3** — migrate structured events + error/warn sites to `tracing`; flip `infer_level` to prefer declared level. The big correctness win.
3. **WS4** — API-500 and auth-rejection logging.
4. **WS5** — docs + final verification pass.

## 9. Deferred (explicitly *next*, not now)

Metrics/`/metrics` endpoint, request/correlation IDs (`x-request-id`), deeper readiness probe, and a broader `println!` sweep are intentionally deferred. If they come up, note them for a follow-up — do not pull them into this goal.

---

## Recommended `/goal` launch condition

Open a fresh session in this repo and run:

```
/goal Implement the observability-foundations brief in documents/GOAL_OBSERVABILITY_FOUNDATIONS.md. Done when: tracing + tracing-subscriber back all Rust crates via a single init_logging() called from every binary main; SCENEWORKS_LOG_FORMAT=json|pretty|auto works with auto=JSON-when-non-TTY; the structured event vocabulary in docs/observability.md is emitted via tracing under its existing names with secrets still redacted; session_log uses declared levels (falling back to heuristics only for legacy lines); API 5xx emit an error-level api_error log and auth rejections emit a warn-level auth_rejected log; every acceptance-criteria checkbox in the brief is checked; and cargo fmt --check, cargo clippy --workspace, and cargo test --workspace all pass. Stay within the brief's non-goals (no metrics, no /metrics, no new correlation IDs). Stop after 40 turns if blocked and report what's left.
```
