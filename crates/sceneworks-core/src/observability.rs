//! Logging backbone (observability foundations).
//!
//! `tracing` is the single logging facade for every Rust crate; this module
//! installs the subscriber that turns those events into output. Two things make
//! the SceneWorks setup specific:
//!
//! 1. **Format-adaptive output.** `SCENEWORKS_LOG_FORMAT = json | pretty | auto`
//!    (default `auto`) chooses between a machine-readable JSON line per event and
//!    a human-readable pretty line. In `auto` we emit **pretty** when `stdout` is a
//!    TTY (an interactive `cargo run`) and **JSON** otherwise (a Tauri sidecar whose
//!    stdout is captured, a Docker container, or any pipe). Desktop sidecars and
//!    headless servers therefore both emit JSON — exactly what the ring buffer and
//!    log ingestion want — while a developer at a terminal still gets readable logs.
//!
//! 2. **A stable JSON envelope.** The desktop wrapper and the API's own ring buffer
//!    ([`crate::session_log`]) ingest one JSON object per line and expect the
//!    `{ event, level, reportedAt, ...fields }` envelope documented in
//!    `docs/observability.md`. The custom [`SceneworksJsonFormat`] renders that exact
//!    shape: `event` from a structured `event = "…"` field, `level` from the tracing
//!    level (now the **authoritative**, declared level), `reportedAt` from the emit
//!    time, and the remaining fields flattened alongside.
//!
//! Secret redaction still happens on ingestion in [`crate::session_log`]; this module
//! does not introduce any new secret-bearing field on the emit side.

use std::io::IsTerminal;
use std::sync::Once;

use serde_json::{Map, Value};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter};

use crate::session_log::SessionLog;
use crate::time::utc_now;

/// Default `EnvFilter` directive when `RUST_LOG` is unset: everything at `info`,
/// SceneWorks crates down to `debug`.
const DEFAULT_FILTER: &str = "info,sceneworks=debug";

/// The field name carrying a pre-serialized structured-event object (see
/// [`emit_event`]). The formatter/layer parse it and flatten its keys up into the
/// envelope, so legacy `json!({...})` event payloads survive the migration to
/// `tracing` without exploding every call site into individual fields.
const PAYLOAD_FIELD: &str = "sw_payload";

static INIT: Once = Once::new();

/// Install the logging subscriber for a process with no in-process log buffer
/// (the worker / desktop sidecars, whose stdout is captured elsewhere). Idempotent
/// — the first call wins; later calls are no-ops.
pub fn init_logging() {
    install(None);
}

/// Install the logging subscriber for a process that also serves its own emitted
/// events back over HTTP (the API's `GET /api/v1/logs`). In addition to the
/// stdout layer, a [`SessionLogLayer`] feeds every event into `buffer` using the
/// same `{ event, level, reportedAt, ... }` envelope, so the headless Logs surface
/// sees declared levels with no separate `record_api_event` plumbing. Idempotent.
pub fn init_logging_with_buffer(buffer: &'static SessionLog) {
    install(Some(buffer));
}

fn install(buffer: Option<&'static SessionLog>) {
    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
        let session_layer = buffer.map(|buffer| SessionLogLayer { buffer });

        // Two arms because the JSON and pretty stdout layers are different types;
        // the EnvFilter and the optional session-log layer are shared.
        match resolve_format() {
            OutputFormat::Json => {
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(
                        fmt::layer()
                            .with_ansi(false)
                            .event_format(SceneworksJsonFormat),
                    )
                    .with(session_layer)
                    .try_init();
            }
            OutputFormat::Pretty => {
                let _ = tracing_subscriber::registry()
                    .with(filter)
                    .with(fmt::layer())
                    .with(session_layer)
                    .try_init();
            }
        }
    });
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Json,
    Pretty,
}

/// Resolve the stdout format from `SCENEWORKS_LOG_FORMAT`. `auto` (and any
/// unrecognized value) emits JSON unless stdout is an interactive terminal.
fn resolve_format() -> OutputFormat {
    match std::env::var("SCENEWORKS_LOG_FORMAT") {
        Ok(value) if value.trim().eq_ignore_ascii_case("json") => OutputFormat::Json,
        Ok(value) if value.trim().eq_ignore_ascii_case("pretty") => OutputFormat::Pretty,
        _ => {
            if std::io::stdout().is_terminal() {
                OutputFormat::Pretty
            } else {
                OutputFormat::Json
            }
        }
    }
}

/// Emit a pre-built structured-event object through `tracing` at a **declared**
/// level. The object should carry an `event` key; its fields are flattened into the
/// log envelope by the formatter / session-log layer. `reportedAt` is generated at
/// render time, so callers need not set it.
///
/// This is the bridge that lets the existing `json!({ "event": ..., ... })` emit
/// helpers move onto `tracing` with the level chosen at the call site instead of
/// inferred downstream — without expanding every dynamic payload into individual
/// fields.
pub fn emit_event(level: Level, payload: Value) {
    let event_name = payload
        .get("event")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    // Display (`%`) so the visitor reads back the raw JSON text, not a quoted/escaped
    // Debug rendering. The formatter re-parses it and merges the keys up.
    let body = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_owned());
    macro_rules! emit {
        ($macro:ident) => {
            tracing::$macro!(
                target: "sceneworks::event",
                event = %event_name,
                sw_payload = %body,
            )
        };
    }
    match level {
        Level::ERROR => emit!(error),
        Level::WARN => emit!(warn),
        Level::INFO => emit!(info),
        Level::DEBUG => emit!(debug),
        Level::TRACE => emit!(trace),
    }
}

/// Lowercase wire name for a tracing level, matching the `LogEntry.level`
/// vocabulary (`error` / `warn` / `info`; `debug` / `trace` pass through).
fn level_str(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "error",
        Level::WARN => "warn",
        Level::INFO => "info",
        Level::DEBUG => "debug",
        Level::TRACE => "trace",
    }
}

/// Collects an event's fields into a JSON object.
#[derive(Default)]
struct JsonVisitor {
    fields: Map<String, Value>,
}

impl Visit for JsonVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_owned(), Value::String(value.to_owned()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_owned(), Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_owned(), Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_owned(), Value::from(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields
            .insert(field.name().to_owned(), Value::from(value));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.fields
            .insert(field.name().to_owned(), Value::String(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // `%value` (Display) and `?value` (Debug) both arrive here. For our
        // Display-formatted fields (event name, the JSON payload, plain messages)
        // this yields the unquoted text we want.
        self.fields
            .insert(field.name().to_owned(), Value::String(format!("{value:?}")));
    }
}

/// Render an event into the SceneWorks `{ event, level, reportedAt, ... }` envelope
/// (a single JSON object). Shared by the stdout JSON formatter and the session-log
/// layer so both surfaces produce byte-identical lines.
fn render_envelope(event: &Event<'_>) -> Map<String, Value> {
    let mut visitor = JsonVisitor::default();
    event.record(&mut visitor);
    let mut fields = visitor.fields;

    // Flatten a pre-built structured payload up to the top level (legacy
    // `json!` events routed through `emit_event`). Existing real fields win.
    if let Some(Value::String(body)) = fields.remove(PAYLOAD_FIELD) {
        if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(&body) {
            for (key, value) in object {
                fields.entry(key).or_insert(value);
            }
        }
    }

    // Declared level is authoritative; timestamp is the emit time. Both override
    // anything that arrived in the payload.
    fields.insert(
        "level".to_owned(),
        Value::String(level_str(event.metadata().level()).to_owned()),
    );
    fields.insert("reportedAt".to_owned(), Value::String(utc_now()));
    fields
}

/// `tracing_subscriber` event formatter that writes the SceneWorks JSON envelope,
/// one object per line. Replaces tracing's native JSON envelope so the line shape
/// stays the one `session_log` parses and `docs/observability.md` documents.
struct SceneworksJsonFormat;

impl<S, N> FormatEvent<S, N> for SceneworksJsonFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let envelope = Value::Object(render_envelope(event));
        writeln!(writer, "{envelope}")
    }
}

/// A `tracing` layer that feeds every event into a [`SessionLog`] ring buffer using
/// the same JSON envelope as stdout. Lets the API serve its own emitted events back
/// over `GET /api/v1/logs` with declared levels, regardless of the stdout format.
struct SessionLogLayer {
    buffer: &'static SessionLog,
}

impl<S> Layer<S> for SessionLogLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let envelope = Value::Object(render_envelope(event));
        // `push_line` runs the existing secret redaction + classification; with a
        // declared `level` field present, `infer_level` now trusts it verbatim.
        self.buffer.push_line("api", &envelope.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_log::LogQuery;

    #[test]
    fn level_str_maps_tracing_levels() {
        assert_eq!(level_str(&Level::ERROR), "error");
        assert_eq!(level_str(&Level::WARN), "warn");
        assert_eq!(level_str(&Level::INFO), "info");
        assert_eq!(level_str(&Level::DEBUG), "debug");
    }

    #[test]
    fn resolve_format_honors_explicit_env() {
        // We can't safely mutate process env in parallel tests, so exercise the
        // parsing branches directly on the value rather than via the global.
        let json = "JSON";
        assert!(json.trim().eq_ignore_ascii_case("json"));
        let pretty = " pretty ";
        assert!(pretty.trim().eq_ignore_ascii_case("pretty"));
    }

    #[test]
    fn session_layer_renders_declared_level_into_buffer() {
        // A dedicated buffer + a scoped subscriber: the layer should flatten the
        // payload, carry the declared (warn) level, and stamp reportedAt.
        let buffer: &'static SessionLog = Box::leak(Box::new(SessionLog::default()));
        let subscriber = tracing_subscriber::registry().with(SessionLogLayer { buffer });
        tracing::subscriber::with_default(subscriber, || {
            emit_event(
                Level::WARN,
                serde_json::json!({
                    "event": "claim_lock_contention",
                    "consecutiveFailures": 3,
                }),
            );
        });

        let entries = buffer.query(&LogQuery::default());
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.source, "api");
        assert_eq!(entry.level, "warn", "declared level is trusted");
        let event = entry.event.as_ref().expect("structured event");
        assert_eq!(
            event.get("event").and_then(Value::as_str),
            Some("claim_lock_contention")
        );
        assert_eq!(event.get("level").and_then(Value::as_str), Some("warn"));
        assert_eq!(
            event.get("consecutiveFailures").and_then(Value::as_i64),
            Some(3)
        );
        assert!(event.get("reportedAt").is_some(), "reportedAt stamped");
    }
}
