use super::*;

use std::sync::OnceLock;

use sceneworks_core::session_log::{LogEntry, LogQuery, SessionLog};

/// Process-global API-side session log (sc-3453). On the desktop the richer
/// multi-source buffer lives in the Tauri wrapper (sc-3451), fed by every
/// sidecar's stdout; this buffer covers headless/web/Docker runtimes that have no
/// wrapper by retaining the structured events the API process itself emits (MLX
/// routing decisions, API errors, auth rejections, etc.), served by
/// `GET /api/v1/logs`. It is fed by the tracing `SessionLogLayer`
/// ([`sceneworks_core::observability::init_logging_with_buffer`]), so entries carry
/// the **declared** level. Same `LogEntry` shape as the desktop buffer so the in-app
/// Logs screen is source-agnostic.
static API_SESSION_LOG: OnceLock<SessionLog> = OnceLock::new();

pub(crate) fn api_session_log() -> &'static SessionLog {
    API_SESSION_LOG.get_or_init(SessionLog::default)
}

/// `GET /api/v1/logs` — the current process's session events, filtered by the
/// `LogQuery` params (`afterSeq`, `limit`, `source`, `level`, `search`).
pub(crate) async fn list_logs(Query(query): Query<LogQuery>) -> Json<Vec<LogEntry>> {
    Json(api_session_log().query(&query))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_queries_api_event() {
        // Unique marker so this is robust against other tests sharing the global buffer.
        // In production the tracing `SessionLogLayer` feeds this same buffer; here we
        // push directly to assert the query/shape contract.
        let marker = "route-decision-test-marker-9f3a";
        api_session_log().push_line(
            "api",
            &json!({
                "event": "mlx_route_decision",
                "decision": "fell_back_to_torch",
                "reason": "no_idle_mlx_worker",
                "model": marker,
                "level": "info",
                "reportedAt": "2026-06-07T00:00:00Z"
            })
            .to_string(),
        );
        let hits = api_session_log().query(&LogQuery {
            search: Some(marker.to_owned()),
            ..Default::default()
        });
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "api");
        assert!(hits[0].message.contains("decision=fell_back_to_torch"));
        assert!(hits[0].event.is_some());
    }
}
