use super::*;

// Short-lived query-param tickets for endpoints the browser cannot reach with an
// auth header (epic 4484). Two flavors share this store:
//  - SSE (`/api/v1/jobs/events`): `EventSource` can't set headers → single-use
//    `issue()` + `consume()` tickets (sc-4484 story: events ticket).
//  - Media (`GET /api/v1/projects/:id/files/*`, `GET /api/v1/poses/preview/*`):
//    `<img src>`/`<video src>`/`<a download>` requests can't set headers either →
//    reusable `issue_sliding()` + `validate()` tickets (sc-8810). A page renders
//    dozens of media URLs (and <video> issues multiple Range requests), so these
//    tickets are multi-use within their TTL; the web client refreshes well before
//    expiry and `issue_sliding` keeps returning (and re-arming) the same ticket so
//    already-rendered <img>/<video> URLs stay valid without a re-render.
//
// Separate `TicketStore` instances give scope isolation for free: an event ticket
// is never accepted on the files route and vice versa.
#[derive(Debug)]
pub(crate) struct TicketStore {
    ttl: Duration,
    state: Mutex<TicketStoreState>,
}

#[derive(Debug, Default)]
struct TicketStoreState {
    tickets: HashMap<String, Instant>,
    // Most recently issued ticket, so `issue_sliding` can hand the same value to
    // every caller while it stays alive (URL stability across React re-renders).
    latest: Option<String>,
}

impl TicketStore {
    pub(crate) fn new(ttl_seconds: u64) -> Self {
        Self {
            ttl: Duration::from_secs(ttl_seconds),
            state: Mutex::new(TicketStoreState::default()),
        }
    }

    /// Issue a fresh single-use ticket (SSE flavor; pair with `consume`).
    pub(crate) fn issue(&self) -> TicketResponse {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        let ticket = Uuid::new_v4().simple().to_string();
        state.tickets.insert(ticket.clone(), now + self.ttl);
        state.latest = Some(ticket.clone());
        TicketResponse {
            ticket,
            expires_in_seconds: self.ttl.as_secs(),
        }
    }

    /// Issue a reusable ticket (media flavor; pair with `validate`). While the most
    /// recent ticket is still valid this returns it again and re-arms its expiry to
    /// a full TTL, so a client refreshing on an interval keeps one stable ticket
    /// alive for its whole session. A leaked URL therefore dies at most one TTL
    /// after the last authenticated refresh.
    pub(crate) fn issue_sliding(&self) -> TicketResponse {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        if let Some(ticket) = state.latest.clone() {
            if state.tickets.contains_key(&ticket) {
                state.tickets.insert(ticket.clone(), now + self.ttl);
                return TicketResponse {
                    ticket,
                    expires_in_seconds: self.ttl.as_secs(),
                };
            }
        }
        drop(state);
        self.issue()
    }

    /// Single-use check: the ticket is removed whether or not it is still valid.
    pub(crate) fn consume(&self, ticket: &str) -> bool {
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        matches!(state.tickets.remove(ticket), Some(expires_at) if expires_at >= now)
    }

    /// Non-consuming check for multi-use (media) tickets.
    pub(crate) fn validate(&self, ticket: &str) -> bool {
        if ticket.is_empty() {
            return false;
        }
        let now = Instant::now();
        let mut state = self.state.lock();
        prune_tickets(&mut state.tickets, now);
        matches!(state.tickets.get(ticket), Some(expires_at) if *expires_at >= now)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TicketResponse {
    pub(crate) ticket: String,
    pub(crate) expires_in_seconds: u64,
}

/// POST /api/v1/files/ticket — auth-protected (header token or loopback trust via
/// the access-control middleware), so only an already-authenticated client can
/// mint a media ticket.
pub(crate) async fn create_media_ticket(State(state): State<AppState>) -> Json<TicketResponse> {
    Json(state.media_tickets.issue_sliding())
}

fn prune_tickets(tickets: &mut HashMap<String, Instant>, now: Instant) {
    tickets.retain(|_, expires_at| *expires_at >= now);
}
