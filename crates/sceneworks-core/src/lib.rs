pub mod angle_kps;
pub mod app_paths;
pub mod asset_index;
pub mod builtin_manifests;
pub mod character_store;
pub mod contracts;
pub mod credentials;
pub mod dataset_quality;
pub mod hf_home;
pub mod ideogram_caption;
pub mod image_request;
pub mod jobs_store;
pub mod jsonc;
pub mod lora_family;
pub mod lora_url;
pub mod media_convert;
pub mod observability;
pub mod payload_util;
pub mod project_store;
pub mod session_log;
pub mod slug;
pub mod store_util;
pub mod time;
pub mod training;
pub mod training_store;
pub mod video_request;

pub const API_PREFIX: &str = "/api/v1";
pub const HEALTH_ROUTE: &str = "/health";

/// Stdout sentinel for remote worker-restart (epic 4484 story 12). The API process
/// doesn't supervise the desktop's GPU worker, so `POST /api/v1/worker/restart` prints
/// this exact line to stdout; the desktop shell — which already reads the API sidecar's
/// stdout — matches it and performs the same kill-and-respawn as its local "Restart
/// worker" button. Shared here so the emitter (rust-api) and matcher (desktop) can
/// never drift. Deliberately unique/unlikely to appear in ordinary log output.
pub const WORKER_RESTART_SENTINEL: &str = "__SCENEWORKS_WORKER_RESTART_REQUESTED__";

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HealthContract {
    route: &'static str,
    service: &'static str,
}

impl HealthContract {
    pub const fn new(route: &'static str, service: &'static str) -> Self {
        Self { route, service }
    }

    pub const fn route(&self) -> &'static str {
        self.route
    }

    pub const fn service(&self) -> &'static str {
        self.service
    }

    pub fn absolute_path(&self) -> String {
        format!("{API_PREFIX}{}", self.route)
    }
}

impl Default for HealthContract {
    fn default() -> Self {
        Self::new(HEALTH_ROUTE, "sceneworks-api")
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthContract, API_PREFIX, HEALTH_ROUTE};

    #[test]
    fn health_contract_matches_python_route_prefix() {
        let contract = HealthContract::default();

        assert_eq!(API_PREFIX, "/api/v1");
        assert_eq!(contract.route(), HEALTH_ROUTE);
        assert_eq!(contract.absolute_path(), "/api/v1/health");
        assert_eq!(contract.service(), "sceneworks-api");
    }
}
