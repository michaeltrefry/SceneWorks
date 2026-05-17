pub mod contracts;
pub mod jobs_store;
pub mod project_store;

pub const API_PREFIX: &str = "/api/v1";
pub const HEALTH_ROUTE: &str = "/health";

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
