# Rust Migration Contract Fixtures

These fixtures capture the Python backend contract before the Rust backend
migration begins. Rust ports should treat this folder as the compatibility
baseline for HTTP route coverage, worker queue protocol strings, SSE event
names, and persisted project sidecar shapes.

The companion test `tests/test_rust_migration_contract_fixtures.py` checks the
fixtures against the current Python implementation. When the Python contract
changes intentionally, update these fixtures in the same change so Rust parity
tests can see the new baseline.

