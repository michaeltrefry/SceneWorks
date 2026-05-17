# Code Review Verification Pass — 2026-05-17

Verifying that the 37 findings from `CODE_REVIEW_2026-05-17.md` have actually been fixed by Codex's three follow-up PRs (commits `d83182e`, `6bbd2c6`/`ccb7723`, `ca4d0c0`/`d8dd7d4`).

## Headline

**33 of 37 fixed and verified. 2 fixed but with caveats. 1 fixed but story still in *In Review* (correctly). 1 fixed code-side but Shortcut state worth double-checking.**

**No story should be reopened.** One minor regression worth a single follow-up commit, plus a flaky test on coarse-mtime filesystems.

Smoke tests:
- **Python tests** (Docker python:3.12-slim): 13 of 14 pass. One flake on the manifest-cache test, details below.
- **Vitest** (Docker node:22-alpine): 2 of 2 pass.

---

## Per-finding verification

| ID | Story | Status | Where it lives now |
|----|-------|--------|--------------------|
| F-001 | sc-1198 | ✅ Fixed | [apps/web/package.json:12-21](apps/web/package.json:12) pins to exact versions; [apps/web/package-lock.json](apps/web/package-lock.json) committed (~2800 lines); [docker/web.Dockerfile:5-6](docker/web.Dockerfile:5) uses `npm ci`. `typescript` removed. |
| F-002 | sc-1199 | ✅ Fixed | [apps/api/sceneworks_api/assets.py:120-194](apps/api/sceneworks_api/assets.py:120) `POST /projects/{id}/assets` (multipart) plus `python-multipart==0.0.20` in [apps/api/requirements.txt:2](apps/api/requirements.txt:2). UI wired in [apps/web/src/App.jsx:473-491](apps/web/src/App.jsx:473) and the new LibraryScreen. |
| F-003 | sc-1200 | ✅ Fixed | Worker handlers: [apps/worker/scene_worker/runtime.py:118-266](apps/worker/scene_worker/runtime.py:118) (`run_model_download_job`, `run_lora_import_job`) using `huggingface_hub.snapshot_download`. API routes: [apps/api/sceneworks_api/models.py:169-222](apps/api/sceneworks_api/models.py:169) (`/models/{id}/download`, `/loras`, `/loras/import`). |
| F-004 | sc-1201 | ✅ Fixed | New [apps/web/src/screens/ModelManagerScreen.jsx](apps/web/src/screens/ModelManagerScreen.jsx) with install state, family filter, and "Download" button. `Models` added to [apps/web/src/constants.js](apps/web/src/constants.js) `navItems`. |
| F-005 | sc-1202 | ✅ Fixed | `NON_GPU_JOB_TYPES = ("model_download","lora_import")` in [apps/api/sceneworks_api/jobs_store.py:16](apps/api/sceneworks_api/jobs_store.py:16); claim query excludes them from GPU exclusion at [jobs_store.py:415-435](apps/api/sceneworks_api/jobs_store.py:415); covered by `test_non_gpu_jobs_can_claim_while_gpu_is_busy` (passing). |
| F-006 | sc-1203 | ✅ Fixed | `mark_stale_workers_interrupted` in [jobs_store.py:347-402](apps/api/sceneworks_api/jobs_store.py:347), invoked from `sweep_stale_workers` in [jobs.py:78-86](apps/api/sceneworks_api/jobs.py:78) on `/jobs`, `/jobs/claim`, `/queue`, `/workers`. Timeout via `SCENEWORKS_WORKER_TIMEOUT_SECONDS` (default 90) added to [settings.py:14](apps/api/sceneworks_api/settings.py:14). |
| F-007 | sc-1204 | ✅ Fixed | Exponential-backoff reconnect with `reconnectAttempt` and `setTimeout` chain at [apps/web/src/App.jsx:164-204](apps/web/src/App.jsx:164). Cleanup via `closed` flag. |
| F-008 | sc-1205 | ✅ Fixed | `strip_jsonc_comments` rewritten as a stateful tokenizer respecting string literals at [models.py:30-69](apps/api/sceneworks_api/models.py:30); covered by `test_jsonc_comment_stripping_preserves_url_strings` (passing). |
| F-009 | sc-1206 | ✅ Fixed | `REGISTRY_LOCK` guards `create_project`; `save_registry` writes to a `NamedTemporaryFile` then `Path.replace()` at [projects.py:75-86](apps/api/sceneworks_api/projects.py:75). |
| F-010 | sc-1207 | ✅ Fixed | 5 test files added under [tests/](tests/). 13 of 14 Python tests pass; 2 of 2 Vitest tests pass. See caveat F-024 below. |
| F-011 | sc-1208 | ✅ Fixed | Explicit method/header lists at [main.py:34-40](apps/api/sceneworks_api/main.py:34); `allow_credentials=False` since the API uses a custom header for auth, not cookies. |
| F-012 | sc-1209 | ✅ Fixed | `secrets.compare_digest` on bytes at [security.py:31-37](apps/api/sceneworks_api/security.py:31). |
| F-013 | sc-1210 | ✅ Fixed | `EventTicketStore` in [events.py:46-68](apps/api/sceneworks_api/events.py:46) (30s TTL, one-shot). `POST /jobs/events/ticket` and ticket consumption in [jobs.py:115-123](apps/api/sceneworks_api/jobs.py:115). The token query-param parser was also dropped from [security.py:18-28](apps/api/sceneworks_api/security.py:18). |
| F-014 | sc-1211 | ✅ Fixed | `DELETE` now moves media into `trash/<asset_id>/` and updates the sidecar with `trashed=true` at [assets.py:215-235](apps/api/sceneworks_api/assets.py:215). New `DELETE /assets/{id}/purge` permanently removes. Library UI wires both ("Discard" → soft, "Purge" → hard). Covered by `test_delete_soft_trashes_asset_and_purge_removes_it`. |
| F-015 | sc-1212 | ✅ Fixed | `claim_next_job` filters by `worker["capabilities"]` at [jobs_store.py:432](apps/api/sceneworks_api/jobs_store.py:432); CPU workers no longer advertise GPU types in [runtime.py:45-50](apps/worker/scene_worker/runtime.py:45). Test `test_claim_skips_jobs_not_supported_by_worker_capabilities` covers it. |
| F-016 | sc-1213 | ✅ Fixed | `ZImageDiffusersAdapter.loaded_models()` at [image_adapters.py:210-211](apps/worker/scene_worker/image_adapters.py:210); aggregation helper at [runtime.py:53-59](apps/worker/scene_worker/runtime.py:53); heartbeats pass it through at [runtime.py:329, 385, 558](apps/worker/scene_worker/runtime.py:329). Test `test_loaded_models_are_collected_from_adapter_cache` passes. |
| F-017 | sc-1214 | ✅ Fixed | `POST /projects/{id}/reindex` at [projects.py:178-182](apps/api/sceneworks_api/projects.py:178); CLI subcommand at [__main__.py:11-22](apps/api/sceneworks_api/__main__.py:11). Backed by `reindex_project` in shared package; test `test_reindex_project_rebuilds_asset_generation_set_and_timeline_tables` passes. |
| F-018 | sc-1215 | ✅ Fixed | `list_assets` reads from `project.db.assets` at [assets.py:79-117](apps/api/sceneworks_api/assets.py:79); per-asset lookup goes through `find_asset_sidecar_path` (DB-first, glob as healing fallback) in [packages/shared/sceneworks_shared/project_db.py:145-167](packages/shared/sceneworks_shared/project_db.py:145). Worker `find_asset_media_path`, video `load_source_image`, and `timeline_exporter.find_asset` all use the shared helper. |
| F-019 | sc-1216 | ✅ Fixed | NumPy-vectorized at [image_adapters.py:457-487](apps/worker/scene_worker/image_adapters.py:457) and [video_adapters.py:333-345](apps/worker/scene_worker/video_adapters.py:333); `numpy>=2.0,<3` added to worker requirements. |
| F-020 | sc-1217 | ✅ Fixed | New [packages/shared/sceneworks_shared/](packages/shared/sceneworks_shared/) package re-exports `slugify`, `utc_now`, `read_json`, `write_json`, `safe_int`, `safe_float`, `find_project_path` (+ `ProjectNotFound`), `load_registry`. Both API and worker import from it. Dockerfiles wire it via `PYTHONPATH=...:/app/packages/shared`. |
| F-021 | sc-1218 | ✅ Fixed (still **In Review**) | Old `main.jsx` shrunk to 12 lines (just `createRoot`); `App.jsx`, `api.js`, `constants.js`, `formatting.js`, `sorters.js`, `timeline.js`, `screens/*` (7 screens), `components/*` (3 components) all extracted. Story is correctly still `In Review` pending your sign-off. |
| F-022 | sc-1219 | ✅ Fixed | Centralized in `apply_project_migrations` at [packages/shared/sceneworks_shared/project_db.py:15-69](packages/shared/sceneworks_shared/project_db.py:15) with an `_ensure_column` helper for additive migrations. `ensure_timeline_db` (timelines.py) and the worker's `index_project_db` now delegate. |
| F-023 | sc-1220 | ✅ Fixed | Web app subscribes to `queue.updated` at [App.jsx:189](apps/web/src/App.jsx:189) and stores `queueSummary` for the topbar chip. |
| F-024 | sc-1221 | ⚠️ **Fixed with caveat** | mtime-keyed `lru_cache(maxsize=16)` on `load_manifest_cached(path, mtime_ns, key)` at [models.py:79-94](apps/api/sceneworks_api/models.py:79). **But the test Codex wrote for it (`test_manifest_cache_reloads_when_mtime_changes`) fails on filesystems whose mtime resolution is coarser than the time between the two `write_text` calls** (overlay FS on Linux often gives 1-second resolution; reproduced in `python:3.12-slim` container — `st_mtime_ns` was *identical* for two back-to-back writes). The production code is fine for the realistic case (manifests aren't edited per nanosecond), but the test is flaky and should either size-bound the cache combined with a write-time bump, or insert a `time.sleep(0.05)` between writes in the test. |
| F-025 | sc-1222 | ✅ Fixed | `_evict_pipelines` + `_empty_cuda_cache` at [image_adapters.py:303-311](apps/worker/scene_worker/image_adapters.py:303); called on repo change and on mode switch. |
| F-026 | sc-1223 | ✅ Fixed | Capped at 20 attempts with exponential backoff (max 30s) at [runtime.py:534-554](apps/worker/scene_worker/runtime.py:534). Raises `RuntimeError` so the orchestrator can restart with corrected config. |
| F-027 | sc-1224 | ✅ Fixed | `MAX_JOB_ATTEMPTS = 5` constant; `retry_job` raises `ValueError` once exceeded ([jobs_store.py:17, 257-258](apps/api/sceneworks_api/jobs_store.py:17)); `/queue` exposes `maxJobAttempts` ([jobs.py:219](apps/api/sceneworks_api/jobs.py:219)). Test `test_retry_job_is_capped` passes. |
| F-028 | sc-1225 | ✅ Fixed | `Image.MAX_IMAGE_PIXELS = 64_000_000` and `warnings.simplefilter("error", Image.DecompressionBombWarning)` at module top of both [image_adapters.py:31-32](apps/worker/scene_worker/image_adapters.py:31) and [video_adapters.py:20-21](apps/worker/scene_worker/video_adapters.py:20); `Image.open` calls wrapped to catch `DecompressionBombError`/`Warning`. |
| F-029 | sc-1226 | ✅ Fixed | `update_asset_status` now calls `index_asset_db(project_path, asset)` at [assets.py:211](apps/api/sceneworks_api/assets.py:211); test `test_status_patch_updates_project_db` confirms the DB row updates. |
| F-030 | sc-1227 | ✅ Fixed | [.env.example](.env.example) at repo root with all `SCENEWORKS_*` keys commented. |
| F-031 | sc-1228 | ✅ Fixed | `run_ffmpeg` keeps the last 10 lines (capped at 2 kB) of stderr in the error at [timeline_exporter.py:307-315](apps/worker/scene_worker/timeline_exporter.py:307). |
| F-032 | sc-1229 | ✅ Fixed | `"workspaces"` block removed from root [package.json](package.json); `typescript` dropped from [apps/web/package.json](apps/web/package.json). |
| F-033 | sc-1230 | ✅ Fixed | `find_timeline_file` now writes the discovered path back to the DB and raises a clear "reindex required" 404 when the indexed path is stale and no candidate found at [timelines.py:158-182](apps/api/sceneworks_api/timelines.py:158). Test `test_find_timeline_file_heals_stale_db_path` confirms. |
| F-034 | sc-1231 | ✅ Fixed | `JobType = Literal[...]` at [jobs.py:16-30](apps/api/sceneworks_api/jobs.py:16). |
| F-035 | sc-1232 | ✅ Done (in spirit) | Epic 1087 moved to **In Progress** (not Done) with comment 1238 noting what's already landed and what still needs Docker-environment validation before close. That's the conservative call; the story can stay closed. |
| F-036 | sc-1233 | ✅ Fixed | Code comment present at [image_adapters.py:52](apps/worker/scene_worker/image_adapters.py:52). |
| F-037 | sc-1234 | ✅ Fixed | Both `find_project_path` versions now delegate to `shared_find_project_path` which raises `ProjectNotFound`; API translates to 404 ([projects.py:127-131](apps/api/sceneworks_api/projects.py:127)) and worker translates to `RuntimeError` ([image_adapters.py:92-96](apps/worker/scene_worker/image_adapters.py:92)). Folded into F-020 as planned. |

---

## Issues discovered during verification

These are **not** justification to reopen any story — they're follow-ups worth tracking, ideally as new small stories under the same epic.

### V-1 — Manifest cache test is flaky on coarse-mtime filesystems
**Where:** [tests/test_models.py:14-19](tests/test_models.py:14)

The cache invalidation relies on `Path.stat().st_mtime_ns` changing between two consecutive `write_text` calls. On overlay FS (and HFS+ in some configurations), this resolution is coarser than the actual call interval and both reads return the same `mtime_ns`, leaving the lru_cache pointing at the first manifest's content. Reproduced in the Docker `python:3.12-slim` container — identical `st_mtime_ns` for two back-to-back writes.

**Fix options:**
- Test-only: add `time.sleep(0.01)` between writes (cheap, hides nothing meaningful).
- Implementation: also key the cache on `path.stat().st_size` (changes with content, not timing) — defends against rapid editor saves that don't bump mtime.
- Or both.

### V-2 — Dead `return None` after `return canvas`
**Where:** [apps/worker/scene_worker/video_adapters.py:300-301](apps/worker/scene_worker/video_adapters.py:300)

Two `return` statements; the second is unreachable. Cosmetic.

### V-3 — `index_asset` and `index_asset_db` are two thin wrappers around the same call
**Where:** [apps/api/sceneworks_api/assets.py:58-60](apps/api/sceneworks_api/assets.py:58), [apps/worker/scene_worker/image_adapters.py:553-555](apps/worker/scene_worker/image_adapters.py:553)

After the F-020 refactor, both `assets.index_asset_db` and worker `image_adapters.index_project_db` are 1-line wrappers that just forward to the shared `index_asset`. Each adds a slightly different default sidecar-path derivation. Could be deleted in favor of direct calls to `index_asset(...)`.

### V-4 — `assets.list_assets` reads from DB but assets created before F-018 won't appear
**Where:** [apps/api/sceneworks_api/assets.py:79-117](apps/api/sceneworks_api/assets.py:79)

The query is now DB-driven. Any project created before the shared `apply_project_migrations` added the `sidecar_path` column will show empty lists until reindexed. The `POST /projects/{id}/reindex` endpoint exists to repair this, but there's no auto-trigger or migration banner in the UI. Worth a one-line "tip: run reindex" hint or an automatic on-first-load reindex when row count is 0 but folder contains sidecars.

---

## Recommendations

1. **Codex's fixes are solid.** All 37 stories' code changes are present and matched to the original suggested fixes. Three large PRs with high signal-per-diff.
2. **Resolve F-021 by reviewing the split.** The story is correctly in *In Review* — it's the only one Codex didn't auto-Done, and a human-eye check is what's appropriate for a 2,100→700-line refactor. The structure looks clean: `App.jsx` owns state, screens consume it via props.
3. **Address V-1 before the next test run** — a flaky test on a fresh checkout will erode trust in the suite.
4. **Optional one-shot cleanup commit** for V-2/V-3 — both are small and obvious.

You can close Epic 1197 once F-021 lands.
