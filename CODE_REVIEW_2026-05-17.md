# Full Codebase Review — SceneWorks — 2026-05-17

## Executive summary

- Repository at a glance: Python 3.12 (FastAPI API + worker), React 18 / Vite frontend (single JSX file), SQLite for job/asset metadata, JSONC manifests, FFmpeg for timeline export, optional Hugging Face `diffusers` for real image generation. ~30 source files, ~7,000 LOC across `apps/`, `packages/schemas/`, `scripts/`, `docker/`.
- Coverage: every source file under `apps/`, `packages/`, `scripts/`, `docker/`, `config/manifests/` was read fully. JSON schemas in `packages/schemas/` were spot-checked. CSS (`apps/web/src/styles.css`, 1.1k lines) was not reviewed in depth; design-only. `data/` is content storage and was excluded.
- Headline: the runtime skeleton and the vertical slices for Phases 0–7 are in place and the code is mostly cohesive, but five completed epics carry meaningful gaps against their own acceptance criteria — there is no asset *import* endpoint or UI (Epic 1082), no `model_download` / `lora_import` job handlers or Model Manager screen (Epic 1084), no worker-liveness detection (Epic 1083), and no automated tests anywhere in the repo. There is also one outright security defect (pinned `"latest"` deps in the web app), several real bugs that will misbehave under simple inputs (JSONC comment stripper, EventSource has no reconnect), and a large pile of duplicated utilities across the worker and API that should be lifted into `packages/shared` before more vertical slices land.
- Counts: Critical: 1 | High: 9 | Medium: 14 | Low: 10 | Info: 3.

---

## Critical findings

#### [F-001] Pin web app dependencies — `"latest"` everywhere
- **Category:** bad-pattern
- **Severity:** Critical
- **Location:** [apps/web/package.json](apps/web/package.json:12-19)
- **Finding:** Every dependency in `apps/web/package.json` is `"latest"`, including `react`, `react-dom`, `vite`, `@vitejs/plugin-react`, and `typescript`. There is no `package-lock.json`, no `.npmrc` pin, and Docker installs fresh on every build.
- **Impact:** Any upstream breaking release (React 19/20, Vite major, plugin-react peer mismatch) silently breaks the build the next time `docker compose up --build` runs. Two contributors on different days can produce different bundles from the same commit. Reviews of the running app no longer correspond to the repo state.
- **Suggested fix:** Pin to known-good versions (e.g. `"react": "^18.3.1"`, `"vite": "^5.4.x"`, etc.), commit `package-lock.json`, and switch the Dockerfile to `npm ci`. Remove `typescript` from `dependencies` if the app stays JS-only; the file is `.jsx` and no `tsconfig.json` exists.
- **Confidence:** High

---

## High findings

#### [F-002] Asset import endpoint and UI are missing — Epic 1082 acceptance gap
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/api/sceneworks_api/assets.py](apps/api/sceneworks_api/assets.py:14), [apps/web/src/main.jsx](apps/web/src/main.jsx:763) (LibraryScreen)
- **Finding:** Epic 1082 acceptance says "User imports an image and video. Imported files are copied into the project." The `assets/uploads` folder is created on project init, the `asset` schema has an `upload` type, but there is no `POST /projects/{id}/assets` (or equivalent upload) endpoint and no upload UI in the Library. `python-multipart` is not in `apps/api/requirements.txt`.
- **Impact:** The only way assets enter a project today is via the worker writing generation outputs. Users cannot bring their own footage in, which directly contradicts the "Imported files are copied into the project" criterion and the Phase-1 deliverable list.
- **Suggested fix:** Add `POST /projects/{project_id}/assets` with multipart upload, write the sidecar via the existing `build_asset_sidecar`-style helper, copy the file under `assets/uploads`, and call `index_project_db`. Add a small upload button in the Library screen. Add `python-multipart` to requirements.
- **Confidence:** High

#### [F-003] `model_download` / `lora_import` job types unimplemented — Epic 1084 acceptance gap
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/worker/scene_worker/runtime.py:356-375](apps/worker/scene_worker/runtime.py:356), [apps/api/sceneworks_api/models.py](apps/api/sceneworks_api/models.py)
- **Finding:** Epic 1084 closes claiming "non-GPU model_download/lora_import job foundations" exist. The worker dispatch in `runtime.py` only handles `placeholder`, `image_generate`, `image_edit`, `video_generate`, `video_extend`, `video_bridge`, and `timeline_export`; anything else (including `model_download` and `lora_import`) immediately fails with "No adapter exists for this job type yet." There is also no API route that creates such jobs and no `/api/v1/loras` endpoint.
- **Impact:** The two acceptance criteria "Missing model can create a download job" and "LoRA compatibility can be filtered by model family" are not satisfied. Manifests list HF repos, but nothing in the system can actually pre-fetch or import them.
- **Suggested fix:** Add a `model_download` job route that creates a queued job with `requested_gpu="cpu"` (or a new sentinel), add a worker handler that calls `huggingface_hub.snapshot_download`, and skip the GPU-exclusion check in `claim_next_job` for these job types (see F-005). Add a `/api/v1/loras` endpoint mirroring `models.py`.
- **Confidence:** High

#### [F-004] Model Manager UI does not exist — Epic 1084 acceptance gap
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/web/src/main.jsx:7](apps/web/src/main.jsx:7) (`navItems`)
- **Finding:** Epic 1084 lists a Model Manager screen with installed/missing/downloadable states, predownload button, and a LoRA list. The web app's `navItems` is `["Library", "Image", "Video", "Characters", "Editor", "Queue"]` — there is no Models view, and `models` state in `App()` is consumed only by `<select>` dropdowns inside Image/Video studios. The "Characters" route renders `PlaceholderSurface`.
- **Impact:** Users cannot see what models are installed, what would be downloaded, or trigger downloads. The acceptance criteria for Phase 5 cannot be demonstrated.
- **Suggested fix:** Add a "Models" entry to `navItems` and a `ModelManagerScreen` component that lists `/api/v1/models` + (planned) `/api/v1/loras`, shows install state (file or HF cache check via API), and offers a "Download" button that posts a `model_download` job from F-003.
- **Confidence:** High

#### [F-005] GPU exclusion blocks non-GPU jobs — Epic 1084 ("Download jobs do not consume GPU slots")
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/api/sceneworks_api/jobs_store.py:346-394](apps/api/sceneworks_api/jobs_store.py:346) (`claim_next_job`)
- **Finding:** `claim_next_job` always checks whether the worker's `gpu_id` already has any active job, regardless of the queued job's type. There is no concept of "this job does not need a GPU slot," so a future `model_download` or `lora_import` job will be serialized behind any in-flight image/video generation on the same GPU.
- **Impact:** Even once F-003 lands, the Phase-5 acceptance "Download jobs do not consume GPU slots" cannot be met. A download starting during a long render would either block or, worse, hold the GPU lock for a non-GPU task.
- **Suggested fix:** Add a non-GPU job-type set (e.g. `NON_GPU_JOB_TYPES = {"model_download", "lora_import"}`) and skip the `active_gpu_job` check when the queued job's type is in that set. Track these as a separate concurrency lane (e.g. cap N concurrent CPU jobs per worker).
- **Confidence:** High

#### [F-006] Dead worker leaves jobs orphaned indefinitely
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/api/sceneworks_api/jobs_store.py:117-141](apps/api/sceneworks_api/jobs_store.py:117) (`mark_interrupted_on_startup`), [apps/api/sceneworks_api/jobs_store.py:318-344](apps/api/sceneworks_api/jobs_store.py:318) (`heartbeat_worker`)
- **Finding:** The API only marks jobs interrupted at API startup. There is no periodic sweep that detects workers that haven't sent a heartbeat (`last_seen_at` is recorded but never compared). If a worker process crashes or is killed mid-job, that job stays in `preparing`/`running`/etc. forever, the worker row stays `busy`, and no other worker will claim a queued job for that GPU (because of F-005's exclusion logic).
- **Impact:** "Browser-refresh persistence" (Epic 1083) holds for browser refreshes but not for the more common case of a worker crash. The queue silently wedges.
- **Suggested fix:** Add a background task in the API (or run on every `claim_next_job` call) that finds workers with `last_seen_at < now - 2 * heartbeat_seconds` and marks them offline + transitions their active jobs to `interrupted`. Document the timeout via a new `SCENEWORKS_WORKER_TIMEOUT_SECONDS` env var (default ~90s).
- **Confidence:** High

#### [F-007] EventSource has no reconnect — UI silently goes stale on any network blip
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/web/src/main.jsx:273-300](apps/web/src/main.jsx:273)
- **Finding:** The SSE handler is `events.onerror = () => events.close();`. The browser's automatic reconnect is intentionally suppressed, and there is no retry logic. The cleanup function also unconditionally closes on unmount.
- **Impact:** Any transient API restart, network glitch, or proxy timeout permanently severs live updates. Users won't see job progress or worker state until they refresh the page, which directly undermines Phase 3's "Progress updates live" acceptance.
- **Suggested fix:** On `onerror`, schedule a `setTimeout` reconnect with exponential backoff (e.g. 1s → 5s → 30s, capped). Or simply remove the `events.close()` call inside `onerror` and let the browser auto-reconnect, but still re-subscribe to events explicitly if needed.
- **Confidence:** High

#### [F-008] JSONC comment stripper corrupts URLs containing `//`
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/api/sceneworks_api/models.py:14-17](apps/api/sceneworks_api/models.py:14)
- **Finding:** `strip_jsonc_comments` is `re.sub(r"//.*", "", value)`. This naively matches `//` anywhere on the line, including inside string literals. Any future user manifest entry containing `"https://..."` (e.g. a download URL, license URL, doc URL) will be silently truncated to `"https:` before `json.loads`, producing either an invalid JSON error or — worse — a successfully parsed but corrupted value.
- **Impact:** Adding any URL-bearing string to `config/manifests/user.models.jsonc` will break model loading. The same applies to LoRA manifests when they're loaded similarly.
- **Suggested fix:** Use a real JSONC parser (`json5`, or `pyjson5`) or, at minimum, strip line comments only when not inside a string by switching to a tokenizer-based approach. Add a unit test with `"url": "https://example.com"` in the manifest fixture.
- **Confidence:** High

#### [F-009] Recent-projects registry has read-modify-write races on simultaneous creates
- **Category:** bad-pattern
- **Severity:** High
- **Location:** [apps/api/sceneworks_api/projects.py:62-74,211-213](apps/api/sceneworks_api/projects.py:62)
- **Finding:** `create_project` does `registry = load_registry(...)` → mutate → `save_registry(...)` without any cross-request lock or atomic write. Two concurrent `POST /projects` calls will both read the old list and the second write will clobber the first.
- **Impact:** Lost project entries in `recent-projects.json`; the new project's folder still exists on disk but is not discoverable through `GET /projects`. Also, `save_registry` writes directly (no temp-file + rename), so an interrupted process can leave a truncated JSON file that breaks subsequent reads.
- **Suggested fix:** Wrap registry mutation in a process-level lock (`threading.Lock`) and write via `tempfile.NamedTemporaryFile` + `Path.replace()` for atomicity. Even better, move the recent-projects list into the same SQLite store as jobs.
- **Confidence:** High

#### [F-010] No automated tests in the entire repository
- **Category:** bad-pattern
- **Severity:** High
- **Location:** repo root (no `tests/`, no `*_test.py`, no `*.spec.*`)
- **Finding:** `find` for any test files returns nothing. Phase 2 acceptance ("Sidecars validate against schema", "API responses use typed models") and every other epic's behavioral acceptance is verified only by manual smoke tests recorded in Shortcut comments.
- **Impact:** Regressions in any subsystem (queue state machine, asset sidecar shape, manifest parsing, FFmpeg pipeline) will land silently. The codebase has reached a size (~7k LOC, multiple subsystems) where this is now actively dangerous — the next feature slice may break a quiet path you don't notice until a user does.
- **Suggested fix:** Add `pytest` to API and worker requirements; start with three targeted suites: (1) `jobs_store` state-machine tests (create → claim → progress → terminal), (2) `models.py` JSONC parsing including URL strings (catches F-008), (3) `assets.py` path-traversal protection (`get_project_file` already has a guard, lock it down with a test). Add a Vite/Vitest smoke test that renders `<App/>` against a mocked fetch.
- **Confidence:** High

---

## Medium findings

#### [F-011] CORS uses wildcard methods/headers with `allow_credentials=True`
- **Category:** security
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/main.py:33-39](apps/api/sceneworks_api/main.py:33)
- **Finding:** `CORSMiddleware` is configured with `allow_origins=settings.cors_origins` (explicit), `allow_credentials=True`, `allow_methods=["*"]`, and `allow_headers=["*"]`. The CORS spec forbids wildcard methods/headers when credentials are sent.
- **Impact:** Starlette's CORSMiddleware happens to echo back the request's method/headers when wildcards are configured with credentials, so this works today — but the configuration violates spec and any reverse-proxy or stricter implementation will block requests. Also broadens the attack surface unnecessarily.
- **Suggested fix:** Replace wildcards with explicit lists: `allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"]`, `allow_headers=["Content-Type","X-SceneWorks-Token","Authorization"]`. Keep `allow_credentials=True` only if it's actually needed (it isn't — the API uses a custom header for auth, not cookies; consider setting it to `False`).
- **Confidence:** High

#### [F-012] Access token comparison is not constant-time
- **Category:** security
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/security.py:33-37](apps/api/sceneworks_api/security.py:33)
- **Finding:** `is_authorized` does `token_from_request(request) == settings.access_token`. Python `==` on strings short-circuits on first mismatch, leaking information about the token via response-time timing.
- **Impact:** Realistic exploitability is low on a LAN, but the token gates project access, asset deletion, and job creation — it's worth treating as a real credential. The fix is one line.
- **Suggested fix:** `from secrets import compare_digest; return compare_digest(token_from_request(request), settings.access_token)`. Compare bytes to avoid the Unicode-normalization quirk.
- **Confidence:** High

#### [F-013] Access token accepted via URL query parameter
- **Category:** security
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/security.py:17-19](apps/api/sceneworks_api/security.py:17), [apps/web/src/main.jsx:78-84](apps/web/src/main.jsx:78)
- **Finding:** The API accepts `?token=...` as authentication, and the web app uses it for the SSE EventSource URL (because EventSource cannot send headers). Query-string tokens are written to web-server access logs, browser history, and any HTTP referrer.
- **Impact:** The token leaks into log files an operator may not realize are token-bearing. On a multi-user dev machine, anyone with shell access can grep the logs.
- **Suggested fix:** Document the limitation in the README's auth section. For SSE, prefer a short-lived "stream ticket" issued by `POST /api/v1/jobs/events/ticket` (returns a one-shot UUID) and validate the ticket in the SSE handler. Strip `?token=` from the access log format.
- **Confidence:** High

#### [F-014] Trash folder created but never used; `deleteAsset` hard-deletes
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/projects.py:16-29](apps/api/sceneworks_api/projects.py:16), [apps/api/sceneworks_api/assets.py:102-112](apps/api/sceneworks_api/assets.py:102), [apps/web/src/main.jsx:534-543](apps/web/src/main.jsx:534)
- **Finding:** Project init creates a `trash/` folder and the asset schema has a `trashed` status flag, but `DELETE /projects/{id}/assets/{id}` unlinks the file outright. The "Discard" button in the UI calls this destructive endpoint. The `trashed` flag is only set via the status PATCH and is never used to move anything into `trash/`.
- **Impact:** Epic 1082 lists "Trash support" as a Phase-1 deliverable. Today there is no undo for a discard — a misclick on the review grid permanently destroys generated media.
- **Suggested fix:** Either (a) make `DELETE` move the media + sidecar into `trash/` and set `trashed=true` (so the row is recoverable) and add a separate "purge" endpoint for permanent deletion, or (b) rip out the unused trash folder and `trashed` flag entirely and document the change.
- **Confidence:** High

#### [F-015] Worker capabilities advertised but never used by the dispatcher
- **Category:** dead-code
- **Severity:** Medium
- **Location:** [apps/worker/scene_worker/runtime.py:43-56](apps/worker/scene_worker/runtime.py:43), [apps/api/sceneworks_api/jobs_store.py:346-394](apps/api/sceneworks_api/jobs_store.py:346)
- **Finding:** The worker registration sends `capabilities: ["image_generate","image_edit","video_generate",...]` and `loadedModels: []`, but `claim_next_job` does not match queued job `type` against the worker's `capabilities` (it only checks GPU exclusivity and `requested_gpu`). A CPU-only worker (`gpu = {"id":"cpu",...}`) would happily claim an `image_generate` job and fail when `torch.cuda.is_available()` returns False.
- **Impact:** Misleading data in worker rows; future heterogenous fleets can't be expressed. Once F-003 lands (download jobs), this becomes a real correctness problem.
- **Suggested fix:** Either drop the `capabilities` field everywhere (it's dead) or extend `claim_next_job` with an `and exists (select 1 from json_each(capabilities_json) where value = jobs.type)` clause and add an integration test.
- **Confidence:** High

#### [F-016] `loadedModels` field is hard-coded to `[]` — dead feature
- **Category:** dead-code
- **Severity:** Medium
- **Location:** [apps/worker/scene_worker/runtime.py:63-67](apps/worker/scene_worker/runtime.py:63), [apps/worker/scene_worker/image_adapters.py:205-292](apps/worker/scene_worker/image_adapters.py:205)
- **Finding:** Phase 4 deliverable: "Loaded-model hint field." Worker heartbeats and `register_worker` always send `loadedModels=[]`. The `ZImageDiffusersAdapter` caches `_text_pipe`/`_img2img_pipe` and tracks `_loaded_repo`, but never reports the loaded repo back to the API.
- **Impact:** The dispatcher cannot favor a worker that already has the right model loaded — every claim risks an expensive model load. The schema field is misleading.
- **Suggested fix:** Make `register_worker`/`heartbeat` accept an injected `loaded_models` list and have the runtime pull it from the adapter (e.g., `adapter.loaded_models()` returning `[repo]`). Once populated, prefer-claim logic can be added later in `claim_next_job`.
- **Confidence:** High

#### [F-017] Reindex command is missing — Phase 2 acceptance gap
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** repo (no CLI), [apps/api/sceneworks_api/projects.py](apps/api/sceneworks_api/projects.py)
- **Finding:** Phase 2 acceptance: "Reindex command can scan a project and report assets." There is no CLI entry point, no admin API, and no script. The `project.db.assets` table is populated by `index_project_db` only when the worker writes a new asset; nothing repopulates it after a manual file removal or schema migration.
- **Impact:** If `project.db` is deleted or corrupted, there's no recovery path other than regenerating everything. Future migrations have nowhere to live.
- **Suggested fix:** Add a `python -m sceneworks_api reindex --project <path>` command (and matching `POST /projects/{id}/reindex` admin endpoint) that walks the project folder, reads sidecars, and rebuilds `assets`/`generation_sets`/`timelines` tables.
- **Confidence:** High

#### [F-018] `find_asset_sidecar` scans every sidecar on every call (asset operations are O(N))
- **Category:** efficiency
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/assets.py:48-57](apps/api/sceneworks_api/assets.py:48), [apps/worker/scene_worker/image_adapters.py:418-430](apps/worker/scene_worker/image_adapters.py:418), [apps/worker/scene_worker/video_adapters.py:306-328](apps/worker/scene_worker/video_adapters.py:306), [apps/worker/scene_worker/timeline_exporter.py:179-188](apps/worker/scene_worker/timeline_exporter.py:179)
- **Finding:** Every asset PATCH, DELETE, edit source-image lookup, video source-image lookup, and timeline export segment lookup iterates `glob("*.sceneworks.json")` and re-parses every sidecar JSON. With 500 assets and a 50-clip timeline that's 25,000 disk reads + JSON parses per export.
- **Impact:** Library responsiveness degrades quickly; timeline export becomes I/O-bound on metadata long before FFmpeg work starts.
- **Suggested fix:** Use the existing `project.db.assets` table — `select file_path from assets where id = ?` is O(1) with the existing primary key. Persist `sidecar_path` in the row too. Asset list could also read from the DB instead of globbing.
- **Confidence:** High

#### [F-019] Procedural image renderer iterates pixels in Python (very slow at 1024×1024)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** [apps/worker/scene_worker/image_adapters.py:433-463](apps/worker/scene_worker/image_adapters.py:433), [apps/worker/scene_worker/video_adapters.py:360-373](apps/worker/scene_worker/video_adapters.py:360)
- **Finding:** `render_preview_image` and `gradient_frame` use `pixels[x, y] = ...` in nested Python loops. At 1024×1024 that's ~1 million PIL pixel writes per frame; the video adapter does this for every preview frame too.
- **Impact:** Procedural previews take many seconds per image on the fallback path, even though the path is explicitly "dev/test" per Epic 1085. Worse on the video adapter which currently produces *all* video output as procedural WebPs.
- **Suggested fix:** Build the gradient with NumPy (`numpy.linspace` + `numpy.broadcast_to`) and convert to PIL via `Image.fromarray`. Cuts wall time by roughly 100×. Add `numpy` to worker requirements (already a torch transitive).
- **Confidence:** High

#### [F-020] Duplicated utility functions across modules (slugify, utc_now, write_json, read_json, safe_int, safe_float, find_project_path)
- **Category:** redundant
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/projects.py:43-49](apps/api/sceneworks_api/projects.py:43), [apps/api/sceneworks_api/jobs_store.py:17-28](apps/api/sceneworks_api/jobs_store.py:17), [apps/api/sceneworks_api/timelines.py:113-135](apps/api/sceneworks_api/timelines.py:113), [apps/worker/scene_worker/image_adapters.py:70-93](apps/worker/scene_worker/image_adapters.py:70), [apps/worker/scene_worker/video_adapters.py:259-282](apps/worker/scene_worker/video_adapters.py:259), [apps/worker/scene_worker/timeline_exporter.py:36-54](apps/worker/scene_worker/timeline_exporter.py:36)
- **Finding:** `slugify` is defined four times with slightly different fallback strings ("project", "timeline", "image", "video", "timeline-export") and different max lengths. `utc_now` is defined in five places. `write_json` / `read_json` in three each. `safe_int` / `safe_float` duplicated in worker. `find_project_path` exists in both API and worker (different exception types).
- **Impact:** Drift is already visible (different truncation lengths, different fallback strings). The `packages/shared` directory exists for exactly this purpose and is empty.
- **Suggested fix:** Promote these into a tiny `packages/shared/sceneworks_shared/` Python module installed via local path in both API and worker requirements. Centralize `slugify(prefix=...)`, `utc_now()`, `read_json`, `write_json`, `safe_int`, `safe_float`, and `find_project_path` (with a `PathNotFound` exception that callers translate).
- **Confidence:** High

#### [F-021] `web/src/main.jsx` is a 2,116-line god module
- **Category:** readability
- **Severity:** Medium
- **Location:** [apps/web/src/main.jsx](apps/web/src/main.jsx)
- **Finding:** A single JSX file holds `App` and every screen component (`LibraryScreen`, `ImageStudio`, `VideoStudio`, `EditorScreen`, `QueueScreen`, `AssetGrid`, `AssetDetail`, `AssetCard`, `FullscreenPreview`, `JobRow`, `PlaceholderSurface`), all helpers (`apiFetch`, `eventUrl`, timeline math, sort helpers), and `fallbackModels`/`navItems` constants. State for all screens lives in `App` and is drilled through props.
- **Impact:** Reviewing or modifying any single screen requires loading the whole file in context. The Editor and Image studios have already started to share concerns (asset selection, GPU pickers) that are hard to factor while everything is co-located. Future epics (Character Studio, Model Manager) will push the file past comfortable read size.
- **Suggested fix:** Split into `src/screens/{Library,ImageStudio,VideoStudio,Editor,Queue,Placeholder}.jsx`, `src/components/{AssetMedia,AssetCard,AssetGrid,AssetDetail,FullscreenPreview,JobRow}.jsx`, `src/api.js` (fetch wrappers, EventSource), and `src/timeline.js` (timeline math). Keep `App.jsx` focused on routing/state. Do not introduce TypeScript in the same change; that's a separate decision.
- **Confidence:** Medium

#### [F-022] Multiple modules execute `create table if not exists` on every write
- **Category:** redundant
- **Severity:** Medium
- **Location:** [apps/worker/scene_worker/image_adapters.py:535-553](apps/worker/scene_worker/image_adapters.py:535) (`index_project_db`), [apps/api/sceneworks_api/timelines.py:138-155](apps/api/sceneworks_api/timelines.py:138) (`ensure_timeline_db`), [apps/api/sceneworks_api/projects.py:77-135](apps/api/sceneworks_api/projects.py:77) (`create_project_db`)
- **Finding:** Three locations independently `CREATE TABLE IF NOT EXISTS assets` / `timelines`. The schema is also defined twice for `assets` (once in `projects.py` at project creation, once in `index_project_db` on every write). Drift risk if a column is added in one place but not the other.
- **Impact:** Today the schemas agree, but the next column addition is one missed copy away from a broken write. Also slightly wasteful — every asset write re-issues a CREATE TABLE.
- **Suggested fix:** Centralize per-project migrations in one helper (e.g. `apply_project_migrations(connection)`) called by `create_project_db` at create time and by a new `ensure_project_db_ready(project_path)` callable at open time. Remove the inline `create table` from `index_project_db` and `ensure_timeline_db`.
- **Confidence:** High

#### [F-023] EventHub `queue.updated` event is published but never consumed
- **Category:** dead-code
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/jobs.py:87-89,124-125,143-145,154-156,169-171,202-203,240-241](apps/api/sceneworks_api/jobs.py:87), [apps/web/src/main.jsx:278-300](apps/web/src/main.jsx:278)
- **Finding:** Every job-modifying endpoint publishes both `job.updated` and `queue.updated`. The web app only registers listeners for `job.updated` and `worker.updated`. Computing `queue_summary(request)` is non-trivial (loads up to 500 jobs and all workers from SQLite) and runs on every publish.
- **Impact:** Wasted CPU + SQLite read per mutation; misleading API contract (the event is documented by emission but no client uses it).
- **Suggested fix:** Either subscribe the web app to `queue.updated` and consume it for the topbar `Queue N` chip (currently derived client-side), or drop the `queue.updated` publish calls and the `queue_summary` recomputation. Pick one.
- **Confidence:** High

#### [F-024] No model registry caching — manifests re-parsed on every `/api/v1/models` call
- **Category:** efficiency
- **Severity:** Medium
- **Location:** [apps/api/sceneworks_api/models.py:27-38](apps/api/sceneworks_api/models.py:27)
- **Finding:** `list_models` reads both manifest files, strips comments with regex, parses JSON, merges, and sorts on every request. The web app calls this on each `refreshData()` (after every job action).
- **Impact:** Negligible at one user; under any load or once the user.models manifest grows it's wasted work. The JSONC regex bug (F-008) also runs more often than it needs to.
- **Suggested fix:** Cache by manifest mtime — `lru_cache` keyed on `(path, mtime)`, or simply read once at app start and reload when SIGHUP'd / on a dedicated `POST /api/v1/models/reload`.
- **Confidence:** High

---

## Low findings

#### [F-025] Z-Image pipeline kept loaded indefinitely — potential VRAM creep
- **Category:** efficiency
- **Severity:** Low
- **Location:** [apps/worker/scene_worker/image_adapters.py:205-292](apps/worker/scene_worker/image_adapters.py:205)
- **Finding:** `ZImageDiffusersAdapter` caches `_text_pipe` and `_img2img_pipe` and never evicts them. Switching between models loads a new one without releasing the previous (only the `_loaded_repo != repo` check forces a reload, and both pipe slots stay alive concurrently).
- **Impact:** With multiple model targets in sequence, VRAM usage grows monotonically until OOM. Per epic intent the cache is correct for warm starts, but there's no upper bound.
- **Suggested fix:** When loading a new repo, explicitly `del` the previous pipe and call `torch.cuda.empty_cache()`. Or cap the cache at one pipe per slot and document the cold-start cost.
- **Confidence:** Medium

#### [F-026] Worker has no retry cap on registration loop
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** [apps/worker/scene_worker/runtime.py:338-344](apps/worker/scene_worker/runtime.py:338)
- **Finding:** `while True: try register; except sleep(poll_seconds)` retries forever with a fixed 3-second sleep.
- **Impact:** If the API is misconfigured (wrong base URL, wrong token), the worker silently logs forever without backing off. Logs flood the journal.
- **Suggested fix:** Exponential backoff capped at 30 s. Exit (or emit a louder error) after, say, 20 failed attempts so the orchestrator can restart with corrected config.
- **Confidence:** High

#### [F-027] Retry loop has no maximum attempts
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** [apps/api/sceneworks_api/jobs_store.py:255-266](apps/api/sceneworks_api/jobs_store.py:255)
- **Finding:** `retry_job` always increments `attempts` by 1 and re-queues. There is no cap and no UI guard.
- **Impact:** A user mashing "Retry" on a permanently-failing job can produce unbounded queue churn; no signal that further retries are unlikely to help.
- **Suggested fix:** Add a `max_attempts` setting (default ~5) and surface "attempted N times" on the JobRow with the Retry button disabled at the cap. Or simply log a warning past N attempts.
- **Confidence:** Medium

#### [F-028] No image-decompression-bomb guard on source image loads
- **Category:** security
- **Severity:** Low
- **Location:** [apps/worker/scene_worker/image_adapters.py:408-415](apps/worker/scene_worker/image_adapters.py:408), [apps/worker/scene_worker/video_adapters.py:306-328](apps/worker/scene_worker/video_adapters.py:306)
- **Finding:** `Image.open(source_path)` is called without setting `Image.MAX_IMAGE_PIXELS`. A pathological PNG could OOM the worker.
- **Impact:** Local-only and user-supplied, so realistic exploit surface is small, but a 50000×50000 image dropped into `assets/uploads` would crash the worker.
- **Suggested fix:** Set `Image.MAX_IMAGE_PIXELS = 64_000_000` (or a project-appropriate cap) at module import. Catch `Image.DecompressionBombError` and surface as a user-visible job failure.
- **Confidence:** High

#### [F-029] `assets.py` updates JSON sidecar but never re-syncs the SQLite row
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** [apps/api/sceneworks_api/assets.py:85-99](apps/api/sceneworks_api/assets.py:85)
- **Finding:** `update_asset_status` writes the sidecar JSON but does not update `project.db.assets.favorite/rating/rejected/trashed`. Only the worker's `index_project_db` (on write) ever sets those columns; the API's PATCH leaves them stale.
- **Impact:** Anything that reads from the DB (currently nothing, but a future Library list that uses DB-backed sorting/filtering — see F-018) will return stale flags. Today the sidecar is the source of truth and the DB row is partly informational.
- **Suggested fix:** Update both in the same function. Better, decide which is authoritative: pick the DB for queryable fields, the sidecar for portability, and write both atomically.
- **Confidence:** High

#### [F-030] No `.env.example` checked in despite README reference
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** [README.md:32-35](README.md:32)
- **Finding:** README says "copy `.env.example` to `.env` and set: SCENEWORKS_ACCESS_TOKEN=…". There is no `.env.example` in the repo (`ls -a` confirms).
- **Impact:** First-run instructions fail. Users must guess the variable names from `docker-compose.yml`.
- **Suggested fix:** Add `/.env.example` with all `SCENEWORKS_*` keys from `docker-compose.yml`, each commented with its purpose and default.
- **Confidence:** High

#### [F-031] FFmpeg failure surfaces only the last line of stderr
- **Category:** readability
- **Severity:** Low
- **Location:** [apps/worker/scene_worker/timeline_exporter.py:334-338](apps/worker/scene_worker/timeline_exporter.py:334)
- **Finding:** `run_ffmpeg` discards all of stderr except the last line on failure.
- **Impact:** Debugging timeline-export failures from job error messages is painful — the actual ffmpeg complaint (missing codec, invalid filter chain) is usually 3–10 lines from the end.
- **Suggested fix:** Capture the last ~10 lines of stderr (or the whole thing capped at 2 kB) into the job's `error` field. Optionally write the full stderr to a sidecar log file in the temp dir.
- **Confidence:** High

#### [F-032] `package.json` carries a lone `apps/web` workspace and an unused `typescript` dep
- **Category:** dead-code
- **Severity:** Low
- **Location:** [package.json:11-13](package.json:11), [apps/web/package.json:18](apps/web/package.json:18)
- **Finding:** Root `workspaces` is `["apps/web"]` — a one-entry workspace adds tooling without benefit. The web `package.json` lists `"typescript": "latest"` but the app is JavaScript only (no `tsconfig.json`, no `.ts/.tsx` files).
- **Impact:** Extra install time; misleading first impression about the language stack.
- **Suggested fix:** Either remove `workspaces` (use a normal install) or commit to it (add other JS packages to it later). Drop `typescript` from `apps/web/package.json` until it's actually used.
- **Confidence:** High

#### [F-033] `find_timeline_file` glob fallback bypasses the index
- **Category:** efficiency
- **Severity:** Low
- **Location:** [apps/api/sceneworks_api/timelines.py:195-209](apps/api/sceneworks_api/timelines.py:195)
- **Finding:** When the DB row exists but the file path on it doesn't, the function silently falls back to globbing all `*.sceneworks.timeline.json` files. There's no log of the inconsistency.
- **Impact:** A renamed file silently re-resolves but the DB row keeps the stale path forever. The next call repeats the glob.
- **Suggested fix:** When the fallback path resolves, update the DB row with the discovered relative path. If the glob also fails, raise a clear `HTTPException(404, "Timeline file not found at indexed path X; reindex required")`.
- **Confidence:** High

#### [F-034] `JobCreateRequest` allows any `type` string — no allow-list
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** [apps/api/sceneworks_api/jobs.py:17-22](apps/api/sceneworks_api/jobs.py:17)
- **Finding:** The generic `POST /jobs` endpoint accepts `type: str` (default `placeholder`) with no validator. A typo like `image_genrate` is accepted and queued; the worker eventually fails it with "No adapter exists for this job type yet."
- **Impact:** Bad UX — failures are deferred to worker pickup time instead of rejected at submission. Also makes it easy to litter the queue with mistyped types that won't ever run.
- **Suggested fix:** Define `JOB_TYPES = Literal["placeholder","image_generate","image_edit","video_generate","video_extend","video_bridge","timeline_export","model_download","lora_import"]` and use it in `JobCreateRequest.type`.
- **Confidence:** High

---

## Informational

#### [F-035] Editor MP4 export is checked in even though Epic 1087 is "To Do"
- **Category:** —
- **Severity:** Info
- **Location:** [apps/api/sceneworks_api/timelines.py](apps/api/sceneworks_api/timelines.py), [apps/worker/scene_worker/timeline_exporter.py](apps/worker/scene_worker/timeline_exporter.py), commit `3d3eca0`
- **Finding:** Epic 1087 (SceneWorks: Editor & MP4 Export) is in the "To Do" column, but the corresponding code has already landed via commit `3d3eca0 "Add editor timelines and MP4 export"`. The features actually exist and work.
- **Impact:** Shortcut state and repo state are out of sync. Future scoping for that epic will need to be clear about what's left vs. what's done.
- **Suggested fix:** Move 1087 to "In Review" or "Done" with a comment listing what's in and any deferred items.
- **Confidence:** High

#### [F-036] Z-Image-Edit is mapped to the Z-Image-Turbo repo
- **Category:** —
- **Severity:** Info
- **Location:** [apps/worker/scene_worker/image_adapters.py:34-41](apps/worker/scene_worker/image_adapters.py:34), [config/manifests/builtin.models.jsonc:53-90](config/manifests/builtin.models.jsonc:53)
- **Finding:** Both `z_image_turbo` and `z_image_edit` resolve to the same HF repo `Tongyi-MAI/Z-Image-Turbo`. The manifest UI description notes this is intentional pending the dedicated Edit checkpoint, but the worker code does not document it.
- **Impact:** None today — `ZImageImg2ImgPipeline` against the Turbo weights is the stated stopgap. Worth a code comment so a future contributor doesn't "fix" it.
- **Suggested fix:** One-line comment in `MODEL_TARGETS` next to the `z_image_edit` entry: `# Uses Turbo weights via ZImageImg2ImgPipeline until the dedicated Edit checkpoint is released`.
- **Confidence:** High

#### [F-037] Two `find_project_path` implementations diverge on error type
- **Category:** —
- **Severity:** Info
- **Location:** [apps/api/sceneworks_api/projects.py:170-178](apps/api/sceneworks_api/projects.py:170), [apps/worker/scene_worker/image_adapters.py:95-99](apps/worker/scene_worker/image_adapters.py:95)
- **Finding:** The API version raises `HTTPException(404)`; the worker version raises `RuntimeError`. The worker version also returns the first match without checking that the path exists (the API version checks).
- **Impact:** A renamed/moved project folder is reported differently depending on whether the API or worker discovers the discrepancy. Tied to F-020 (shared utility).
- **Suggested fix:** When F-020 is addressed, the shared helper should raise `ProjectNotFound`; callers translate (API to 404, worker to a job-failure exception).
- **Confidence:** High

---

## Themes and systemic observations

- **Acceptance-criteria drift across multiple done epics.** F-002 (no asset import), F-003 (no `model_download`/`lora_import` worker handlers), F-004 (no Model Manager screen), F-005 (no non-GPU job lane), F-006 (no worker liveness), and F-017 (no reindex) each correspond to specific deliverables in Phases 1–5. The Shortcut comments record validation evidence for what was built, but no one wrote a checklist back across the original acceptance criteria. A "verify against acceptance" sub-task at epic close would have caught these.

- **Two parallel utility codebases (API and worker) with no shared module.** `packages/shared/` is a placeholder. The duplication noted in F-020 is now widespread enough that every new helper in one app gets re-implemented in the other a few weeks later (F-031, F-018, sidecar lookup, slugify, etc.). Promoting a tiny shared package now is much cheaper than after Phases 8–10.

- **The DB and the sidecar are both "sources of truth," and they disagree.** Assets get their canonical state from the sidecar JSON on read (`list_assets` globs and parses), but the DB row gets stale on PATCH (F-029) and is sometimes the only place data lives (timelines, F-033). Pick one per concern: DB for queryable indexes, sidecar for portability — and route every write through both. F-018/F-022/F-029/F-033 collapse into one refactor.

- **No tests + tightly coupled god module = upcoming refactor pain.** As the project moves into Phases 8–11 (Character Studio, Person Tracking, Multi-GPU Polish), the 2,116-line `main.jsx` and the testless backend will start to slow feature work. The split in F-021 + a small test harness in F-010 should land before another vertical slice.

- **Procedural fallback paths are over-invested in.** Procedural image and video renderers (F-019) and several "preview-only" code paths are full Python implementations. They were valuable in early phases but are now a maintenance tax — they have their own filename/sidecar/index code branches. Consider scoping them to a single small module flagged as test-only.

---

## Coverage notes

- **Reviewed:** every `.py`, `.jsx`, `.json`, `.jsonc`, `.mjs`, and Dockerfile in `apps/`, `packages/`, `scripts/`, `docker/`, `config/manifests/`. `package.json` (root + web), `docker-compose.yml`, `README.md`, `CODEGRAPH.md`, `documents/IMPLEMENTATION_PLAN.md` (Phases 0–10 read to support cross-checking).
- **Skimmed / spot-checked only:** `apps/web/src/styles.css` (1,114 lines, design CSS — not a target for this review). `documents/*_RESEARCH.md` (architectural research, not implementation).
- **Excluded:** `data/` (runtime content), `.git/`, `node_modules/` (none present yet), `.claude/`.
- **Not verified at runtime:** I did not start the stack, run FFmpeg against a real timeline, exercise the SSE stream, or load a real Z-Image checkpoint. Findings whose confidence depends on runtime behavior (F-006, F-007, F-025, F-026, F-027) are tagged Medium confidence accordingly.
- **No prior CODE_REVIEW_*.md** existed in the repo; this is the first.
