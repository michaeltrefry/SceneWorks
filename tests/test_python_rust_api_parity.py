from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "python_rust_api_parity" / "snapshots.json"
UPDATE_SNAPSHOTS = os.getenv("SCENEWORKS_UPDATE_PARITY_SNAPSHOTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
UPDATED_SNAPSHOTS: dict[str, Any] = {}
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x01\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe"
    b"\x02\xfeA\xe2&\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
PREFIXED_ID_RE = re.compile(
    r"\b(project|asset|job|timeline|track|transition|genset|generation_set|"
    r"look|character|lora|worker)_[0-9a-f]{8,32}\b"
)
UPLOAD_SUFFIX_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+)*)-[0-9a-f]{8}(?=\.)")
HEX_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
CLAIM_WORKER_RE = re.compile(r"\bclaim-worker-[a-z]\b")


def write_updated_snapshots() -> None:
    if not UPDATE_SNAPSHOTS:
        return
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(UPDATED_SNAPSHOTS, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


atexit.register(write_updated_snapshots)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def pythonpath_env(previous: str | None) -> str:
    paths = [
        ROOT / "apps" / "api",
        ROOT / "apps" / "worker",
        ROOT / "packages" / "shared",
    ]
    parts = [str(path) for path in paths]
    if previous:
        parts.append(previous)
    return os.pathsep.join(parts)


def write_parity_manifests(config_dir: Path) -> None:
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "builtin.models.jsonc").write_text(
        """
        {
          // Comments are part of the manifest contract.
          "schemaVersion": 1,
          "models": [
            {
              "id": "base-model",
              "name": "Base Model",
              "family": "z-image",
              "type": "image",
              "adapter": "z_image_diffusers",
              "capabilities": ["text_to_image"],
              "downloads": [
                {
                  "provider": "huggingface",
                  "repo": "owner/base-model",
                  "files": ["*.safetensors"]
                }
              ],
              "paths": {},
              "defaults": { "width": 1024, "height": 1024 },
              "limits": {},
              "loraCompatibility": { "families": ["z-image"] },
              "ui": { "label": "Base" }
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "user.models.jsonc").write_text(
        """
        {
          "schemaVersion": 1,
          "models": [
            { "id": "base-model", "name": "User Model", "ui": { "label": "User" } }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "builtin.loras.jsonc").write_text(
        """
        {
          "schemaVersion": 1,
          "loras": [
            {
              "id": "style-lora",
              "name": "Style LoRA",
              "family": "z-image",
              "triggerWords": ["style"],
              "compatibility": { "families": ["z-image", "wan-video"] },
              "source": { "provider": "local", "path": "loras/style.safetensors" }
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    (manifest_dir / "user.loras.jsonc").write_text(
        '{ "schemaVersion": 1, "loras": [] }\n',
        encoding="utf-8",
    )


def wait_for_health(base_url: str, process: subprocess.Popen[str], runtime: str) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"{runtime} API exited early with code {process.returncode}: {stderr}")
        try:
            response = httpx.get(f"{base_url}/api/v1/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(f"{runtime} API did not become healthy within 30s: {last_error}")


def parse_response_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


@dataclass
class ApiResponse:
    status_code: int
    body: Any
    headers: dict[str, str]


class ServerApiHarness:
    def __init__(self, root: Path, runtime: str) -> None:
        if runtime == "rust" and shutil.which("cargo") is None:
            if os.getenv("SCENEWORKS_REQUIRE_CARGO_PARITY"):
                pytest.fail("cargo is required for Python/Rust API parity tests")
            pytest.skip("cargo is required for Python/Rust API parity tests")

        self.root = root
        self.runtime = runtime
        write_parity_manifests(root / "config")
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.update(
            {
                "SCENEWORKS_API_RUNTIME": runtime,
                "SCENEWORKS_API_HOST": "127.0.0.1",
                "SCENEWORKS_API_PORT": str(port),
                "SCENEWORKS_DATA_DIR": str(root / "data"),
                "SCENEWORKS_CONFIG_DIR": str(root / "config"),
                "SCENEWORKS_JOBS_DB_PATH": str(root / "data" / "cache" / "jobs.db"),
                "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
                "PYTHONPATH": pythonpath_env(env.get("PYTHONPATH")),
            }
        )
        rust_binary = ROOT / "target" / "debug" / (
            "sceneworks-rust-api.exe" if os.name == "nt" else "sceneworks-rust-api"
        )
        command = (
            [
                sys.executable,
                "-m",
                "uvicorn",
                "sceneworks_api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ]
            if runtime == "python"
            else (
                [str(rust_binary)]
                if rust_binary.exists()
                else ["cargo", "run", "-q", "-p", "sceneworks-rust-api"]
            )
        )
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_health(self.base_url, self.process, runtime)
        self.client = httpx.Client(base_url=self.base_url, timeout=10)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | None = None,
    ) -> ApiResponse:
        response = self.client.request(
            method,
            path,
            json=json_payload,
            files=files,
            headers=headers,
            content=content,
        )
        return ApiResponse(response.status_code, parse_response_body(response), dict(response.headers))

    def close(self) -> None:
        self.client.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@dataclass
class ParityRuntime:
    name: str
    api: ServerApiHarness
    roots: tuple[Path, ...]
    project_id: str | None = None
    project_path: Path | None = None
    asset_id: str | None = None
    job_id: str | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | None = None,
    ) -> ApiResponse:
        response = self.api.request(
            method,
            path,
            json_payload=json_payload,
            files=files,
            headers=headers,
            content=content,
        )
        assert response.status_code < 500, (
            f"{self.name} {method} {path} returned {response.status_code}: {response.body}"
        )
        return response


@pytest.fixture()
def parity_runtimes(tmp_path):
    runtimes = [
        ParityRuntime("python", ServerApiHarness(tmp_path / "python", "python"), (tmp_path / "python",)),
        ParityRuntime("rust", ServerApiHarness(tmp_path / "rust", "rust"), (tmp_path / "rust",)),
    ]
    try:
        yield runtimes
    finally:
        for runtime in runtimes:
            runtime.api.close()


def normalize_contract(value: Any, roots: tuple[Path, ...], path: str = "$") -> Any:
    if isinstance(value, dict):
        return {key: normalize_contract(item, roots, f"{path}.{key}") for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_contract(item, roots, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for root in roots:
            normalized = normalized.replace(str(root).replace("\\", "/"), "<runtime-root>")
        normalized = TIMESTAMP_RE.sub("<timestamp>", normalized)
        normalized = UPLOAD_SUFFIX_RE.sub(r"\1-<upload-suffix>", normalized)
        normalized = PREFIXED_ID_RE.sub(lambda match: f"{match.group(1)}_fixture", normalized)
        normalized = CLAIM_WORKER_RE.sub("claim-worker-fixture", normalized)
        # FastAPI/Pydantic and Axum report parser internals differently. The parity contract is
        # the stable 422 envelope; the exact JSON parser wording/offset is intentionally ignored.
        if path.endswith(".ctx.error"):
            return "<json-decode-error>"
        if path.endswith(".ticket") and HEX_TOKEN_RE.match(normalized):
            return "<event-ticket>"
        if normalized in {"python", "rust"}:
            return "<runtime>"
        return normalized
    if path.endswith(".loc[1]") and isinstance(value, int):
        return "<json-error-offset>"
    return value


def diff_values(left: Any, right: Any, path: str = "$") -> list[str]:
    if path.endswith(".elapsedSeconds"):
        if left is None and right is None:
            return []
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(left - right) <= 2:
            return []
        return [f"{path}: elapsed seconds drifted beyond tolerance: {left!r} != {right!r}"]
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        diffs = []
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{key}: only in python")
        for key in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{key}: only in rust")
        for key in sorted(left_keys & right_keys):
            diffs.extend(diff_values(left[key], right[key], f"{path}.{key}"))
            if len(diffs) >= 12:
                break
        return diffs
    if isinstance(left, list):
        diffs = []
        if len(left) != len(right):
            diffs.append(f"{path}: list length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diffs.extend(diff_values(left_item, right_item, f"{path}[{index}]"))
            if len(diffs) >= 12:
                break
        return diffs
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


def assert_parity(label: str, python_runtime: ParityRuntime, rust_runtime: ParityRuntime, left: Any, right: Any) -> Any:
    normalized_python = normalize_contract(left, python_runtime.roots)
    normalized_rust = normalize_contract(right, rust_runtime.roots)
    diffs = diff_values(normalized_python, normalized_rust)
    assert not diffs, (
        f"{label} contract drifted between Python and Rust:\n"
        + "\n".join(diffs)
        + "\n\nPython:\n"
        + json.dumps(normalized_python, indent=2, sort_keys=True)
        + "\n\nRust:\n"
        + json.dumps(normalized_rust, indent=2, sort_keys=True)
    )
    return normalized_python


def snapshots() -> dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        return {}
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def assert_snapshot(label: str, normalized_value: Any) -> None:
    if UPDATE_SNAPSHOTS:
        UPDATED_SNAPSHOTS[label] = normalized_value
        return
    expected = snapshots().get(label)
    assert expected is not None, f"Missing parity snapshot for {label}"
    diffs = diff_values(expected, normalized_value)
    assert not diffs, (
        f"{label} no longer matches the saved parity baseline:\n"
        + "\n".join(diffs)
        + "\n\nExpected:\n"
        + json.dumps(expected, indent=2, sort_keys=True)
        + "\n\nActual:\n"
        + json.dumps(normalized_value, indent=2, sort_keys=True)
    )


def assert_response_parity(
    label: str,
    python_runtime: ParityRuntime,
    rust_runtime: ParityRuntime,
    python_response: ApiResponse,
    rust_response: ApiResponse,
    *,
    expected_status: int | None = None,
    snapshot: bool = False,
) -> Any:
    assert python_response.status_code == rust_response.status_code, (
        f"{label} status drifted: Python {python_response.status_code}, Rust {rust_response.status_code}"
    )
    if expected_status is not None:
        assert python_response.status_code == expected_status, (
            f"{label} returned {python_response.status_code}, expected {expected_status}"
        )
    else:
        assert python_response.status_code < 400, f"{label} unexpectedly failed with {python_response.status_code}"
    normalized = assert_parity(label, python_runtime, rust_runtime, python_response.body, rust_response.body)
    if snapshot:
        assert_snapshot(label, normalized)
    return normalized


def db_counts(project_path: Path) -> dict[str, int]:
    with sqlite3.connect(project_path / "project.db") as connection:
        return {
            "assets": connection.execute("select count(*) from assets").fetchone()[0],
            "generationSets": connection.execute("select count(*) from generation_sets").fetchone()[0],
            "timelines": connection.execute("select count(*) from timelines").fetchone()[0],
        }


def sidecar_payload(project_path: Path, asset_id: str) -> dict[str, Any]:
    for sidecar in project_path.glob("assets/**/*.sceneworks.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("id") == asset_id:
            return payload
    raise AssertionError(f"Missing sidecar for {asset_id} under {project_path}")


def create_projects(runtimes: list[ParityRuntime], python_runtime: ParityRuntime, rust_runtime: ParityRuntime) -> None:
    responses = [runtime.request("POST", "/api/v1/projects", json_payload={"name": "Parity Project"}) for runtime in runtimes]
    for runtime, response in zip(runtimes, responses):
        runtime.project_id = response.body["id"]
        runtime.project_path = Path(response.body["path"])
    assert_response_parity(
        "project create response",
        python_runtime,
        rust_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def upload_assets(runtimes: list[ParityRuntime], python_runtime: ParityRuntime, rust_runtime: ParityRuntime) -> None:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/assets",
            files={"file": ("fixture image.png", PNG_1X1, "image/png")},
        )
        for runtime in runtimes
    ]
    for runtime, response in zip(runtimes, responses):
        runtime.asset_id = response.body["id"]
    assert_response_parity(
        "asset upload sidecar response",
        python_runtime,
        rust_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def test_system_manifest_and_http_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes

    for label, path, method, expected in [
        ("health response", "/api/v1/health", "GET", 200),
        ("access response", "/api/v1/access", "GET", 200),
        ("auth verify response", "/api/v1/auth/verify", "POST", 200),
        ("event ticket response", "/api/v1/jobs/events/ticket", "POST", 200),
        ("model manifest response", "/api/v1/models", "GET", 200),
        ("lora manifest response", "/api/v1/loras?modelFamily=wan-video", "GET", 200),
    ]:
        responses = [runtime.request(method, path) for runtime in parity_runtimes]
        assert_response_parity(label, python_runtime, rust_runtime, responses[0], responses[1], expected_status=expected, snapshot=True)

    cors_headers = {
        "origin": "http://localhost:5173",
        "access-control-request-method": "POST",
        "access-control-request-headers": "X-SceneWorks-Token",
    }
    cors = [runtime.request("OPTIONS", "/api/v1/jobs", headers=cors_headers) for runtime in parity_runtimes]
    assert cors[0].status_code == cors[1].status_code == 200
    assert cors[0].headers.get("access-control-allow-origin") == cors[1].headers.get("access-control-allow-origin")
    assert cors[0].headers.get("access-control-allow-origin") == "http://localhost:5173"
    for response in cors:
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        assert "x-sceneworks-token" in allow_headers


def test_project_asset_sidecar_delete_and_db_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes
    create_projects(parity_runtimes, python_runtime, rust_runtime)

    listed_projects = [runtime.request("GET", "/api/v1/projects") for runtime in parity_runtimes]
    assert_response_parity("project list response", python_runtime, rust_runtime, listed_projects[0], listed_projects[1], snapshot=True)

    upload_assets(parity_runtimes, python_runtime, rust_runtime)
    patched_assets = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/status",
            json_payload={"favorite": True, "rating": 4},
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity(
        "asset status project DB update response",
        python_runtime,
        rust_runtime,
        patched_assets[0],
        patched_assets[1],
        expected_status=200,
        snapshot=True,
    )

    listed_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in parity_runtimes]
    assert_response_parity("asset list response", python_runtime, rust_runtime, listed_assets[0], listed_assets[1], snapshot=True)

    sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in parity_runtimes]
    normalized_sidecar = assert_parity("persisted upload sidecar", python_runtime, rust_runtime, sidecars[0], sidecars[1])
    assert_snapshot("persisted upload sidecar", normalized_sidecar)

    counts = [db_counts(runtime.project_path) for runtime in parity_runtimes]
    assert_parity("project DB counts after sidecar writes", python_runtime, rust_runtime, counts[0], counts[1])

    reindexed = [runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/reindex") for runtime in parity_runtimes]
    assert_response_parity("project reindex response", python_runtime, rust_runtime, reindexed[0], reindexed[1], expected_status=200, snapshot=True)

    deleted = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}")
        for runtime in parity_runtimes
    ]
    assert_response_parity("asset delete response", python_runtime, rust_runtime, deleted[0], deleted[1], expected_status=200, snapshot=True)

    visible_after_delete = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in parity_runtimes]
    assert_response_parity(
        "asset list after delete response",
        python_runtime,
        rust_runtime,
        visible_after_delete[0],
        visible_after_delete[1],
        expected_status=200,
        snapshot=True,
    )

    purged = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/purge")
        for runtime in parity_runtimes
    ]
    assert_response_parity("asset purge response", python_runtime, rust_runtime, purged[0], purged[1], expected_status=200, snapshot=True)


def test_timeline_and_worker_job_creation_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes
    create_projects(parity_runtimes, python_runtime, rust_runtime)
    upload_assets(parity_runtimes, python_runtime, rust_runtime)

    created_timelines = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines",
            json_payload={"name": "Main timeline", "aspectRatio": "16:9", "fps": 24},
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("timeline create response", python_runtime, rust_runtime, created_timelines[0], created_timelines[1], expected_status=201, snapshot=True)

    updated_timelines = []
    for runtime, created in zip(parity_runtimes, created_timelines):
        timeline = json.loads(json.dumps(created.body))
        timeline_id = timeline["id"]
        timeline["tracks"][0]["items"] = [
            {
                "id": "item-1",
                "trackId": "track_main",
                "assetId": runtime.asset_id,
                "type": "image",
                "displayName": "Still",
                "sourceIn": 0,
                "sourceOut": 1,
                "timelineStart": 0,
                "timelineEnd": 1,
                "speed": 1,
                "fit": "fit",
                "volume": 1,
            }
        ]
        updated_timelines.append(
            runtime.request(
                "PUT",
                f"/api/v1/projects/{runtime.project_id}/timelines/{timeline_id}",
                json_payload={"timeline": timeline},
            )
        )
    assert_response_parity("timeline update response", python_runtime, rust_runtime, updated_timelines[0], updated_timelines[1], expected_status=200, snapshot=True)

    listed_timelines = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines") for runtime in parity_runtimes]
    assert_response_parity("timeline list response", python_runtime, rust_runtime, listed_timelines[0], listed_timelines[1], expected_status=200, snapshot=True)

    detail_timelines = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}")
        for runtime, created in zip(parity_runtimes, created_timelines)
    ]
    assert_response_parity("timeline detail response", python_runtime, rust_runtime, detail_timelines[0], detail_timelines[1], expected_status=200, snapshot=True)

    frame_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/items/item-1/frames",
            json_payload={"playheadSeconds": 0.5, "intendedUse": "reuse"},
        )
        for runtime, created in zip(parity_runtimes, created_timelines)
    ]
    assert_response_parity("timeline frame job response", python_runtime, rust_runtime, frame_jobs[0], frame_jobs[1], expected_status=201, snapshot=True)

    export_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/exports",
            json_payload={"resolution": 640, "fps": 24, "requestedGpu": "auto"},
        )
        for runtime, created in zip(parity_runtimes, created_timelines)
    ]
    assert_response_parity("timeline export job response", python_runtime, rust_runtime, export_jobs[0], export_jobs[1], expected_status=201, snapshot=True)


def test_model_lora_and_image_job_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes
    create_projects(parity_runtimes, python_runtime, rust_runtime)

    model_jobs = [
        runtime.request("POST", "/api/v1/models/base-model/download", json_payload={"requestedGpu": ""})
        for runtime in parity_runtimes
    ]
    assert_response_parity("model download job response", python_runtime, rust_runtime, model_jobs[0], model_jobs[1], expected_status=201, snapshot=True)

    lora_jobs = [
        runtime.request(
            "POST",
            "/api/v1/loras/import",
            json_payload={"repo": "owner/style-lora", "name": "Imported LoRA", "files": ["adapter.safetensors"]},
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("lora import job response", python_runtime, rust_runtime, lora_jobs[0], lora_jobs[1], expected_status=201, snapshot=True)

    image_jobs = [
        runtime.request(
            "POST",
            "/api/v1/image/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "prompt": "mist over hills",
                "model": "base-model",
                "seed": 101,
                "requestedGpu": "auto",
            },
        )
        for runtime in parity_runtimes
    ]
    for runtime, response in zip(parity_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_parity("image job response", python_runtime, rust_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)


def test_job_state_transition_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes
    create_projects(parity_runtimes, python_runtime, rust_runtime)

    workers = [
        runtime.request(
            "POST",
            "/api/v1/workers/register",
            json_payload={
                "workerId": "parity-worker",
                "gpuId": "gpu-0",
                "gpuName": "Fixture GPU",
                "capabilities": ["image_generate"],
                "loadedModels": ["Tongyi-MAI/Z-Image-Turbo"],
            },
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("worker registration response", python_runtime, rust_runtime, workers[0], workers[1], expected_status=200, snapshot=True)

    image_jobs = [
        runtime.request(
            "POST",
            "/api/v1/image/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "prompt": "mist over hills",
                "model": "base-model",
                "seed": 101,
                "requestedGpu": "auto",
            },
        )
        for runtime in parity_runtimes
    ]
    for runtime, response in zip(parity_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_parity("transition image job response", python_runtime, rust_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)

    claimed_jobs = [runtime.request("POST", "/api/v1/jobs/claim", json_payload={"workerId": "parity-worker"}) for runtime in parity_runtimes]
    assert_response_parity("job claim response", python_runtime, rust_runtime, claimed_jobs[0], claimed_jobs[1], expected_status=200, snapshot=True)

    busy_workers = [
        runtime.request(
            "POST",
            "/api/v1/workers/parity-worker/heartbeat",
            json_payload={
                "status": "busy",
                "currentJobId": runtime.job_id,
                "loadedModels": ["Tongyi-MAI/Z-Image-Turbo"],
            },
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("busy worker heartbeat response", python_runtime, rust_runtime, busy_workers[0], busy_workers[1], expected_status=200, snapshot=True)

    cancel_requests = [runtime.request("POST", f"/api/v1/jobs/{runtime.job_id}/cancel") for runtime in parity_runtimes]
    assert_response_parity("job cancel request response", python_runtime, rust_runtime, cancel_requests[0], cancel_requests[1], expected_status=200, snapshot=True)

    canceled_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/jobs/{runtime.job_id}/progress",
            json_payload={
                "status": "canceled",
                "stage": "canceled",
                "progress": 1,
                "message": "Worker canceled the job before completion.",
            },
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("job terminal progress response", python_runtime, rust_runtime, canceled_jobs[0], canceled_jobs[1], expected_status=200, snapshot=True)

    queues = [runtime.request("GET", "/api/v1/queue") for runtime in parity_runtimes]
    assert_response_parity("queue summary after transitions", python_runtime, rust_runtime, queues[0], queues[1], expected_status=200, snapshot=True)


def concurrent_claim_result(runtime: ParityRuntime) -> list[Any]:
    runtime.request(
        "POST",
        "/api/v1/workers/register",
        json_payload={"workerId": "claim-worker-a", "gpuId": "gpu-0", "capabilities": ["image_generate"], "loadedModels": []},
    )
    runtime.request(
        "POST",
        "/api/v1/workers/register",
        json_payload={"workerId": "claim-worker-b", "gpuId": "gpu-0", "capabilities": ["image_generate"], "loadedModels": []},
    )
    runtime.request("POST", "/api/v1/jobs", json_payload={"type": "image_generate", "payload": {}, "requestedGpu": "auto"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(runtime.request, "POST", "/api/v1/jobs/claim", json_payload={"workerId": worker_id})
            for worker_id in ["claim-worker-a", "claim-worker-b"]
        ]
    responses = [future.result() for future in futures]
    assert all(response.status_code == 200 for response in responses)
    claims = [response.body for response in responses]
    claims.sort(key=lambda claim: claim["job"]["id"] if claim["job"] else "")
    return claims


def test_concurrent_claim_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes
    claims = [concurrent_claim_result(runtime) for runtime in parity_runtimes]
    normalized_claims = assert_parity("concurrent claim response set", python_runtime, rust_runtime, claims[0], claims[1])
    assert_snapshot("concurrent claim response set", normalized_claims)
    for runtime_claims in claims:
        assert sum(1 for claim in runtime_claims if claim["job"] is not None) == 1
        assert sum(1 for claim in runtime_claims if claim["job"] is None) == 1


def test_error_case_contracts(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes

    missing_projects = [runtime.request("GET", "/api/v1/projects/project_missing") for runtime in parity_runtimes]
    assert_response_parity("missing project error response", python_runtime, rust_runtime, missing_projects[0], missing_projects[1], expected_status=404, snapshot=True)

    invalid_statuses = [runtime.request("GET", "/api/v1/jobs?status=not_a_status") for runtime in parity_runtimes]
    assert_response_parity("invalid job status error response", python_runtime, rust_runtime, invalid_statuses[0], invalid_statuses[1], expected_status=400, snapshot=True)

    malformed_bodies = [
        runtime.request(
            "POST",
            "/api/v1/projects",
            headers={"content-type": "application/json"},
            content='{"name":',
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("malformed json error response", python_runtime, rust_runtime, malformed_bodies[0], malformed_bodies[1], expected_status=422, snapshot=True)
