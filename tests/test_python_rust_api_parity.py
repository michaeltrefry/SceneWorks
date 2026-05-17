from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from sceneworks_api.main import create_app
from sceneworks_api.settings import Settings


ROOT = Path(__file__).resolve().parents[1]
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe"
    b"\x02\xfeA\xe2&\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
)
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
HEX_ID_REPLACEMENTS = (
    (re.compile(r"project_[0-9a-f]{32}"), "project_fixture"),
    (re.compile(r"asset_[0-9a-f]{32}"), "asset_fixture"),
    (re.compile(r"job_[0-9a-f]{32}"), "job_fixture"),
    (re.compile(r"timeline_[0-9a-f]{32}"), "timeline_fixture"),
)
UPLOAD_SUFFIX_RE = re.compile(r"fixture-image-[0-9a-f]{8}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
              "downloads": [],
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


def wait_for_health(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 180
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"Rust API exited early with code {process.returncode}: {stderr}")
        try:
            response = httpx.get(f"{base_url}/api/v1/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(f"Rust API did not become healthy: {last_error}")


@dataclass
class ApiResponse:
    status_code: int
    json: Any


class PythonApiHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        write_parity_manifests(root / "config")
        monkeypatch.setenv("SCENEWORKS_API_RUNTIME", "python")
        monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(root / "data"))
        monkeypatch.setenv("SCENEWORKS_CONFIG_DIR", str(root / "config"))
        monkeypatch.setenv("SCENEWORKS_JOBS_DB_PATH", str(root / "cache" / "jobs.db"))
        self.client = TestClient(create_app(Settings()))

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> ApiResponse:
        response = self.client.request(method, path, json=json_payload, files=files)
        return ApiResponse(response.status_code, response.json())

    def close(self) -> None:
        self.client.close()


class RustApiHarness:
    def __init__(self, root: Path) -> None:
        if shutil.which("cargo") is None:
            pytest.skip("cargo is required for Python/Rust API parity tests")

        self.root = root
        write_parity_manifests(root / "config")
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.update(
            {
                "SCENEWORKS_API_RUNTIME": "rust",
                "SCENEWORKS_API_HOST": "127.0.0.1",
                "SCENEWORKS_API_PORT": str(port),
                "SCENEWORKS_DATA_DIR": str(root / "data"),
                "SCENEWORKS_CONFIG_DIR": str(root / "config"),
                "SCENEWORKS_JOBS_DB_PATH": str(root / "cache" / "jobs.db"),
                "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
            }
        )
        self.process = subprocess.Popen(
            ["cargo", "run", "-q", "-p", "sceneworks-rust-api"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_health(self.base_url, self.process)
        self.client = httpx.Client(base_url=self.base_url, timeout=10)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Any = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> ApiResponse:
        response = self.client.request(method, path, json=json_payload, files=files)
        return ApiResponse(response.status_code, response.json())

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
    api: PythonApiHarness | RustApiHarness
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
    ) -> ApiResponse:
        response = self.api.request(method, path, json_payload=json_payload, files=files)
        assert response.status_code < 500, f"{self.name} {method} {path} returned {response.status_code}: {response.json}"
        return response


@pytest.fixture()
def parity_runtimes(tmp_path, monkeypatch):
    python_api = PythonApiHarness(tmp_path / "python", monkeypatch)
    rust_api = RustApiHarness(tmp_path / "rust")
    runtimes = [
        ParityRuntime("python", python_api, (tmp_path / "python",)),
        ParityRuntime("rust", rust_api, (tmp_path / "rust",)),
    ]
    try:
        yield runtimes
    finally:
        for runtime in runtimes:
            runtime.api.close()


def normalize_contract(value: Any, roots: tuple[Path, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_contract(item, roots)
            for key, item in sorted(value.items())
            if key not in {"elapsedSeconds"}
        }
    if isinstance(value, list):
        return [normalize_contract(item, roots) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for root in roots:
            normalized = normalized.replace(str(root).replace("\\", "/"), "<runtime-root>")
        normalized = TIMESTAMP_RE.sub("<timestamp>", normalized)
        normalized = UPLOAD_SUFFIX_RE.sub("fixture-image-<asset-suffix>", normalized)
        for pattern, replacement in HEX_ID_REPLACEMENTS:
            normalized = pattern.sub(replacement, normalized)
        if normalized in {"python", "rust"}:
            return "<runtime>"
        return normalized
    return value


def diff_values(left: Any, right: Any, path: str = "$") -> list[str]:
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


def assert_parity(label: str, python_runtime: ParityRuntime, rust_runtime: ParityRuntime, left: Any, right: Any) -> None:
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


def assert_response_parity(
    label: str,
    python_runtime: ParityRuntime,
    rust_runtime: ParityRuntime,
    python_response: ApiResponse,
    rust_response: ApiResponse,
) -> None:
    assert python_response.status_code == rust_response.status_code, (
        f"{label} status drifted: Python {python_response.status_code}, Rust {rust_response.status_code}"
    )
    assert_parity(label, python_runtime, rust_runtime, python_response.json, rust_response.json)


def db_counts(project_path: Path) -> dict[str, int]:
    with sqlite3.connect(project_path / "project.db") as connection:
        return {
            "assets": connection.execute("select count(*) from assets").fetchone()[0],
            "generationSets": connection.execute("select count(*) from generation_sets").fetchone()[0],
            "timelines": connection.execute("select count(*) from timelines").fetchone()[0],
        }


def sidecar_payload(project_path: Path, asset_id: str) -> dict[str, Any]:
    sidecars = list(project_path.glob("assets/**/*.sceneworks.json"))
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("id") == asset_id:
            return payload
    raise AssertionError(f"Missing sidecar for {asset_id} under {project_path}")


def test_python_and_rust_api_contracts_stay_in_parity(parity_runtimes):
    python_runtime, rust_runtime = parity_runtimes

    health = [runtime.request("GET", "/api/v1/health") for runtime in parity_runtimes]
    assert_response_parity("health response", python_runtime, rust_runtime, health[0], health[1])

    models = [runtime.request("GET", "/api/v1/models") for runtime in parity_runtimes]
    assert_response_parity("model manifest response", python_runtime, rust_runtime, models[0], models[1])

    loras = [runtime.request("GET", "/api/v1/loras?modelFamily=wan-video") for runtime in parity_runtimes]
    assert_response_parity("lora manifest response", python_runtime, rust_runtime, loras[0], loras[1])

    projects = [
        runtime.request("POST", "/api/v1/projects", json_payload={"name": "Parity Project"})
        for runtime in parity_runtimes
    ]
    for runtime, response in zip(parity_runtimes, projects):
        runtime.project_id = response.json["id"]
        runtime.project_path = Path(response.json["path"])
    assert_response_parity("project create response", python_runtime, rust_runtime, projects[0], projects[1])

    listed_projects = [runtime.request("GET", "/api/v1/projects") for runtime in parity_runtimes]
    assert_response_parity("project list response", python_runtime, rust_runtime, listed_projects[0], listed_projects[1])

    uploaded_assets = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/assets",
            files={"file": ("fixture image.png", PNG_1X1, "image/png")},
        )
        for runtime in parity_runtimes
    ]
    for runtime, response in zip(parity_runtimes, uploaded_assets):
        runtime.asset_id = response.json["id"]
    assert_response_parity("asset upload sidecar response", python_runtime, rust_runtime, uploaded_assets[0], uploaded_assets[1])

    patched_assets = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/status",
            json_payload={"favorite": True, "rating": 4},
        )
        for runtime in parity_runtimes
    ]
    assert_response_parity("asset status project DB update response", python_runtime, rust_runtime, patched_assets[0], patched_assets[1])

    listed_assets = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets")
        for runtime in parity_runtimes
    ]
    assert_response_parity("asset list response", python_runtime, rust_runtime, listed_assets[0], listed_assets[1])

    sidecars = [
        sidecar_payload(runtime.project_path, runtime.asset_id)
        for runtime in parity_runtimes
        if runtime.project_path and runtime.asset_id
    ]
    assert_parity("persisted upload sidecar", python_runtime, rust_runtime, sidecars[0], sidecars[1])

    counts = [db_counts(runtime.project_path) for runtime in parity_runtimes if runtime.project_path]
    assert_parity("project DB counts after sidecar writes", python_runtime, rust_runtime, counts[0], counts[1])

    reindexed = [
        runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/reindex")
        for runtime in parity_runtimes
    ]
    assert_response_parity("project reindex response", python_runtime, rust_runtime, reindexed[0], reindexed[1])

    worker_registration = [
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
    assert_response_parity("worker registration response", python_runtime, rust_runtime, worker_registration[0], worker_registration[1])

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
        runtime.job_id = response.json["id"]
    assert_response_parity("image job response", python_runtime, rust_runtime, image_jobs[0], image_jobs[1])

    claimed_jobs = [
        runtime.request("POST", "/api/v1/jobs/claim", json_payload={"workerId": "parity-worker"})
        for runtime in parity_runtimes
    ]
    assert_response_parity("job claim response", python_runtime, rust_runtime, claimed_jobs[0], claimed_jobs[1])

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
    assert_response_parity("busy worker heartbeat response", python_runtime, rust_runtime, busy_workers[0], busy_workers[1])

    cancel_requests = [
        runtime.request("POST", f"/api/v1/jobs/{runtime.job_id}/cancel")
        for runtime in parity_runtimes
    ]
    assert_response_parity("job cancel request response", python_runtime, rust_runtime, cancel_requests[0], cancel_requests[1])

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
    assert_response_parity("job terminal progress response", python_runtime, rust_runtime, canceled_jobs[0], canceled_jobs[1])

    queues = [runtime.request("GET", "/api/v1/queue") for runtime in parity_runtimes]
    assert_response_parity("queue summary after transitions", python_runtime, rust_runtime, queues[0], queues[1])
