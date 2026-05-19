from __future__ import annotations

import atexit
import json
import os
import re
import socket
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "rust_api_contract_snapshots" / "snapshots.json"
UPDATE_SNAPSHOTS = os.getenv(
    "UPDATE_SNAPSHOTS",
    os.getenv("SCENEWORKS_UPDATE_PARITY_SNAPSHOTS", ""),
).strip().lower() in {
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
    r"look|character_lora|character|lora|worker)_[0-9a-f]{8,32}\b"
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


def write_contract_manifests(config_dir: Path) -> None:
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
                  "repo": "owner/alternate-base-model",
                  "files": ["*.bin"],
                  "estimatedSizeBytes": 536870912
                },
                {
                  "provider": "huggingface",
                  "repo": "owner/base-model",
                  "files": ["*.safetensors"],
                  "default": true,
                  "estimatedSizeBytes": 12884901888
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
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime = "rust"
        write_contract_manifests(root / "config")
        port = free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.update(
            {
                "SCENEWORKS_API_HOST": "127.0.0.1",
                "SCENEWORKS_API_PORT": str(port),
                "SCENEWORKS_DATA_DIR": str(root / "data"),
                "SCENEWORKS_CONFIG_DIR": str(root / "config"),
                "SCENEWORKS_JOBS_DB_PATH": str(root / "data" / "cache" / "jobs.db"),
                "SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE": "1",
            }
        )
        rust_binary = ROOT / "target" / "debug" / (
            "sceneworks-rust-api.exe" if os.name == "nt" else "sceneworks-rust-api"
        )
        command = [str(rust_binary)] if rust_binary.exists() else ["cargo", "run", "-q", "-p", "sceneworks-rust-api"]
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_health(self.base_url, self.process, "rust")
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
class ContractRuntime:
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
def contract_runtimes(tmp_path):
    """Run two isolated Rust APIs to catch non-deterministic wire output before snapshot comparison."""
    runtimes = [
        ContractRuntime("rust-a", ServerApiHarness(tmp_path / "rust-a"), (tmp_path / "rust-a",)),
        ContractRuntime("rust-b", ServerApiHarness(tmp_path / "rust-b"), (tmp_path / "rust-b",)),
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
        # The exact JSON parser wording/offset is intentionally ignored; the
        # stable contract is the 422 envelope.
        if path.endswith(".ctx.error"):
            return "<json-decode-error>"
        if path.endswith(".ticket") and HEX_TOKEN_RE.match(normalized):
            return "<event-ticket>"
        if normalized == "rust":
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
            diffs.append(f"{path}.{key}: only in baseline")
        for key in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{key}: only in candidate")
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


def assert_runtime_consistency(
    label: str,
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    left: Any,
    right: Any,
) -> Any:
    normalized_baseline = normalize_contract(left, baseline_runtime.roots)
    normalized_candidate = normalize_contract(right, candidate_runtime.roots)
    diffs = diff_values(normalized_baseline, normalized_candidate)
    assert not diffs, (
        f"{label} contract drifted between isolated Rust runs:\n"
        + "\n".join(diffs)
        + "\n\nBaseline:\n"
        + json.dumps(normalized_baseline, indent=2, sort_keys=True)
        + "\n\nCandidate:\n"
        + json.dumps(normalized_candidate, indent=2, sort_keys=True)
    )
    return normalized_baseline


def snapshots() -> dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        return {}
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def assert_snapshot(label: str, normalized_value: Any) -> None:
    if UPDATE_SNAPSHOTS:
        UPDATED_SNAPSHOTS[label] = normalized_value
        return
    expected = snapshots().get(label)
    assert expected is not None, f"Missing Rust API contract snapshot for {label}"
    diffs = diff_values(expected, normalized_value)
    assert not diffs, (
        f"{label} no longer matches the saved Rust API contract snapshot:\n"
        + "\n".join(diffs)
        + "\n\nExpected:\n"
        + json.dumps(expected, indent=2, sort_keys=True)
        + "\n\nActual:\n"
        + json.dumps(normalized_value, indent=2, sort_keys=True)
    )


def assert_response_contract(
    label: str,
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    baseline_response: ApiResponse,
    candidate_response: ApiResponse,
    *,
    expected_status: int | None = None,
    snapshot: bool = False,
) -> Any:
    assert baseline_response.status_code == candidate_response.status_code, (
        f"{label} status drifted: baseline {baseline_response.status_code}, candidate {candidate_response.status_code}"
    )
    if expected_status is not None:
        assert baseline_response.status_code == expected_status, (
            f"{label} returned {baseline_response.status_code}, expected {expected_status}"
        )
    else:
        assert baseline_response.status_code < 400, f"{label} unexpectedly failed with {baseline_response.status_code}"
    normalized = assert_runtime_consistency(
        label,
        baseline_runtime,
        candidate_runtime,
        baseline_response.body,
        candidate_response.body,
    )
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


def character_sidecar_payload(project_path: Path, character_id: str) -> dict[str, Any]:
    for sidecar in project_path.glob("characters/*.sceneworks.character.json"):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("id") == character_id:
            return payload
    raise AssertionError(f"Missing character sidecar for {character_id} under {project_path}")


def write_person_track_sidecar(runtime: ContractRuntime) -> None:
    assert runtime.project_id is not None
    assert runtime.project_path is not None
    track_dir = runtime.project_path / "person-tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "track_fixture.sceneworks.person-track.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "track_fixture",
                "projectId": runtime.project_id,
                "name": "Hero",
                "createdAt": "2026-05-17T00:00:00Z",
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "frames": [],
                "status": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def create_projects(runtimes: list[ContractRuntime], baseline_runtime: ContractRuntime, candidate_runtime: ContractRuntime) -> None:
    responses = [runtime.request("POST", "/api/v1/projects", json_payload={"name": "Parity Project"}) for runtime in runtimes]
    for runtime, response in zip(runtimes, responses):
        runtime.project_id = response.body["id"]
        runtime.project_path = Path(response.body["path"])
    assert_response_contract(
        "project create response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def upload_assets(runtimes: list[ContractRuntime], baseline_runtime: ContractRuntime, candidate_runtime: ContractRuntime) -> None:
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
    assert_response_contract(
        "asset upload sidecar response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=True,
    )


def create_character_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    *,
    payload: dict[str, Any] | None = None,
    label: str = "character create response",
    snapshot: bool = True,
) -> tuple[list[str], Any]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters",
            json_payload=payload
            or {"name": "Mira", "type": "person", "description": "Lead performer"},
        )
        for runtime in runtimes
    ]
    character_ids = [response.body["id"] for response in responses]
    normalized = assert_response_contract(
        label,
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    return character_ids, normalized


def attach_character_reference_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    approved: bool = True,
    label: str = "character reference attach response",
    snapshot: bool = True,
) -> Any:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references",
            json_payload={
                "assetId": runtime.asset_id,
                "approved": approved,
                "role": "hero" if approved else "reference",
                "notes": "Approved face" if approved else "Primary face",
            },
        )
        for runtime, character_id in zip(runtimes, character_ids)
    ]
    return assert_response_contract(
        label,
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )


def create_character_look_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    snapshot: bool = True,
) -> tuple[list[str], Any]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks",
            json_payload={
                "name": "Rain coat",
                "description": "Night exterior look",
                "approvedReferenceIds": [runtime.asset_id],
                "recipeSettings": {"style": "noir"},
            },
        )
        for runtime, character_id in zip(runtimes, character_ids)
    ]
    normalized = assert_response_contract(
        "character look create response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    look_ids = [response.body["looks"][0]["id"] for response in responses]
    return look_ids, normalized


def write_lora_sources(runtimes: list[ContractRuntime]) -> list[Path]:
    lora_sources = []
    for runtime in runtimes:
        # The parity harness root contains each runtime's data dir, so sourcePath is both valid
        # for the API and normalized to <runtime-root> in snapshots.
        lora_dir = runtime.roots[0] / "data" / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)
        lora_source = lora_dir / "mira.safetensors"
        lora_source.write_bytes(b"lora")
        lora_sources.append(lora_source)
    return lora_sources


def attach_character_lora_pair(
    runtimes: list[ContractRuntime],
    baseline_runtime: ContractRuntime,
    candidate_runtime: ContractRuntime,
    character_ids: list[str],
    *,
    snapshot: bool = True,
) -> tuple[list[str], Any]:
    responses = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras",
            json_payload={
                "name": "Mira LoRA",
                "sourcePath": str(lora_source),
                "triggerWords": ["mira"],
                "compatibility": {"families": ["z-image"]},
            },
        )
        for runtime, character_id, lora_source in zip(runtimes, character_ids, write_lora_sources(runtimes))
    ]
    normalized = assert_response_contract(
        "character lora attach response",
        baseline_runtime,
        candidate_runtime,
        responses[0],
        responses[1],
        expected_status=201,
        snapshot=snapshot,
    )
    lora_link_ids = [response.body["loras"][0]["id"] for response in responses]
    for runtime, response in zip(runtimes, responses):
        copied_path = runtime.project_path / response.body["loras"][0]["projectPath"]
        assert copied_path.exists(), f"{runtime.name} did not copy character LoRA to {copied_path}"
        assert copied_path.read_bytes() == b"lora"
    copied_rel_paths = [response.body["loras"][0]["projectPath"] for response in responses]
    assert_runtime_consistency("character lora copied project path", baseline_runtime, candidate_runtime, copied_rel_paths[0], copied_rel_paths[1])
    return lora_link_ids, normalized


def read_sse_message(lines: Any) -> dict[str, Any]:
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                parsed_data = data
            return {"event": event, "data": parsed_data}
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    raise AssertionError("SSE stream ended before a complete event arrived")


def collect_sse_events_during(runtime: ContractRuntime, action: Any, expected_events: tuple[str, ...]) -> tuple[Any, list[dict[str, Any]]]:
    timeout = httpx.Timeout(10.0, connect=5.0, read=5.0)
    with httpx.Client(base_url=runtime.api.base_url, timeout=timeout) as client:
        ticket = client.post("/api/v1/jobs/events/ticket").json()["ticket"]
        with client.stream("GET", f"/api/v1/jobs/events?ticket={ticket}") as response:
            response.raise_for_status()
            lines = response.iter_lines()
            ready = read_sse_message(lines)
            assert ready == {"event": "ready", "data": {"status": "connected"}}
            action_result = action()
            events: list[dict[str, Any]] = []
            try:
                while len(events) < len(expected_events):
                    event = read_sse_message(lines)
                    if event["event"] in expected_events:
                        events.append(event)
            except httpx.ReadTimeout as exc:
                raise AssertionError(f"{runtime.name} did not emit expected SSE events {expected_events}: {events}") from exc
    assert [event["event"] for event in events] == list(expected_events)
    return action_result, events


def test_system_manifest_and_http_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes

    for label, path, method, expected in [
        ("health response", "/api/v1/health", "GET", 200),
        ("access response", "/api/v1/access", "GET", 200),
        ("auth verify response", "/api/v1/auth/verify", "POST", 200),
        ("event ticket response", "/api/v1/jobs/events/ticket", "POST", 200),
        ("model manifest response", "/api/v1/models", "GET", 200),
        ("lora manifest response", "/api/v1/loras?modelFamily=wan-video", "GET", 200),
    ]:
        responses = [runtime.request(method, path) for runtime in contract_runtimes]
        assert_response_contract(label, baseline_runtime, candidate_runtime, responses[0], responses[1], expected_status=expected, snapshot=True)

    cors_headers = {
        "origin": "http://localhost:5173",
        "access-control-request-method": "POST",
        "access-control-request-headers": "X-SceneWorks-Token",
    }
    cors = [runtime.request("OPTIONS", "/api/v1/jobs", headers=cors_headers) for runtime in contract_runtimes]
    assert cors[0].status_code == cors[1].status_code == 200
    assert cors[0].headers.get("access-control-allow-origin") == cors[1].headers.get("access-control-allow-origin")
    assert cors[0].headers.get("access-control-allow-origin") == "http://localhost:5173"
    for response in cors:
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        assert "x-sceneworks-token" in allow_headers


def test_project_asset_sidecar_delete_and_db_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    listed_projects = [runtime.request("GET", "/api/v1/projects") for runtime in contract_runtimes]
    assert_response_contract("project list response", baseline_runtime, candidate_runtime, listed_projects[0], listed_projects[1], snapshot=True)

    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    patched_assets = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/status",
            json_payload={"favorite": True, "rating": 4},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "asset status project DB update response",
        baseline_runtime,
        candidate_runtime,
        patched_assets[0],
        patched_assets[1],
        expected_status=200,
        snapshot=True,
    )

    listed_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    assert_response_contract("asset list response", baseline_runtime, candidate_runtime, listed_assets[0], listed_assets[1], snapshot=True)

    sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_sidecar = assert_runtime_consistency("persisted upload sidecar", baseline_runtime, candidate_runtime, sidecars[0], sidecars[1])
    assert_snapshot("persisted upload sidecar", normalized_sidecar)

    counts = [db_counts(runtime.project_path) for runtime in contract_runtimes]
    assert_runtime_consistency("project DB counts after sidecar writes", baseline_runtime, candidate_runtime, counts[0], counts[1])

    reindexed = [runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/reindex") for runtime in contract_runtimes]
    assert_response_contract("project reindex response", baseline_runtime, candidate_runtime, reindexed[0], reindexed[1], expected_status=200, snapshot=True)

    deleted = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}")
        for runtime in contract_runtimes
    ]
    assert_response_contract("asset delete response", baseline_runtime, candidate_runtime, deleted[0], deleted[1], expected_status=200, snapshot=True)

    visible_after_delete = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    assert_response_contract(
        "asset list after delete response",
        baseline_runtime,
        candidate_runtime,
        visible_after_delete[0],
        visible_after_delete[1],
        expected_status=200,
        snapshot=True,
    )

    purged = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/assets/{runtime.asset_id}/purge")
        for runtime in contract_runtimes
    ]
    assert_response_contract("asset purge response", baseline_runtime, candidate_runtime, purged[0], purged[1], expected_status=200, snapshot=True)


@pytest.mark.parity
def test_character_lifecycle_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)

    listed_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list response",
        baseline_runtime,
        candidate_runtime,
        listed_characters[0],
        listed_characters[1],
        expected_status=200,
        snapshot=True,
    )

    fetched_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}")
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character detail response",
        baseline_runtime,
        candidate_runtime,
        fetched_characters[0],
        fetched_characters[1],
        expected_status=200,
        snapshot=True,
    )

    updated_characters = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}",
            json_payload={"description": "Lead performer, updated", "archived": False},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character update response",
        baseline_runtime,
        candidate_runtime,
        updated_characters[0],
        updated_characters[1],
        expected_status=200,
        snapshot=True,
    )

    archived_characters = [
        runtime.request("POST", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/archive")
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    normalized_post_archive_response = assert_response_contract(
        "character archive response",
        baseline_runtime,
        candidate_runtime,
        archived_characters[0],
        archived_characters[1],
        expected_status=200,
        snapshot=True,
    )

    hidden_archived = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list after archive response",
        baseline_runtime,
        candidate_runtime,
        hidden_archived[0],
        hidden_archived[1],
        expected_status=200,
        snapshot=True,
    )

    visible_archived = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters?includeArchived=true")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "character list include archived response",
        baseline_runtime,
        candidate_runtime,
        visible_archived[0],
        visible_archived[1],
        expected_status=200,
        snapshot=True,
    )

    delete_archive_ids, _ = create_character_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        payload={"name": "Delete Archive", "type": "object"},
        label="character create for delete archive response",
    )
    delete_archive_responses = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}",
        )
        for runtime, character_id in zip(contract_runtimes, delete_archive_ids)
    ]
    normalized_delete_archive_response = assert_response_contract(
        "character delete archive response",
        baseline_runtime,
        candidate_runtime,
        delete_archive_responses[0],
        delete_archive_responses[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_delete_archive_response == normalized_post_archive_response

    purge_ids, _ = create_character_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        payload={"name": "Purge Me", "type": "object"},
        label="character create for purge response",
    )
    purge_responses = [
        runtime.request("DELETE", f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/purge")
        for runtime, character_id in zip(contract_runtimes, purge_ids)
    ]
    assert_response_contract(
        "character purge response",
        baseline_runtime,
        candidate_runtime,
        purge_responses[0],
        purge_responses[1],
        expected_status=200,
        snapshot=True,
    )
    for runtime, character_id in zip(contract_runtimes, purge_ids):
        assert not (runtime.project_path / "characters" / f"{character_id}.sceneworks.character.json").exists()


@pytest.mark.parity
def test_character_reference_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)

    attach_character_reference_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
        approved=False,
    )

    updated_references = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references/{runtime.asset_id}",
            json_payload={"approved": True, "role": "hero", "notes": "Approved face"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character reference update response",
        baseline_runtime,
        candidate_runtime,
        updated_references[0],
        updated_references[1],
        expected_status=200,
        snapshot=True,
    )

    sidecars = [
        character_sidecar_payload(runtime.project_path, character_id)
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    normalized_sidecar = assert_runtime_consistency("persisted character sidecar", baseline_runtime, candidate_runtime, sidecars[0], sidecars[1])
    assert_snapshot("persisted character sidecar", normalized_sidecar)

    listed_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    normalized_asset_response = assert_response_contract(
        "asset list after character reference response",
        baseline_runtime,
        candidate_runtime,
        listed_assets[0],
        listed_assets[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_asset_response[0]["metadata"]["characterReferences"][0]["characterId"] == "character_fixture"

    asset_sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_asset_sidecar = assert_runtime_consistency(
        "asset sidecar after character reference",
        baseline_runtime,
        candidate_runtime,
        asset_sidecars[0],
        asset_sidecars[1],
    )
    assert_snapshot("asset sidecar after character reference", normalized_asset_sidecar)

    removed_references = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/references/{runtime.asset_id}",
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "character reference delete response",
        baseline_runtime,
        candidate_runtime,
        removed_references[0],
        removed_references[1],
        expected_status=200,
        snapshot=True,
    )

    cleaned_assets = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/assets") for runtime in contract_runtimes]
    normalized_cleaned_asset_response = assert_response_contract(
        "asset list after character reference removal response",
        baseline_runtime,
        candidate_runtime,
        cleaned_assets[0],
        cleaned_assets[1],
        expected_status=200,
        snapshot=True,
    )
    assert normalized_cleaned_asset_response[0]["metadata"]["characterReferences"] == []

    cleaned_asset_sidecars = [sidecar_payload(runtime.project_path, runtime.asset_id) for runtime in contract_runtimes]
    normalized_cleaned_asset_sidecar = assert_runtime_consistency(
        "asset sidecar after character reference removal",
        baseline_runtime,
        candidate_runtime,
        cleaned_asset_sidecars[0],
        cleaned_asset_sidecars[1],
    )
    assert_snapshot("asset sidecar after character reference removal", normalized_cleaned_asset_sidecar)
    for asset_sidecar in cleaned_asset_sidecars:
        assert asset_sidecar["metadata"]["characterReferences"] == []


@pytest.mark.parity
def test_character_look_and_lora_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)
    attach_character_reference_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)

    look_ids, normalized_look_response = create_character_look_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
    )

    updated_looks = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/{look_id}",
            json_payload={
                "name": "Rain coat revised",
                "description": "Night exterior hero look",
                "approvedReferenceIds": [runtime.asset_id],
                "recipeSettings": {"style": "noir", "lens": "85mm"},
            },
        )
        for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids)
    ]
    normalized_look_update_response = assert_response_contract(
        "character look update response",
        baseline_runtime,
        candidate_runtime,
        updated_looks[0],
        updated_looks[1],
        expected_status=200,
        snapshot=True,
    )

    lora_link_ids, _ = attach_character_lora_pair(
        contract_runtimes,
        baseline_runtime,
        candidate_runtime,
        character_ids,
    )

    updated_loras = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/{link_id}",
            json_payload={
                "name": "Mira LoRA revised",
                "triggerWords": ["mira", "rain"],
                "defaultWeight": 0.65,
                "compatibility": {"families": ["z-image"], "notes": "parity"},
                "scope": "project",
            },
        )
        for runtime, character_id, link_id in zip(contract_runtimes, character_ids, lora_link_ids)
    ]
    assert_response_contract(
        "character lora update response",
        baseline_runtime,
        candidate_runtime,
        updated_loras[0],
        updated_loras[1],
        expected_status=200,
        snapshot=True,
    )

    deleted_looks = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/{look_id}",
        )
        for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids)
    ]
    assert_response_contract(
        "character look delete response",
        baseline_runtime,
        candidate_runtime,
        deleted_looks[0],
        deleted_looks[1],
        expected_status=200,
        snapshot=True,
    )

    detached_loras = [
        runtime.request(
            "DELETE",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/{link_id}",
        )
        for runtime, character_id, link_id in zip(contract_runtimes, character_ids, lora_link_ids)
    ]
    assert_response_contract(
        "character lora delete response",
        baseline_runtime,
        candidate_runtime,
        detached_loras[0],
        detached_loras[1],
        expected_status=200,
        snapshot=True,
    )

    assert normalized_look_response["looks"][0]["approvedReferenceIds"] == ["asset_fixture"]
    assert normalized_look_update_response["looks"][0]["recipeSettings"]["lens"] == "85mm"


@pytest.mark.parity
def test_character_test_job_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)
    character_ids, _ = create_character_pair(contract_runtimes, baseline_runtime, candidate_runtime)
    attach_character_reference_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)
    look_ids, _ = create_character_look_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)
    attach_character_lora_pair(contract_runtimes, baseline_runtime, candidate_runtime, character_ids, snapshot=False)

    test_jobs = []
    event_sets = []
    for runtime, character_id, look_id in zip(contract_runtimes, character_ids, look_ids):
        response, events = collect_sse_events_during(
            runtime,
            lambda runtime=runtime, character_id=character_id, look_id=look_id: runtime.request(
                "POST",
                f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/test-jobs",
                json_payload={"prompt": "portrait in rain", "lookId": look_id, "count": 2},
            ),
            ("job.updated", "queue.updated"),
        )
        test_jobs.append(response)
        event_sets.append(events)

    assert_response_contract(
        "character test job response",
        baseline_runtime,
        candidate_runtime,
        test_jobs[0],
        test_jobs[1],
        expected_status=201,
        snapshot=True,
    )
    normalized_events = assert_runtime_consistency(
        "character test job SSE events",
        baseline_runtime,
        candidate_runtime,
        event_sets[0],
        event_sets[1],
    )
    assert_snapshot("character test job SSE events", normalized_events)


def test_timeline_and_worker_job_creation_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    upload_assets(contract_runtimes, baseline_runtime, candidate_runtime)

    created_timelines = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines",
            json_payload={"name": "Main timeline", "aspectRatio": "16:9", "fps": 24},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("timeline create response", baseline_runtime, candidate_runtime, created_timelines[0], created_timelines[1], expected_status=201, snapshot=True)

    updated_timelines = []
    for runtime, created in zip(contract_runtimes, created_timelines):
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
    assert_response_contract("timeline update response", baseline_runtime, candidate_runtime, updated_timelines[0], updated_timelines[1], expected_status=200, snapshot=True)

    listed_timelines = [runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines") for runtime in contract_runtimes]
    assert_response_contract("timeline list response", baseline_runtime, candidate_runtime, listed_timelines[0], listed_timelines[1], expected_status=200, snapshot=True)

    detail_timelines = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}")
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    assert_response_contract("timeline detail response", baseline_runtime, candidate_runtime, detail_timelines[0], detail_timelines[1], expected_status=200, snapshot=True)

    frame_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/items/item-1/frames",
            json_payload={"playheadSeconds": 0.5, "intendedUse": "reuse"},
        )
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    assert_response_contract("timeline frame job response", baseline_runtime, candidate_runtime, frame_jobs[0], frame_jobs[1], expected_status=201, snapshot=True)

    export_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/timelines/{created.body['id']}/exports",
            json_payload={"resolution": 640, "fps": 24, "requestedGpu": "auto"},
        )
        for runtime, created in zip(contract_runtimes, created_timelines)
    ]
    assert_response_contract("timeline export job response", baseline_runtime, candidate_runtime, export_jobs[0], export_jobs[1], expected_status=201, snapshot=True)


def test_person_tracking_and_replace_person_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    for runtime in contract_runtimes:
        write_person_track_sidecar(runtime)

    listed_tracks = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/person-tracks")
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track list response", baseline_runtime, candidate_runtime, listed_tracks[0], listed_tracks[1], expected_status=200, snapshot=True)

    track_details = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/person-tracks/track_fixture")
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track detail response", baseline_runtime, candidate_runtime, track_details[0], track_details[1], expected_status=200, snapshot=True)

    detection_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/person-tracks/detections",
            json_payload={"sourceAssetId": "asset-video", "sourceTimestamp": 1.25},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("person detection job response", baseline_runtime, candidate_runtime, detection_jobs[0], detection_jobs[1], expected_status=201, snapshot=True)

    track_jobs = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/person-tracks/jobs",
            json_payload={
                "sourceAssetId": "asset-video",
                "representativeFrameAssetId": "asset-frame",
                "detection": {
                    "id": "person_1",
                    "box": {"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6},
                },
                "trackName": "Hero",
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("person track job response", baseline_runtime, candidate_runtime, track_jobs[0], track_jobs[1], expected_status=201, snapshot=True)

    replace_jobs = [
        runtime.request(
            "POST",
            "/api/v1/video/jobs",
            json_payload={
                "projectId": runtime.project_id,
                "projectName": "Parity Project",
                "mode": "replace_person",
                "prompt": "hero walks through rain",
                "sourceClipAssetId": "asset-video",
                "personTrackId": "track_fixture",
                "characterId": "character_fixture",
            },
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("replace person job response", baseline_runtime, candidate_runtime, replace_jobs[0], replace_jobs[1], expected_status=201, snapshot=True)


def test_model_lora_and_image_job_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

    model_jobs = [
        runtime.request("POST", "/api/v1/models/base-model/download", json_payload={"requestedGpu": ""})
        for runtime in contract_runtimes
    ]
    assert_response_contract("model download job response", baseline_runtime, candidate_runtime, model_jobs[0], model_jobs[1], expected_status=201, snapshot=True)

    lora_jobs = [
        runtime.request(
            "POST",
            "/api/v1/loras/import",
            json_payload={"repo": "owner/style-lora", "name": "Imported LoRA", "files": ["adapter.safetensors"]},
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("lora import job response", baseline_runtime, candidate_runtime, lora_jobs[0], lora_jobs[1], expected_status=201, snapshot=True)

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
        for runtime in contract_runtimes
    ]
    for runtime, response in zip(contract_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_contract("image job response", baseline_runtime, candidate_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)


def test_job_state_transition_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)

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
        for runtime in contract_runtimes
    ]
    assert_response_contract("worker registration response", baseline_runtime, candidate_runtime, workers[0], workers[1], expected_status=200, snapshot=True)

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
        for runtime in contract_runtimes
    ]
    for runtime, response in zip(contract_runtimes, image_jobs):
        runtime.job_id = response.body["id"]
    assert_response_contract("transition image job response", baseline_runtime, candidate_runtime, image_jobs[0], image_jobs[1], expected_status=201, snapshot=True)

    claimed_jobs = [runtime.request("POST", "/api/v1/jobs/claim", json_payload={"workerId": "parity-worker"}) for runtime in contract_runtimes]
    assert_response_contract("job claim response", baseline_runtime, candidate_runtime, claimed_jobs[0], claimed_jobs[1], expected_status=200, snapshot=True)

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
        for runtime in contract_runtimes
    ]
    assert_response_contract("busy worker heartbeat response", baseline_runtime, candidate_runtime, busy_workers[0], busy_workers[1], expected_status=200, snapshot=True)

    cancel_requests = [runtime.request("POST", f"/api/v1/jobs/{runtime.job_id}/cancel") for runtime in contract_runtimes]
    assert_response_contract("job cancel request response", baseline_runtime, candidate_runtime, cancel_requests[0], cancel_requests[1], expected_status=200, snapshot=True)

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
        for runtime in contract_runtimes
    ]
    assert_response_contract("job terminal progress response", baseline_runtime, candidate_runtime, canceled_jobs[0], canceled_jobs[1], expected_status=200, snapshot=True)

    queues = [runtime.request("GET", "/api/v1/queue") for runtime in contract_runtimes]
    assert_response_contract("queue summary after transitions", baseline_runtime, candidate_runtime, queues[0], queues[1], expected_status=200, snapshot=True)


def concurrent_claim_result(runtime: ContractRuntime) -> list[Any]:
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


def test_concurrent_claim_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes
    claims = [concurrent_claim_result(runtime) for runtime in contract_runtimes]
    normalized_claims = assert_runtime_consistency("concurrent claim response set", baseline_runtime, candidate_runtime, claims[0], claims[1])
    assert_snapshot("concurrent claim response set", normalized_claims)
    for runtime_claims in claims:
        assert sum(1 for claim in runtime_claims if claim["job"] is not None) == 1
        assert sum(1 for claim in runtime_claims if claim["job"] is None) == 1


def test_error_case_contracts(contract_runtimes):
    baseline_runtime, candidate_runtime = contract_runtimes

    missing_projects = [runtime.request("GET", "/api/v1/projects/project_missing") for runtime in contract_runtimes]
    assert_response_contract("missing project error response", baseline_runtime, candidate_runtime, missing_projects[0], missing_projects[1], expected_status=404, snapshot=True)

    invalid_statuses = [runtime.request("GET", "/api/v1/jobs?status=not_a_status") for runtime in contract_runtimes]
    assert_response_contract("invalid job status error response", baseline_runtime, candidate_runtime, invalid_statuses[0], invalid_statuses[1], expected_status=400, snapshot=True)

    malformed_bodies = [
        runtime.request(
            "POST",
            "/api/v1/projects",
            headers={"content-type": "application/json"},
            content='{"name":',
        )
        for runtime in contract_runtimes
    ]
    assert_response_contract("malformed json error response", baseline_runtime, candidate_runtime, malformed_bodies[0], malformed_bodies[1], expected_status=422, snapshot=True)

    create_projects(contract_runtimes, baseline_runtime, candidate_runtime)
    characters = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters",
            json_payload={"name": "Errors", "type": "person"},
        )
        for runtime in contract_runtimes
    ]
    character_ids = [response.body["id"] for response in characters]

    missing_characters = [
        runtime.request("GET", f"/api/v1/projects/{runtime.project_id}/characters/character_missing")
        for runtime in contract_runtimes
    ]
    assert_response_contract(
        "missing character error response",
        baseline_runtime,
        candidate_runtime,
        missing_characters[0],
        missing_characters[1],
        expected_status=404,
        snapshot=True,
    )

    missing_looks = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/looks/look_missing",
            json_payload={"name": "Missing"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "missing character look error response",
        baseline_runtime,
        candidate_runtime,
        missing_looks[0],
        missing_looks[1],
        expected_status=404,
        snapshot=True,
    )

    missing_loras = [
        runtime.request(
            "PATCH",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras/character_lora_missing",
            json_payload={"name": "Missing"},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "missing character lora error response",
        baseline_runtime,
        candidate_runtime,
        missing_loras[0],
        missing_loras[1],
        expected_status=404,
        snapshot=True,
    )

    invalid_lora_sources = [
        runtime.request(
            "POST",
            f"/api/v1/projects/{runtime.project_id}/characters/{character_id}/loras",
            json_payload={"name": "Missing source", "sourcePath": str(runtime.roots[0] / "data" / "loras" / "missing.safetensors")},
        )
        for runtime, character_id in zip(contract_runtimes, character_ids)
    ]
    assert_response_contract(
        "invalid character lora source error response",
        baseline_runtime,
        candidate_runtime,
        invalid_lora_sources[0],
        invalid_lora_sources[1],
        expected_status=400,
        snapshot=True,
    )

