from __future__ import annotations

import json
from types import SimpleNamespace

from sceneworks_api.models import (
    LoraImportRequest,
    create_lora_import_job,
    download_size_from_siblings,
    format_bytes,
    lora_catalog,
    load_manifest,
    model_is_installed,
    model_install_marker,
    strip_jsonc_comments,
)


def test_jsonc_comment_stripping_preserves_url_strings():
    payload = '{"repo":"https://example.test/model", "name":"ok"} // trailing comment\n'

    assert strip_jsonc_comments(payload) == '{"repo":"https://example.test/model", "name":"ok"} \n'


def test_manifest_cache_reloads_when_mtime_changes(tmp_path):
    manifest = tmp_path / "models.jsonc"
    manifest.write_text('{"models":[{"id":"first"}]}', encoding="utf-8")

    assert load_manifest(manifest) == [{"id": "first"}]

    manifest.write_text('{"models":[{"id":"second"}]}', encoding="utf-8")
    assert load_manifest(manifest) == [{"id": "second"}]


def test_partial_model_directory_is_not_installed(tmp_path):
    model_dir = tmp_path / "models" / "owner__model"
    model_dir.mkdir(parents=True)
    (model_dir / "partial.bin").write_bytes(b"partial")

    assert not model_is_installed(model_dir)


def test_model_directory_with_completion_marker_is_installed(tmp_path):
    model_dir = tmp_path / "models" / "owner__model"
    model_dir.mkdir(parents=True)
    model_install_marker(model_dir).write_text("{}", encoding="utf-8")

    assert model_is_installed(model_dir)


def test_download_size_from_siblings_respects_allow_patterns():
    siblings = [
        {"rfilename": "model-00001.safetensors", "size": 100},
        {"rfilename": "model-00002.safetensors", "size": 200},
        {"rfilename": "README.md", "size": 50},
    ]

    assert download_size_from_siblings(siblings, ["*.safetensors"]) == 300


def test_download_size_from_siblings_returns_none_when_unknown():
    assert download_size_from_siblings([{"rfilename": "model.bin"}]) is None


def test_format_bytes_for_model_catalog():
    assert format_bytes(None) is None
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"


def test_lora_catalog_merges_global_and_project_scopes(tmp_path):
    config_dir = tmp_path / "config"
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    project_path = data_dir / "projects" / "noir.sceneworks"
    project_path.mkdir(parents=True)
    (data_dir / "loras" / "global_style").mkdir(parents=True)
    (project_path / "loras" / "imports" / "mira").mkdir(parents=True)
    (manifest_dir / "builtin.loras.jsonc").write_text(
        '{"schemaVersion":1,"loras":[{"id":"built_in","name":"Built In","family":"z-image"}]}',
        encoding="utf-8",
    )
    (manifest_dir / "user.loras.jsonc").write_text(
        '{"schemaVersion":1,"loras":[{"id":"global_style","name":"Global Style","family":"z-image","source":{"path":"loras/global_style"}}]}',
        encoding="utf-8",
    )
    (project_path / "loras").mkdir(exist_ok=True)
    (project_path / "loras" / "manifest.jsonc").write_text(
        '{"schemaVersion":1,"loras":[{"id":"mira","name":"Mira","family":"z-image","source":{"path":"loras/imports/mira"}}]}',
        encoding="utf-8",
    )
    registry_path = data_dir / "recent-projects.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps([{"id": "project-1", "name": "Noir", "path": str(project_path)}]),
        encoding="utf-8",
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(config_dir=config_dir, data_dir=data_dir, registry_path=registry_path),
            ),
        ),
    )

    loras = lora_catalog(request, "project-1")

    scopes = {lora["id"]: lora["scope"] for lora in loras}
    assert scopes == {"built_in": "builtin", "global_style": "global", "mira": "project"}
    install_states = {lora["id"]: lora["installState"] for lora in loras}
    assert install_states["built_in"] == "installed"
    assert install_states["mira"] == "installed"


def test_project_lora_import_targets_project_manifest_and_folder(tmp_path):
    class FakeJobsStore:
        def __init__(self):
            self.created = None

        def create_job(self, **kwargs):
            self.created = kwargs
            return {
                "id": "job-1",
                "type": kwargs["job_type"],
                "status": "queued",
                "payload": kwargs["payload"],
                "projectId": kwargs["project_id"],
            }

        def list_jobs(self, limit=500):
            return []

        def list_workers(self):
            return []

        def mark_stale_workers_interrupted(self, _timeout):
            return {"jobs": [], "workers": []}

    class FakeEventHub:
        def publish(self, *_args):
            pass

    config_dir = tmp_path / "config"
    (config_dir / "manifests").mkdir(parents=True)
    (config_dir / "manifests" / "user.loras.jsonc").write_text('{"schemaVersion":1,"loras":[]}', encoding="utf-8")
    data_dir = tmp_path / "data"
    project_path = data_dir / "projects" / "noir.sceneworks"
    (project_path / "loras").mkdir(parents=True)
    registry_path = data_dir / "recent-projects.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps([{"id": "project-1", "name": "Noir", "path": str(project_path)}]),
        encoding="utf-8",
    )
    jobs_store = FakeJobsStore()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(
                    config_dir=config_dir,
                    data_dir=data_dir,
                    registry_path=registry_path,
                    worker_timeout_seconds=90,
                ),
                jobs_store=jobs_store,
                event_hub=FakeEventHub(),
            ),
        ),
    )

    job = create_lora_import_job(
        LoraImportRequest(
            name="Mira Style",
            sourcePath=str(tmp_path / "mira.safetensors"),
            family="z-image",
            scope="project",
            projectId="project-1",
        ),
        request,
    )

    assert job["projectId"] == "project-1"
    payload = jobs_store.created["payload"]
    assert payload["targetDir"] == str(project_path / "loras" / "imports" / "mira_style")
    assert payload["manifestPath"] == str(project_path / "loras" / "manifest.jsonc")
    assert payload["manifestEntry"]["scope"] == "project"
    assert payload["manifestEntry"]["source"]["path"] == "loras/imports/mira_style"
    assert not (project_path / "loras" / "manifest.jsonc").exists()
