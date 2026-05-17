from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package_path in (ROOT / "apps" / "api", ROOT / "apps" / "worker", ROOT / "packages" / "shared"):
    sys.path.insert(0, str(package_path))

from sceneworks_api.jobs_store import MAX_JOB_ATTEMPTS


def load_contract_fixture_module():
    module_path = ROOT / "tests" / "test_rust_migration_contract_fixtures.py"
    spec = importlib.util.spec_from_file_location("rust_migration_contract_fixtures", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: export-rust-live-contracts.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_module = load_contract_fixture_module()
    live_root = output_dir / "live-root"
    live_payloads = fixture_module.create_live_sidecars(live_root)

    sidecar_names = {
        "project": "project.json",
        "imageAsset": "asset-image.sceneworks.json",
        "videoAsset": "asset-video.sceneworks.json",
        "generationSet": "generation-set.json",
        "recipe": "recipe.json",
        "character": "character.sceneworks.character.json",
        "timeline": "timeline.sceneworks.timeline.json",
        "personTrack": "person-track.sceneworks.person-track.json",
        "modelManifestEntry": "model-manifest-entry.json",
        "modelInstallMarker": "model-install-marker.json",
    }
    for key, filename in sidecar_names.items():
        write_json(output_dir / filename, live_payloads[key])

    lora_fixture = fixture_module.load_json(
        ROOT / "tests" / "fixtures" / "rust_migration_contracts" / "sidecars" / "lora-manifest-entry.json"
    )
    write_json(output_dir / "lora-manifest-entry.json", lora_fixture)

    store = fixture_module.JobsStore(output_dir / "jobs.db")
    store.initialize()
    job = store.create_job(
        job_type="image_generate",
        project_id="project_fixture",
        project_name="Fixture Project",
        payload={"prompt": "mist over hills", "model": "z_image_turbo"},
        requested_gpu="auto",
    )
    worker = store.register_worker(
        worker_id="worker-gpu-0",
        gpu_id="gpu-0",
        gpu_name="Fixture GPU",
        capabilities=["placeholder", "gpu", "image_generate"],
        loaded_models=["Tongyi-MAI/Z-Image-Turbo"],
    )
    counts = {status: 0 for status in fixture_module.JOB_STATUSES}
    counts[job["status"]] += 1
    write_json(output_dir / "job-snapshot.json", job)
    write_json(
        output_dir / "queue-summary.json",
        {
            "counts": counts,
            "activeJobs": [job],
            "workers": [worker],
            "maxJobAttempts": MAX_JOB_ATTEMPTS,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
