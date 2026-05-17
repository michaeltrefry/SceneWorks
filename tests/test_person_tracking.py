from __future__ import annotations

import json
from types import SimpleNamespace

from sceneworks_api.jobs_store import JobsStore
from sceneworks_api.person_tracking import (
    PersonDetectionJobRequest,
    PersonTrackJobRequest,
    create_person_detection_job,
    create_person_track_job,
    get_person_track,
    list_person_tracks,
)
from sceneworks_shared import write_json


class EventHub:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event: str, data: dict) -> None:
        self.events.append((event, data))


def request_for_project(tmp_path, project_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    registry_path = data_dir / "recent-projects.json"
    registry_path.write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    jobs_store = JobsStore(tmp_path / "jobs.db")
    jobs_store.initialize()
    state = SimpleNamespace(
        settings=SimpleNamespace(registry_path=registry_path, worker_timeout_seconds=120),
        jobs_store=jobs_store,
        event_hub=EventHub(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_person_detection_endpoint_queues_detection_job(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    request = request_for_project(tmp_path, project_path)

    job = create_person_detection_job(
        "project-1",
        PersonDetectionJobRequest(sourceAssetId="asset-video", sourceTimestamp=1.25),
        request,
    )

    assert job["type"] == "person_detect"
    assert job["payload"]["sourceAssetId"] == "asset-video"
    assert job["payload"]["sourceTimestamp"] == 1.25
    assert request.app.state.event_hub.events[-1][0] == "queue.updated"


def test_person_track_endpoint_queues_selected_detection_job(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    project_path.mkdir()
    request = request_for_project(tmp_path, project_path)
    detection = {"id": "person_1", "box": {"x": 0.3, "y": 0.2, "width": 0.2, "height": 0.6}}

    job = create_person_track_job(
        "project-1",
        PersonTrackJobRequest(
            sourceAssetId="asset-video",
            representativeFrameAssetId="asset-frame",
            detection=detection,
            trackName="Hero",
        ),
        request,
    )

    assert job["type"] == "person_track"
    assert job["payload"]["trackName"] == "Hero"
    assert job["payload"]["detection"] == detection


def test_list_person_tracks_reads_project_sidecars(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    track_dir = project_path / "person-tracks"
    track_dir.mkdir(parents=True)
    write_json(
        track_dir / "track_1.sceneworks.person-track.json",
        {
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": "project-1",
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {},
        },
    )

    tracks = list_person_tracks("project-1", request_for_project(tmp_path, project_path))

    assert tracks[0]["id"] == "track_1"
    assert tracks[0]["path"] == "person-tracks/track_1.sceneworks.person-track.json"


def test_get_person_track_reads_direct_sidecar(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    track_dir = project_path / "person-tracks"
    track_dir.mkdir(parents=True)
    write_json(
        track_dir / "track_1.sceneworks.person-track.json",
        {
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": "project-1",
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {},
        },
    )

    track = get_person_track("project-1", "track_1", request_for_project(tmp_path, project_path))

    assert track["id"] == "track_1"
    assert track["path"] == "person-tracks/track_1.sceneworks.person-track.json"
