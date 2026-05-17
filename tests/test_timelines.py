from __future__ import annotations

import json
from types import SimpleNamespace

from sceneworks_api.jobs_store import JobsStore
from sceneworks_api.timelines import (
    FrameExtractRequest,
    TimelineDocument,
    TimelineItem,
    TimelineTrack,
    extract_timeline_frame,
    save_timeline,
)
from sceneworks_shared import index_asset, write_json


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


def test_timeline_item_defaults_current_version_history():
    item = TimelineItem(
        trackId="track_main",
        assetId="asset-1",
        displayName="Clip",
        sourceIn=0,
        sourceOut=4,
        timelineStart=0,
        timelineEnd=4,
    )

    assert item.currentVersionAssetId == "asset-1"
    assert item.versionAssetIds == ["asset-1"]
    assert item.versionHistory[0].source == "original"


def test_extract_timeline_frame_queues_worker_job_with_source_metadata(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    video_dir = project_path / "assets" / "videos"
    video_dir.mkdir(parents=True)
    media_path = video_dir / "clip.webp"
    media_path.write_bytes(b"webp")
    asset = {
        "id": "asset-1",
        "projectId": "project-1",
        "type": "video",
        "displayName": "Clip",
        "createdAt": "2026-05-17T00:00:00Z",
        "generationSetId": None,
        "file": {"path": "assets/videos/clip.webp", "duration": 8},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
    }
    write_json(video_dir / "clip.sceneworks.json", asset)
    index_asset(project_path, asset)
    timeline = TimelineDocument(
        id="timeline_test",
        projectId="project-1",
        name="Timeline",
        tracks=[
            TimelineTrack(
                id="track_main",
                name="Main",
                kind="video",
                items=[
                    TimelineItem(
                        id="item-1",
                        trackId="track_main",
                        assetId="asset-1",
                        displayName="Clip",
                        sourceIn=2,
                        sourceOut=6,
                        timelineStart=10,
                        timelineEnd=14,
                        speed=1,
                    )
                ],
            )
        ],
    )
    save_timeline(project_path, timeline)
    request = request_for_project(tmp_path, project_path)

    job = extract_timeline_frame(
        "project-1",
        "timeline_test",
        "item-1",
        FrameExtractRequest(playheadSeconds=12.5, intendedUse="first_frame"),
        request,
    )

    assert job["type"] == "frame_extract"
    assert job["payload"]["sourceAssetId"] == "asset-1"
    assert job["payload"]["sourceTimestamp"] == 4.5
    assert job["payload"]["timelineId"] == "timeline_test"
    assert request.app.state.event_hub.events[-1][0] == "queue.updated"
