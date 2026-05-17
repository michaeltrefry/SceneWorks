from __future__ import annotations

import json
import sqlite3

from sceneworks_api.projects import PROJECT_FOLDERS
from sceneworks_api.timelines import find_timeline_file
from sceneworks_shared import reindex_project, write_json


def test_project_folders_include_person_tracks():
    assert "person-tracks" in PROJECT_FOLDERS


def test_find_timeline_file_heals_stale_db_path(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    timeline_dir = project_path / "timelines"
    timeline_dir.mkdir(parents=True)
    actual_path = timeline_dir / "renamed.sceneworks.timeline.json"
    write_json(
        actual_path,
        {
            "id": "timeline-1",
            "projectId": "project-1",
            "name": "Main",
            "aspectRatio": "16:9",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration": 0,
            "tracks": [],
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z",
        },
    )
    with sqlite3.connect(project_path / "project.db") as connection:
        connection.execute(
            """
            create table timelines (
              id text primary key,
              name text not null,
              file_path text not null,
              aspect_ratio text not null,
              width integer not null,
              height integer not null,
              fps integer not null,
              duration real not null default 0,
              created_at text not null,
              updated_at text not null
            )
            """
        )
        connection.execute(
            """
            insert into timelines (
              id, name, file_path, aspect_ratio, width, height, fps, duration, created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("timeline-1", "Main", "timelines/missing.sceneworks.timeline.json", "16:9", 1280, 720, 30, 0, "x", "x"),
        )

    assert find_timeline_file(project_path, "timeline-1") == actual_path

    with sqlite3.connect(project_path / "project.db") as connection:
        row = connection.execute("select file_path from timelines where id = ?", ("timeline-1",)).fetchone()
    assert row[0] == "timelines/renamed.sceneworks.timeline.json"


def test_reindex_project_rebuilds_asset_generation_set_and_timeline_tables(tmp_path):
    project_path = tmp_path / "project.sceneworks"
    image_dir = project_path / "assets" / "images"
    genset_dir = project_path / "generation-sets"
    timeline_dir = project_path / "timelines"
    image_dir.mkdir(parents=True)
    genset_dir.mkdir()
    timeline_dir.mkdir()
    write_json(
        image_dir / "image.sceneworks.json",
        {
            "id": "asset-1",
            "type": "image",
            "displayName": "Image",
            "createdAt": "2026-05-17T00:00:00Z",
            "generationSetId": "genset-1",
            "file": {"path": "assets/images/image.png"},
            "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
        },
    )
    write_json(
        genset_dir / "genset-1.json",
        {
            "id": "genset-1",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "test",
            "createdAt": "2026-05-17T00:00:00Z",
            "jobId": "job-1",
        },
    )
    timeline = {
        "id": "timeline-1",
        "name": "Main",
        "aspectRatio": "16:9",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "duration": 3.5,
        "createdAt": "2026-05-17T00:00:00Z",
        "updatedAt": "2026-05-17T00:00:00Z",
    }
    (timeline_dir / "main.sceneworks.timeline.json").write_text(json.dumps(timeline), encoding="utf-8")

    counts = reindex_project(project_path)

    assert counts == {"assets": 1, "generationSets": 1, "timelines": 1}
    with sqlite3.connect(project_path / "project.db") as connection:
        assert connection.execute("select count(*) from assets").fetchone()[0] == 1
        assert connection.execute("select count(*) from generation_sets").fetchone()[0] == 1
        assert connection.execute("select count(*) from timelines").fetchone()[0] == 1
