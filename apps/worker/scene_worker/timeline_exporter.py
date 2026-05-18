from __future__ import annotations

from dataclasses import dataclass
import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sceneworks_shared import find_asset_sidecar_path, find_project_path, index_asset, read_json, safe_float, slugify, utc_now

from .image_adapters import write_json
from .settings import WorkerSettings


ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]

ASSET_FOLDERS = ("assets/images", "assets/videos", "assets/uploads", "assets/frames", "assets/renders")


@dataclass(frozen=True)
class ExportRequest:
    project_id: str
    timeline_id: str
    timeline_name: str
    timeline_path: str
    resolution: int
    fps: int


def export_request_from_job(job: dict[str, Any]) -> ExportRequest:
    payload = job["payload"]
    return ExportRequest(
        project_id=payload["projectId"],
        timeline_id=payload["timelineId"],
        timeline_name=payload.get("timelineName", "Timeline"),
        timeline_path=payload["timelinePath"],
        resolution=int(payload.get("resolution", 720)),
        fps=int(payload.get("fps", 30)),
    )


def run_timeline_export(
    *,
    settings: WorkerSettings,
    job: dict[str, Any],
    progress: ProgressCallback,
    cancel_requested: CancelCallback,
) -> dict[str, Any]:
    request = export_request_from_job(job)
    project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
    timeline = read_json(project_path / request.timeline_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for timeline export but was not found on PATH.")

    width, height = output_dimensions(timeline.get("aspectRatio", "16:9"), request.resolution)
    items = sorted(main_track_items(timeline), key=lambda item: item.get("timelineStart", 0))
    if not items:
        raise RuntimeError("Timeline has no main video items to export.")

    with tempfile.TemporaryDirectory(prefix=f"sceneworks_export_{job['id']}_") as tmp:
        tmp_path = Path(tmp)
        segments: list[dict[str, Any]] = []
        cursor = 0.0
        total = len(items)
        for index, item in enumerate(items):
            if cancel_requested():
                raise InterruptedError("Timeline export canceled by user.")
            start = safe_float(item.get("timelineStart"), 0)
            if start > cursor:
                gap_duration = start - cursor
                gap_path = tmp_path / f"segment_{len(segments):04d}_gap.mp4"
                render_black_segment(ffmpeg, gap_path, gap_duration, width, height, request.fps)
                segments.append({"path": gap_path, "duration": gap_duration, "transition": None})
                cursor = start

            asset = find_asset(project_path, item["assetId"])
            segment_path = tmp_path / f"segment_{len(segments):04d}_{slugify(item.get('displayName', 'item'), fallback='timeline-export', max_length=48)}.mp4"
            duration = render_item_segment(
                ffmpeg=ffmpeg,
                project_path=project_path,
                item=item,
                asset=asset,
                output_path=segment_path,
                width=width,
                height=height,
                fps=request.fps,
            )
            segments.append(
                {
                    "path": segment_path,
                    "duration": duration,
                    "transition": (item.get("transitionIn") or {}).get("type"),
                    "transitionDuration": safe_float((item.get("transitionIn") or {}).get("duration"), 0.5),
                }
            )
            cursor = max(cursor, safe_float(item.get("timelineEnd"), start + duration))
            progress("running", "rendering", 0.12 + ((index + 1) / total) * 0.58, "Rendering timeline segments.")

        output_rel = f"assets/renders/{utc_now()[:10]}_{slugify(request.timeline_name, fallback='timeline-export', max_length=48)}_{job['id'][-8:]}.mp4"
        output_path = project_path / output_rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        progress("saving", "muxing", 0.78, "Muxing MP4 export.")
        mux_segments(ffmpeg, segments, tmp_path, output_path)

    asset = build_render_asset(
        project_id=request.project_id,
        job_id=job["id"],
        request=request,
        timeline=timeline,
        media_rel=output_rel,
        width=width,
        height=height,
        duration=round(cursor, 3),
    )
    sidecar_path = output_path.with_suffix(".sceneworks.json")
    write_json(sidecar_path, asset)
    write_json(project_path / "recipes" / f"{asset['id']}.recipe.json", asset["recipe"])
    index_asset(project_path, asset)
    return {
        "assetIds": [asset["id"]],
        "assets": [asset],
        "timelineId": request.timeline_id,
        "renderPath": output_rel,
        "adapter": "ffmpeg_timeline",
    }


def main_track_items(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    for track in timeline.get("tracks", []):
        if track.get("id") == "track_main" or track.get("kind") == "video":
            return list(track.get("items", []))
    return []


def output_dimensions(aspect_ratio: str, resolution: int) -> tuple[int, int]:
    if aspect_ratio == "9:16":
        width, height = resolution, math.ceil(resolution * 16 / 9)
    elif aspect_ratio == "1:1":
        width, height = resolution, resolution
    else:
        width, height = math.ceil(resolution * 16 / 9), resolution
    return even(width), even(height)


def even(value: float) -> int:
    parsed = int(round(value))
    return parsed if parsed % 2 == 0 else parsed + 1


def find_asset(project_path: Path, asset_id: str) -> dict[str, Any]:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is not None:
        return read_json(sidecar_path)
    raise RuntimeError(f"Timeline asset not found: {asset_id}")


def render_black_segment(ffmpeg: str, output_path: Path, duration: float, width: int, height: int, fps: int) -> None:
    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}",
            "-t",
            f"{duration:.3f}",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def render_item_segment(
    *,
    ffmpeg: str,
    project_path: Path,
    item: dict[str, Any],
    asset: dict[str, Any],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> float:
    media_path = project_path / asset.get("file", {}).get("path", "")
    if not media_path.exists():
        raise RuntimeError(f"Timeline source file is missing: {media_path}")

    source_in = safe_float(item.get("sourceIn"), 0)
    source_out = safe_float(item.get("sourceOut"), safe_float(item.get("timelineEnd"), 4))
    timeline_duration = safe_float(item.get("timelineEnd"), 4) - safe_float(item.get("timelineStart"), 0)
    source_duration = max(0.1, source_out - source_in)
    speed = max(0.1, float(item.get("speed") or 1))
    duration = max(0.1, timeline_duration or source_duration / speed)
    media_type = asset.get("type")
    mime_type = asset.get("file", {}).get("mimeType", "")
    vf = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        f"fps={fps}",
        "format=yuv420p",
    ]
    transition_in = item.get("transitionIn") or {}
    transition_out = item.get("transitionOut") or {}
    if transition_in.get("type") == "fade_from_black":
        vf.append(f"fade=t=in:st=0:d={min(duration, safe_float(transition_in.get('duration'), 0.5)):.3f}")
    if transition_out.get("type") == "fade_to_black":
        fade_duration = min(duration, safe_float(transition_out.get("duration"), 0.5))
        vf.append(f"fade=t=out:st={max(0, duration - fade_duration):.3f}:d={fade_duration:.3f}")

    if media_type != "video" and (media_type == "image" or mime_type.startswith("image/")):
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-i",
                str(media_path),
                "-t",
                f"{duration:.3f}",
                "-vf",
                ",".join(vf),
                "-an",
                str(output_path),
            ]
        )
        return duration

    setpts = f"setpts={1 / speed:.6f}*PTS"
    run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{source_in:.3f}",
            "-i",
            str(media_path),
            "-t",
            f"{source_duration:.3f}",
            "-vf",
            ",".join([setpts, *vf]),
            "-an",
            str(output_path),
        ]
    )
    return duration


def mux_segments(ffmpeg: str, segments: list[dict[str, Any]], tmp_path: Path, output_path: Path) -> None:
    crossfades = [segment for segment in segments[1:] if segment.get("transition") == "crossfade"]
    if crossfades:
        mux_with_crossfades(ffmpeg, segments, tmp_path, output_path)
        return
    list_path = tmp_path / "concat.txt"
    list_path.write_text(
        "".join(f"file '{segment['path'].as_posix()}'\n" for segment in segments),
        encoding="utf-8",
    )
    run_ffmpeg([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_path)])


def mux_with_crossfades(ffmpeg: str, segments: list[dict[str, Any]], tmp_path: Path, output_path: Path) -> None:
    current = segments[0]["path"]
    current_duration = segments[0]["duration"]
    for index, segment in enumerate(segments[1:], start=1):
        next_path = segment["path"]
        next_duration = segment["duration"]
        transition = segment.get("transition")
        duration = min(1.5, max(0.1, safe_float(segment.get("transitionDuration"), 0.5)))
        merged = tmp_path / f"xfade_{index:04d}.mp4"
        if transition == "crossfade":
            offset = max(0, current_duration - duration)
            run_ffmpeg(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(current),
                    "-i",
                    str(next_path),
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration={duration:.3f}:offset={offset:.3f},format=yuv420p[v]",
                    "-map",
                    "[v]",
                    str(merged),
                ]
            )
            current_duration = current_duration + next_duration - duration
        else:
            list_path = tmp_path / f"concat_{index:04d}.txt"
            list_path.write_text(f"file '{current.as_posix()}'\nfile '{next_path.as_posix()}'\n", encoding="utf-8")
            run_ffmpeg([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(merged)])
            current_duration += next_duration
        current = merged
    current.replace(output_path)


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if not stderr:
            raise RuntimeError("FFmpeg command failed.")
        lines = stderr.splitlines()[-10:]
        message = "\n".join(lines)
        raise RuntimeError(message[-2000:])


def build_render_asset(
    *,
    project_id: str,
    job_id: str,
    request: ExportRequest,
    timeline: dict[str, Any],
    media_rel: str,
    width: int,
    height: int,
    duration: float,
) -> dict[str, Any]:
    asset_id = f"asset_{uuid4().hex}"
    created_at = utc_now()
    source_asset_ids = [
        item["assetId"]
        for track in timeline.get("tracks", [])
        for item in track.get("items", [])
        if item.get("assetId")
    ]
    return {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": project_id,
        "generationSetId": None,
        "type": "render",
        "displayName": f"{request.timeline_name} export",
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "video/mp4",
            "width": width,
            "height": height,
            "duration": duration,
            "fps": request.fps,
        },
        "status": {
            "favorite": False,
            "rating": 0,
            "rejected": False,
            "trashed": False,
        },
        "recipe": {
            "mode": "timeline_export",
            "model": "ffmpeg",
            "adapter": "ffmpeg_timeline",
            "prompt": request.timeline_name,
            "negativePrompt": "",
            "seed": None,
            "loras": [],
            "normalizedSettings": {
                "timelineId": request.timeline_id,
                "resolution": request.resolution,
                "width": width,
                "height": height,
                "fps": request.fps,
                "aspectRatio": timeline.get("aspectRatio", "16:9"),
            },
            "rawAdapterSettings": {
                "timelinePath": request.timeline_path,
                "renderer": "ffmpeg segment concat",
            },
        },
        "lineage": {
            "parents": source_asset_ids,
            "sourceAssetId": request.timeline_id,
            "sourceTimestamp": None,
            "jobId": job_id,
        },
    }
