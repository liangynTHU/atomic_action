#!/usr/bin/env python3
"""Build complete failure-to-recovery episode segmentation review pages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from atomic_seg.io import load_json, save_json
from atomic_seg.media import materialize_camera_frames
from atomic_seg.recovery_cot.io import recovery_state_layout
from atomic_seg.recovery_cot.segmentation import (
    RecoveryProposalConfig,
    propose_recovery_segments,
)

CAMERAS = (
    "cam_high",
    "cam_front",
    "cam_left_wrist",
    "cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, default=None)
    return parser.parse_args()


from atomic_seg.common.rendering import escape as _escape


def _read_jsonl_first(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: expected JSON object")
                return value
    raise ValueError(f"{path}: empty JSONL")


def _one(root: Path, pattern: str) -> Path:
    paths = [path for path in root.glob(pattern) if path.name != "manifest.json"]
    if len(paths) != 1:
        raise ValueError(f"{root}: expected one {pattern}, got {paths}")
    return paths[0]


def _runs(values: Sequence[Any]) -> list[dict[str, Any]]:
    if not values:
        return []
    output: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            output.append(
                {
                    "start_frame": start,
                    "end_frame": index - 1,
                    "label": str(values[start]),
                }
            )
            start = index
    return output


def _video_path(root: Path, camera: str) -> Path:
    path = (
        root
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera}"
        / "episode_000000.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _mapped_frame(
    source_ids: Sequence[Any],
    source_frames: Sequence[Any],
    episode_id: str,
    frame: int,
) -> int:
    matches = [
        index
        for index, (value_id, value_frame) in enumerate(zip(source_ids, source_frames))
        if str(value_id) == episode_id and int(value_frame) == int(frame)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one mapping for {episode_id}:{frame}, got {matches}"
        )
    return matches[0]


def _category(
    start: int,
    end: int,
    attempt_start: int,
    attempt_end: int,
) -> str:
    if end < attempt_start:
        return "source_prefix"
    if start <= attempt_end and end >= attempt_start:
        return "failed_attempt"
    return "failure_observation"


def _semantic_segments(
    *,
    source: Mapping[str, Any],
    recovery: Mapping[str, Any],
    annotation: Mapping[str, Any],
    source_ids: Sequence[Any],
    source_frames: Sequence[Any],
    phase_runs: Sequence[Mapping[str, Any]],
    states: np.ndarray,
    timestamps: np.ndarray,
    state_layout: Sequence[str],
) -> list[dict[str, Any]]:
    source_id = Path(str(recovery["source_episode_dir"])).name
    recovery_id = str(recovery["episode_id"])
    attempt_start = int(annotation["attempt_start_step"])
    attempt_end = int(annotation["attempt_end_step"])
    output: list[dict[str, Any]] = []
    raw_source = source.get("semantic_segments")
    raw_recovery = recovery.get("semantic_segments")
    failure_run = next(
        value for value in phase_runs if value["label"] == "failure_outcome"
    )
    attempt_start_global = _mapped_frame(
        source_ids,
        source_frames,
        source_id,
        attempt_start,
    )
    attempt_end_global = _mapped_frame(
        source_ids,
        source_frames,
        source_id,
        attempt_end,
    )
    proposal_config = RecoveryProposalConfig(
        min_segment_seconds=1.0,
        max_segment_seconds=8.0,
        smoothing_seconds=0.4,
        direction_window_seconds=0.6,
        direction_cosine_threshold=-0.35,
        pause_energy_quantile=0.20,
    )
    if isinstance(raw_source, list) and raw_source:
        for index, segment in enumerate(raw_source):
            native_start = int(segment["start_step"])
            native_end = int(segment["end_step"])
            start = _mapped_frame(
                source_ids,
                source_frames,
                source_id,
                native_start,
            )
            end = _mapped_frame(
                source_ids,
                source_frames,
                source_id,
                native_end,
            )
            output.append(
                {
                    "segment_id": f"S{index}",
                    "start_frame": start,
                    "end_frame": end,
                    "native_start_frame": native_start,
                    "native_end_frame": native_end,
                    "label": str(segment["label"]),
                    "category": _category(
                        native_start,
                        native_end,
                        attempt_start,
                        attempt_end,
                    ),
                    "boundary_source": "source_semantic_segments",
                    "outcome_judgment": (
                        "Reference failure is observed only after the completed "
                        "attempt; no model judgment is generated here."
                    ),
                }
            )
    else:
        source_proposals = propose_recovery_segments(
            states,
            timestamps,
            state_layout,
            start_frame=0,
            end_frame=int(failure_run["start_frame"]) - 1,
            config=proposal_config,
        )
        for index, segment in enumerate(source_proposals["segments"]):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            output.append(
                {
                    "segment_id": f"PS{index}",
                    "start_frame": start,
                    "end_frame": end,
                    "native_start_frame": int(source_frames[start]),
                    "native_end_frame": int(source_frames[end]),
                    "label": f"REVIEW_REQUIRED_source_motion_{index:02d}",
                    "category": _category(
                        start,
                        end,
                        attempt_start_global,
                        attempt_end_global,
                    ),
                    "boundary_source": "generic_state_motion_proposal",
                    "outcome_judgment": (
                        "UNDECIDED: automatic motion proposal; semantic subtask "
                        "and outcome require review."
                    ),
                    "proposal_diagnostics": segment.get("start_boundary_evidence"),
                }
            )
    output.append(
        {
            "segment_id": "F",
            "start_frame": int(failure_run["start_frame"]),
            "end_frame": int(failure_run["end_frame"]),
            "native_start_frame": int(annotation["failure_outcome_step"]),
            "native_end_frame": int(annotation["failure_outcome_step"]),
            "label": "audited_failure_outcome",
            "category": "failure_outcome",
            "boundary_source": "paired_failure_outcome",
            "outcome_judgment": "Audited reference: previous attempt failed.",
        }
    )
    recovery_run = next(
        value for value in phase_runs if value["label"] == "recovery_outcome"
    )
    recovery_phase = next(value for value in phase_runs if value["label"] == "recovery")
    if isinstance(raw_recovery, list) and raw_recovery:
        for index, segment in enumerate(raw_recovery):
            native_start = int(segment["start_step"])
            native_end = int(segment["end_step"])
            start = _mapped_frame(
                source_ids,
                source_frames,
                recovery_id,
                native_start,
            )
            end = _mapped_frame(
                source_ids,
                source_frames,
                recovery_id,
                native_end,
            )
            output.append(
                {
                    "segment_id": f"R{index}",
                    "start_frame": start,
                    "end_frame": end,
                    "native_start_frame": native_start,
                    "native_end_frame": native_end,
                    "label": str(segment["label"]),
                    "category": "recovery",
                    "boundary_source": "recovery_semantic_segments",
                    "outcome_judgment": (
                        "UNDECIDED: the future model should infer this segment's "
                        "outcome at the next decision boundary."
                    ),
                }
            )
    else:
        recovery_proposals = propose_recovery_segments(
            states,
            timestamps,
            state_layout,
            start_frame=int(recovery_phase["start_frame"]),
            end_frame=int(recovery_phase["end_frame"]),
            config=proposal_config,
        )
        for index, segment in enumerate(recovery_proposals["segments"]):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            output.append(
                {
                    "segment_id": f"PR{index}",
                    "start_frame": start,
                    "end_frame": end,
                    "native_start_frame": int(source_frames[start]),
                    "native_end_frame": int(source_frames[end]),
                    "label": f"REVIEW_REQUIRED_recovery_motion_{index:02d}",
                    "category": "recovery",
                    "boundary_source": "generic_state_motion_proposal",
                    "outcome_judgment": (
                        "UNDECIDED: automatic motion proposal; semantic subtask "
                        "and outcome require review."
                    ),
                    "proposal_diagnostics": segment.get("start_boundary_evidence"),
                }
            )
    output.append(
        {
            "segment_id": "O",
            "start_frame": int(recovery_run["start_frame"]),
            "end_frame": int(recovery_run["end_frame"]),
            "native_start_frame": int(recovery["num_frames"]),
            "native_end_frame": int(recovery["num_frames"]),
            "label": "audited_recovery_outcome",
            "category": "recovery_outcome",
            "boundary_source": "paired_recovery_outcome",
            "outcome_judgment": "Audited reference: original task succeeded.",
        }
    )
    return sorted(output, key=lambda value: int(value["start_frame"]))


def _timeline(
    segments: Sequence[Mapping[str, Any]],
    total_frames: int,
    *,
    use_category: bool,
) -> str:
    blocks = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        width = 100.0 * (end - start + 1) / total_frames
        label = str(segment.get("segment_id") or segment["label"])
        category = (
            str(segment["category"]) if use_category else f"phase_{segment['label']}"
        )
        proposal_class = (
            " proposal"
            if segment.get("boundary_source") == "generic_state_motion_proposal"
            else ""
        )
        title = f"{label}: frames {start}-{end}; {segment.get('label', '')}"
        blocks.append(
            f'<button class="timeline-block {_escape(category)}{proposal_class}" '
            f'style="width:{width:.6f}%" onclick="seekAll({start})" '
            f'title="{_escape(title)}">{_escape(label)}</button>'
        )
    return "".join(blocks)


def _boundary_cards(
    *,
    segments: Sequence[Mapping[str, Any]],
    assets: Mapping[tuple[str, int], Path],
    page_dir: Path,
    fps: float,
) -> str:
    cards = []
    for previous, current in zip(segments[:-1], segments[1:]):
        frame = int(current["start_frame"])
        before = max(0, frame - 1)
        images = []
        for camera in CAMERAS:
            for image_frame, side in ((before, "before"), (frame, "after")):
                path = assets[(camera, image_frame)]
                relative = path.relative_to(page_dir).as_posix()
                images.append(
                    "<figure>"
                    f'<img src="{_escape(relative)}" loading="lazy">'
                    f"<figcaption>{_escape(camera)} · {side} · "
                    f"f{image_frame}</figcaption></figure>"
                )
        cards.append(
            f"""
            <article class="boundary-card">
              <header>
                <div><h3>{_escape(previous["segment_id"])} →
                  {_escape(current["segment_id"])}</h3>
                  <p>{_escape(previous["label"])} →
                    {_escape(current["label"])}</p></div>
                <button onclick="seekAll({frame})">Seek frame {frame}
                  ({frame / fps:.2f}s)</button>
              </header>
              <div class="boundary-images">{"".join(images)}</div>
            </article>
            """
        )
    return "".join(cards)


def _episode_page(
    *,
    output_root: Path,
    slug: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    case_root = Path(entry["case_root"]).resolve()
    paired = case_root / "lerobot_failure_then_recovery"
    info = load_json(paired / "meta" / "info.json")
    episode_meta = _read_jsonl_first(paired / "meta" / "episodes.jsonl")
    source_path = _one(case_root, "raw_rollouts/**/episode.json")
    recovery_path = _one(case_root, "recovery_rollouts/**/episode.json")
    scenario_path = _one(case_root, "scenarios/**/*.json")
    source = load_json(source_path)
    recovery = load_json(recovery_path)
    scenario = load_json(scenario_path)
    annotation = scenario["annotation"]
    table = pq.read_table(
        paired / "data" / "chunk-000" / "episode_000000.parquet",
        columns=[
            "phase",
            "source_episode_id",
            "source_frame_index",
            "task_success",
            "observation.state",
            "timestamp",
        ],
    )
    phases = table.column("phase").to_pylist()
    source_ids = table.column("source_episode_id").to_pylist()
    source_frames = table.column("source_frame_index").to_pylist()
    task_success = table.column("task_success").to_pylist()
    states = np.asarray(table.column("observation.state").to_pylist())
    timestamps = np.asarray(table.column("timestamp").to_pylist())
    phase_runs = _runs(phases)
    segments = _semantic_segments(
        source=source,
        recovery=recovery,
        annotation=annotation,
        source_ids=source_ids,
        source_frames=source_frames,
        phase_runs=phase_runs,
        states=states,
        timestamps=timestamps,
        state_layout=recovery_state_layout(paired),
    )
    total_frames = len(phases)
    fps = float(info["fps"])
    page_dir = output_root / "episodes" / slug
    video_dir = page_dir / "videos"
    keyframe_dir = page_dir / "keyframes"
    metadata_dir = page_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_videos: dict[str, Path] = {}
    for camera in CAMERAS:
        source_video = _video_path(paired, camera)
        destination = video_dir / f"{camera}.mp4"
        shutil.copy2(source_video, destination)
        source_videos[camera] = destination
    boundary_frames = sorted(
        {
            value
            for segment in segments[1:]
            for value in (
                max(0, int(segment["start_frame"]) - 1),
                int(segment["start_frame"]),
            )
        }
    )
    assets = materialize_camera_frames(
        source_videos,
        {camera: boundary_frames for camera in CAMERAS},
        keyframe_dir,
    )
    shutil.copy2(source_path, metadata_dir / "source_episode.json")
    shutil.copy2(recovery_path, metadata_dir / "recovery_episode.json")
    shutil.copy2(scenario_path, metadata_dir / "scenario.json")
    review = {
        "schema": "complete_recovery_episode_review",
        "case": slug,
        "task": episode_meta["tasks"][0],
        "fps": fps,
        "frame_count": total_frames,
        "duration_seconds": total_frames / fps,
        "failure_type": annotation["failure_type"],
        "failure_summary": annotation["summary"],
        "failure_evidence": annotation["evidence"],
        "recovery_level": recovery["recovery_level"],
        "recovery_level_name": recovery["recovery_level_name"],
        "phase_runs": phase_runs,
        "semantic_segments": segments,
        "contains_automatic_proposals": any(
            segment["boundary_source"] == "generic_state_motion_proposal"
            for segment in segments
        ),
        "first_success_frame": next(
            (index for index, value in enumerate(task_success) if value),
            None,
        ),
        "outcome_judgment_policy": (
            "No model outcome judgment is generated in this boundary-review "
            "bundle. Recovery transition outcomes are intentionally left "
            "undecided pending the next pipeline revision."
        ),
    }
    save_json(metadata_dir / "episode_review.json", review)

    videos = "".join(
        f"""
        <figure class="video-panel">
          <video controls preload="metadata" data-camera="{_escape(camera)}"
            src="videos/{_escape(camera)}.mp4"></video>
          <figcaption>{_escape(camera)}</figcaption>
        </figure>
        """
        for camera in CAMERAS
    )
    rows = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        rows.append(
            "<tr>"
            f"<td>{_escape(segment['segment_id'])}</td>"
            f"<td>{start}–{end}</td>"
            f"<td>{start / fps:.2f}–{(end + 1) / fps:.2f}s</td>"
            f"<td>{_escape(segment['category'])}</td>"
            f"<td>{_escape(segment['label'])}</td>"
            f"<td>{_escape(segment['boundary_source'])}</td>"
            f"<td>{_escape(segment['outcome_judgment'])}</td>"
            f'<td><button onclick="seekAll({start})">seek</button></td>'
            "</tr>"
        )
    evidence = "".join(f"<li>{_escape(value)}</li>" for value in annotation["evidence"])
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(slug)} complete recovery episode</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ margin:0; padding:22px; background:#0b1020; color:#eef2ff; }}
    main {{ max-width:1700px; margin:auto; }}
    a {{ color:#93c5fd; }}
    section, article {{ background:#151c2f; border:1px solid #334155;
      border-radius:12px; padding:16px; margin:16px 0; }}
    h1,h2,h3 {{ margin-top:0; }}
    .warning {{ border-color:#f59e0b; background:#2a2111; }}
    .videos {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
      gap:12px; }}
    .video-panel {{ margin:0; }}
    video {{ width:100%; background:#000; border-radius:8px; }}
    figcaption {{ color:#cbd5e1; margin-top:5px; }}
    button {{ cursor:pointer; border:0; border-radius:6px; padding:7px 10px;
      background:#334155; color:#fff; }}
    .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; }}
    .timeline {{ display:flex; width:100%; min-height:48px; margin:8px 0;
      overflow:hidden; border-radius:8px; background:#111827; }}
    .timeline-block {{ min-width:2px; border-radius:0; padding:4px 2px;
      overflow:hidden; font-size:11px; border-right:1px solid #0f172a; }}
    .source_prefix,.phase_rollout,.phase_pre_failure {{ background:#2563eb; }}
    .failed_attempt,.phase_attempt,.phase_failure {{ background:#dc2626; }}
    .failure_observation,.phase_failure_observation,.phase_post_failure
      {{ background:#b45309; }}
    .failure_outcome,.phase_failure_outcome {{ background:#7f1d1d; }}
    .recovery,.phase_recovery {{ background:#15803d; }}
    .recovery_outcome,.phase_recovery_outcome {{ background:#047857; }}
    .proposal {{ background:#7e22ce !important; border:2px dashed #e9d5ff;
      color:#fff; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border:1px solid #334155; padding:8px; vertical-align:top; }}
    th {{ background:#1e293b; text-align:left; }}
    .boundary-card header {{ display:flex; justify-content:space-between;
      gap:10px; align-items:flex-start; }}
    .boundary-images {{ display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }}
    .boundary-images figure {{ margin:0; background:#0f172a; padding:6px;
      border-radius:7px; }}
    .boundary-images img {{ width:100%; display:block; border-radius:4px; }}
    .boundary-images figcaption {{ font-size:11px; }}
    .meta {{ color:#cbd5e1; }}
    @media(max-width:950px) {{ .videos,.boundary-images
      {{ grid-template-columns:1fr; }} table {{ font-size:11px; }} }}
  </style>
</head>
<body><main>
  <p><a href="../../index.html">← all episodes</a></p>
  <h1>{_escape(entry.get("title", slug))}</h1>
  <p>{_escape(episode_meta["tasks"][0])}</p>
  <p class="meta">{total_frames} frames · {fps:g} FPS ·
    {total_frames / fps:.2f}s · L{_escape(recovery["recovery_level"])}
    {_escape(recovery["recovery_level_name"])}</p>
  <section class="warning">
    <h2>Boundary review only</h2>
    <p>This page shows complete videos and existing semantic boundaries.
      It does not generate <code>The previous atomic subtask was completed
      successfully.</code> judgments. Recovery transition outcomes remain
      <strong>UNDECIDED</strong> until the model-based judge is redesigned.</p>
    <p>Purple dashed blocks are automatic motion proposals and are not
      semantic training labels.</p>
  </section>
  <section>
    <h2>Complete synchronized episode videos</h2>
    <div class="controls"><button onclick="playAll()">Play all</button>
      <button onclick="pauseAll()">Pause all</button>
      <button onclick="seekAll(0)">Start</button></div>
    <div class="videos">{videos}</div>
  </section>
  <section>
    <h2>Whole-episode timelines</h2>
    <p>Phase timeline</p>
    <div class="timeline">{
        _timeline(phase_runs, total_frames, use_category=False)
    }</div>
    <p>Semantic segmentation timeline</p>
    <div class="timeline">{_timeline(segments, total_frames, use_category=True)}</div>
  </section>
  <section>
    <h2>Failure reference annotation</h2>
    <p><strong>{_escape(annotation["failure_type"])}:</strong>
      {_escape(annotation["summary"])}</p>
    <ul>{evidence}</ul>
  </section>
  <section>
    <h2>Segment table</h2>
    <table><thead><tr><th>ID</th><th>Global frames</th><th>Time</th>
      <th>Category</th><th>Label</th><th>Boundary source</th>
      <th>Outcome judgment</th><th>Video</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table>
  </section>
  <section>
    <h2>Boundary before/after inspection</h2>
    {_boundary_cards(segments=segments, assets=assets, page_dir=page_dir, fps=fps)}
  </section>
  <script>
    const FPS = {fps!r};
    const videos = Array.from(document.querySelectorAll('video'));
    function seekAll(frame) {{
      const time = frame / FPS;
      videos.forEach(video => {{ video.currentTime = time; }});
      window.scrollTo({{top:0, behavior:'smooth'}});
    }}
    function playAll() {{ videos.forEach(video => video.play()); }}
    function pauseAll() {{ videos.forEach(video => video.pause()); }}
  </script>
</main></body></html>
"""
    (page_dir / "index.html").write_text(page, encoding="utf-8")
    return {
        "slug": slug,
        "title": str(entry.get("title", slug)),
        "task": str(episode_meta["tasks"][0]),
        "failure_type": str(annotation["failure_type"]),
        "recovery_level": int(recovery["recovery_level"]),
        "frame_count": total_frames,
        "duration_seconds": total_frames / fps,
        "semantic_segment_count": len(segments),
        "page": f"episodes/{slug}/index.html",
        "video_count": len(CAMERAS),
        "boundary_count": len(segments) - 1,
        "contains_automatic_proposals": review["contains_automatic_proposals"],
    }


def _write_index(output_root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    cards = "".join(
        f"""
        <article><h2>{_escape(entry["title"])}</h2>
          <p>{_escape(entry["task"])}</p>
          <p>L{entry["recovery_level"]} · {_escape(entry["failure_type"])} ·
            {entry["frame_count"]} frames · {entry["duration_seconds"]:.2f}s ·
            {entry["semantic_segment_count"]} review segments</p>
          <a href="{_escape(entry["page"])}">Open complete episode review →</a>
        </article>
        """
        for entry in entries
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complete recovery episode segmentation review</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif}}
body{{margin:0;padding:24px;background:#0b1020;color:#eef2ff}}
main{{max-width:1100px;margin:auto}}article{{background:#151c2f;border:1px solid
#334155;border-radius:12px;padding:18px;margin:14px 0}}a{{color:#93c5fd}}
.warning{{background:#2a2111;border:1px solid #f59e0b;border-radius:10px;
padding:14px}}</style></head><body><main>
<h1>Complete recovery episode segmentation review</h1>
<div class="warning">These pages are for reviewing complete episode
boundaries. They intentionally do not treat every previous recovery subtask as
successful and do not contain model-generated outcome judgments.</div>
{cards}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def _package(root: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(destination) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt archive member: {corrupt}")
        file_count = len(
            [name for name in archive.namelist() if not name.endswith("/")]
        )
    return {
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "file_count": file_count,
    }


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    if manifest.get("schema") != "recovery_episode_review_manifest":
        raise ValueError("unexpected recovery episode review manifest schema")
    raw_entries = manifest.get("episodes")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("review manifest requires episodes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"episodes[{index}] must be an object")
        slug = str(entry.get("slug", f"episode_{index:02d}"))
        result = _episode_page(
            output_root=args.output_dir,
            slug=slug,
            entry=entry,
        )
        entries.append(result)
        print(
            f"episode={slug} frames={result['frame_count']} "
            f"segments={result['semantic_segment_count']}"
        )
    _write_index(args.output_dir, entries)
    save_json(
        args.output_dir / "manifest.json",
        {
            "schema": "complete_recovery_episode_review_bundle",
            "model_outcome_judgments_generated": False,
            "episode_count": len(entries),
            "entries": entries,
        },
    )
    zip_path = args.zip_path or args.output_dir.with_suffix(".zip")
    package = _package(args.output_dir, zip_path)
    print(
        f"zip={package['path']} files={package['file_count']} bytes={package['bytes']}"
    )


if __name__ == "__main__":
    main()
