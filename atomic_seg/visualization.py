"""Static segmentation and complete-episode visualizations."""

from __future__ import annotations

import html
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import CAMERA_KEYS, SuccessEpisode, load_episode, video_path
from .media import materialize_camera_frames
from .segmentation import segment_episode, serializable_segmentation


ACTION_COLORS = {
    "still": "#d8dee9",
    "move": "#5e81ac",
    "turn": "#b48ead",
    "lift": "#a3be8c",
    "lower": "#ebcb8b",
    "close_gripper": "#bf616a",
    "open_gripper": "#d08770",
    "gripper_decrease": "#bf616a",
    "gripper_increase": "#d08770",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _clean_text_file(value: str) -> str:
    """Normalize generated text files and remove trailing whitespace."""

    return "\n".join(line.rstrip() for line in value.splitlines()) + "\n"


def _polyline(
    values: np.ndarray,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return ""
    maximum = float(np.quantile(array, 0.98))
    if maximum <= 1e-12:
        maximum = 1.0
    points = []
    denominator = max(len(array) - 1, 1)
    for index, value in enumerate(array):
        x = x0 + width * index / denominator
        y = y0 + height * (
            1.0 - min(max(float(value) / maximum, 0.0), 1.0)
        )
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_segmentation_svg(
    path: Path | str,
    *,
    episode: SuccessEpisode,
    result: Mapping[str, Any],
) -> None:
    """Write a self-contained SVG for one segmented episode."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width = 1440
    margin = 72
    plot_width = width - 2 * margin
    energy_top = 120
    energy_height = 220
    lane_top = 390
    lane_height = 64
    summary_top = 570
    row_height = 30
    segments = list(result["segments"])
    height = summary_top + row_height * (len(segments) + 3)
    frame_count = int(result["frame_count"])
    denominator = max(frame_count - 1, 1)

    def x_for_frame(frame: int) -> float:
        return margin + plot_width * int(frame) / denominator

    features = result["_state_features"]
    left_energy = np.asarray(
        features["arms"]["left"]["energy_smooth"],
        dtype=np.float64,
    )
    right_energy = np.asarray(
        features["arms"]["right"]["energy_smooth"],
        dtype=np.float64,
    )
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        (
            "text{font-family:Inter,Arial,sans-serif;fill:#2e3440}"
            ".title{font-size:24px;font-weight:700}"
            ".subtitle{font-size:14px;fill:#4c566a}"
            ".axis{stroke:#4c566a;stroke-width:1}"
            ".grid{stroke:#d8dee9;stroke-width:1}"
            ".boundary{stroke:#2e3440;stroke-width:1.5;stroke-dasharray:5 4}"
            ".arm-boundary{stroke-width:1;stroke-dasharray:2 3}"
            ".label{font-size:12px;font-weight:600}"
            ".small{font-size:11px;fill:#4c566a}"
            ".table{font-size:12px}"
        ),
        "</style>",
        '<rect width="100%" height="100%" fill="#eceff4"/>',
        (
            f'<text x="{margin}" y="42" class="title">'
            f'Episode {episode.episode_index}: Atomic Segmentation</text>'
        ),
        (
            f'<text x="{margin}" y="68" class="subtitle">'
            f'{_escape(episode.task)}</text>'
        ),
        (
            f'<text x="{margin}" y="92" class="subtitle">'
            f'{frame_count} frames · {len(segments)} atomic segments · '
            f'success-only demonstration</text>'
        ),
        (
            f'<rect x="{margin}" y="{energy_top}" width="{plot_width}" '
            f'height="{energy_height}" fill="#ffffff" rx="8"/>'
        ),
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin + plot_width * fraction
        parts.append(
            f'<line x1="{x:.2f}" y1="{energy_top}" x2="{x:.2f}" '
            f'y2="{energy_top + energy_height}" class="grid"/>'
        )
        frame = round((frame_count - 1) * fraction)
        parts.append(
            f'<text x="{x:.2f}" y="{energy_top + energy_height + 18}" '
            f'text-anchor="middle" class="small">{frame}</text>'
        )
    parts.extend(
        [
            (
                f'<polyline fill="none" stroke="#5e81ac" stroke-width="2" '
                f'points="{_polyline(left_energy, x0=margin, y0=energy_top, width=plot_width, height=energy_height)}"/>'
            ),
            (
                f'<polyline fill="none" stroke="#bf616a" stroke-width="2" '
                f'points="{_polyline(right_energy, x0=margin, y0=energy_top, width=plot_width, height=energy_height)}"/>'
            ),
            (
                f'<text x="{margin + 12}" y="{energy_top + 22}" '
                f'class="label" fill="#5e81ac">left energy</text>'
            ),
            (
                f'<text x="{margin + 110}" y="{energy_top + 22}" '
                f'class="label" fill="#bf616a">right energy</text>'
            ),
        ]
    )
    for boundary in result["per_arm_boundaries"]["left"]:
        x = x_for_frame(int(boundary))
        parts.append(
            f'<line x1="{x:.2f}" y1="{energy_top}" x2="{x:.2f}" '
            f'y2="{energy_top + energy_height}" '
            f'class="arm-boundary" stroke="#5e81ac"/>'
        )
    for boundary in result["per_arm_boundaries"]["right"]:
        x = x_for_frame(int(boundary))
        parts.append(
            f'<line x1="{x:.2f}" y1="{energy_top}" x2="{x:.2f}" '
            f'y2="{energy_top + energy_height}" '
            f'class="arm-boundary" stroke="#bf616a"/>'
        )
    for boundary in result["boundaries"]:
        x = x_for_frame(int(boundary))
        parts.append(
            f'<line x1="{x:.2f}" y1="{energy_top}" x2="{x:.2f}" '
            f'y2="{lane_top + 2 * lane_height + 12}" class="boundary"/>'
        )
    for lane_index, arm in enumerate(("left", "right")):
        y = lane_top + lane_index * lane_height
        parts.append(
            f'<text x="{margin - 12}" y="{y + 36}" '
            f'text-anchor="end" class="label">{arm}</text>'
        )
        for segment in segments:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            action = str(segment[f"{arm}_action"])
            x = x_for_frame(start)
            x_end = x_for_frame(end)
            block_width = max(2.0, x_end - x)
            color = ACTION_COLORS.get(action, "#88c0d0")
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{block_width:.2f}" '
                f'height="{lane_height - 8}" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1" rx="4"/>'
            )
            if block_width > 54:
                parts.append(
                    f'<text x="{x + block_width / 2:.2f}" y="{y + 32}" '
                    f'text-anchor="middle" class="label">'
                    f'{_escape(action)}</text>'
                )
    parts.extend(
        [
            (
                f'<text x="{margin}" y="{summary_top - 20}" class="label">'
                "Atomic segment table</text>"
            ),
            (
                f'<text x="{margin}" y="{summary_top}" class="table">'
                "#</text>"
            ),
            (
                f'<text x="{margin + 55}" y="{summary_top}" class="table">'
                "frames</text>"
            ),
            (
                f'<text x="{margin + 190}" y="{summary_top}" class="table">'
                "left</text>"
            ),
            (
                f'<text x="{margin + 360}" y="{summary_top}" class="table">'
                "right</text>"
            ),
            (
                f'<text x="{margin + 530}" y="{summary_top}" class="table">'
                "coordination</text>"
            ),
        ]
    )
    for index, segment in enumerate(segments):
        y = summary_top + (index + 1) * row_height
        if index % 2 == 0:
            parts.append(
                f'<rect x="{margin - 8}" y="{y - 20}" '
                f'width="{plot_width + 16}" height="{row_height}" '
                f'fill="#e5e9f0"/>'
            )
        values = (
            str(index),
            (
                f"{segment['start_frame']}–"
                f"{segment['end_frame']}"
            ),
            str(segment["left_action"]),
            str(segment["right_action"]),
            str(segment["coordination_relation"]),
        )
        positions = (
            margin,
            margin + 55,
            margin + 190,
            margin + 360,
            margin + 530,
        )
        for x, value in zip(positions, values):
            parts.append(
                f'<text x="{x}" y="{y}" class="table">'
                f'{_escape(value)}</text>'
            )
    parts.append("</svg>")
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _current_images(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    output = []
    for role, path in zip(row["image_roles"], row["images"]):
        if str(role).startswith("current_"):
            output.append((str(role), str(path)))
    return output


def _reasoning_from_row(row: Mapping[str, Any]) -> str:
    value = str(row["conversations"][1]["value"])
    start = value.find("<think>")
    end = value.find("</think>")
    if start < 0 or end < 0:
        return ""
    return value[start + len("<think>") : end].strip()


def write_episode_html(
    path: Path | str,
    *,
    output_root: Path | str,
    action_episode: Mapping[str, Any],
    cot_rows: Sequence[Mapping[str, Any]],
    videos: Mapping[str, Path],
) -> None:
    """Write a complete three-camera, segment-by-segment episode page."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(output_root)
    rows_by_segment = {
        int(row["segment_index"]): row for row in cot_rows
    }
    media_dir = (
        destination.parent
        / "media"
        / f"episode_{int(action_episode['episode_index']):06d}"
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    local_videos: dict[str, str] = {}
    for camera, source in videos.items():
        target = media_dir / f"{camera}.mp4"
        shutil.copy2(source, target)
        local_videos[camera] = target.relative_to(
            destination.parent
        ).as_posix()
    cards = []
    for segment in action_episode["segments"]:
        index = int(segment["segment_index"])
        row = rows_by_segment[index]
        images = "".join(
            (
                '<figure><img loading="lazy" '
                f'src="../{_escape(image_path)}" '
                f'alt="{_escape(role)}">'
                f"<figcaption>{_escape(role)}</figcaption></figure>"
            )
            for role, image_path in _current_images(row)
        )
        guide_action = json.dumps(
            segment["guide_action"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cards.append(
            f"""
            <section class="segment" id="segment-{index}">
              <header>
                <h2>Segment {index}: frames
                  {segment['start_frame']}–{segment['end_frame']}</h2>
                <span class="badge">{_escape(segment['primary_action_verb'])}</span>
              </header>
              <div class="frames">{images}</div>
              <div class="columns">
                <div>
                  <h3>Action-only supervision</h3>
                  <p>{_escape(segment['sub_task'])}</p>
                  <dl>
                    <dt>coordination</dt>
                    <dd>{_escape(segment['coordination_relation'])}</dd>
                    <dt>num_chunks</dt>
                    <dd>{segment['num_chunks']}</dd>
                    <dt>guide_action</dt>
                    <dd><code>{_escape(guide_action)}</code></dd>
                  </dl>
                </div>
                <div>
                  <h3>Embedded pre-action CoT</h3>
                  <p>{_escape(_reasoning_from_row(row))}</p>
                </div>
              </div>
            </section>
            """
        )
    video_blocks = "".join(
        f"""
        <figure>
          <video controls preload="metadata" src="{_escape(path_value)}"></video>
          <figcaption>{_escape(camera)}</figcaption>
        </figure>
        """
        for camera, path_value in local_videos.items()
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Episode {action_episode['episode_index']} complete visualization</title>
  <style>
    :root {{ color-scheme: light; --bg:#eceff4; --panel:#fff;
      --ink:#2e3440; --muted:#4c566a; --blue:#5e81ac; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:32px; background:var(--bg);
      color:var(--ink); font:15px/1.5 Inter,Arial,sans-serif; }}
    main {{ max-width:1500px; margin:auto; }}
    h1 {{ margin-bottom:4px; }}
    .meta {{ color:var(--muted); margin-top:0; }}
    .videos,.frames {{ display:grid; grid-template-columns:repeat(3,1fr);
      gap:12px; }}
    figure {{ margin:0; }}
    video,img {{ width:100%; border-radius:8px; background:#2e3440; }}
    figcaption {{ margin-top:4px; color:var(--muted); font-size:12px; }}
    .segment {{ background:var(--panel); border-radius:12px;
      padding:20px; margin:20px 0; box-shadow:0 2px 10px #2e344020; }}
    .segment header {{ display:flex; justify-content:space-between;
      gap:16px; align-items:center; }}
    .badge {{ background:var(--blue); color:white; border-radius:999px;
      padding:5px 10px; white-space:nowrap; }}
    .columns {{ display:grid; grid-template-columns:1fr 1fr; gap:24px;
      margin-top:16px; }}
    dl {{ display:grid; grid-template-columns:130px 1fr; gap:6px 12px; }}
    dt {{ color:var(--muted); }}
    dd {{ margin:0; overflow-wrap:anywhere; }}
    code {{ font-size:11px; }}
    @media(max-width:900px) {{
      .videos,.frames,.columns {{ grid-template-columns:1fr; }}
      .segment header {{ align-items:flex-start; flex-direction:column; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Episode {action_episode['episode_index']}: Complete Successful Episode</h1>
  <p class="meta">{_escape(action_episode['task'])}</p>
  <p class="meta">{action_episode['frame_count']} frames ·
    {len(action_episode['segments'])} atomic segments ·
    all three complete camera streams</p>
  <section class="videos">{video_blocks}</section>
  {''.join(cards)}
</main>
</body>
</html>
"""
    destination.write_text(_clean_text_file(content), encoding="utf-8")


def build_episode_visual_assets(
    *,
    data_root: Path | str,
    output_root: Path | str,
    episode: SuccessEpisode,
    action_episode: Mapping[str, Any],
    cot_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Materialize one SVG and one complete episode HTML page."""

    root = Path(output_root)
    source = load_episode(data_root, episode.episode_index)
    result = segment_episode(
        np.asarray(source["observation.state"], dtype=np.float64),
        np.asarray(source["timestamp"], dtype=np.float64),
    )
    segmentation_path = (
        root
        / "segmentation"
        / f"episode_{episode.episode_index:06d}.svg"
    )
    write_segmentation_svg(
        segmentation_path,
        episode=episode,
        result=result,
    )
    segmentation_json = segmentation_path.with_suffix(".json")
    segmentation_json.write_text(
        json.dumps(
            {
                **serializable_segmentation(result),
                "episode_index": episode.episode_index,
                "task": episode.task,
                "success": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    episode_path = (
        root
        / "episodes"
        / f"episode_{episode.episode_index:06d}.html"
    )
    videos = {
        camera: video_path(data_root, episode.episode_index, camera)
        for camera in CAMERA_KEYS
    }
    write_episode_html(
        episode_path,
        output_root=root,
        action_episode=action_episode,
        cot_rows=cot_rows,
        videos=videos,
    )
    return {
        "segmentation_svg": segmentation_path.relative_to(root).as_posix(),
        "segmentation_json": segmentation_json.relative_to(root).as_posix(),
        "episode_html": episode_path.relative_to(root).as_posix(),
    }


def write_visualization_index(
    path: Path | str,
    *,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        f"""
        <article>
          <h2>Episode {entry['episode_index']}</h2>
          <p>{_escape(entry['task'])}</p>
          <a href="{_escape(entry['segmentation_svg'])}">
            Segmentation visualization</a>
          <a href="{_escape(entry['episode_html'])}">
            Complete episode visualization</a>
        </article>
        """
        for entry in entries
    )
    destination.write_text(
        _clean_text_file(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atomic episode visualization examples</title>
  <style>
    body {{ max-width:1000px; margin:40px auto; padding:0 20px;
      background:#eceff4; color:#2e3440; font:16px/1.5 Inter,Arial,sans-serif; }}
    article {{ background:white; padding:22px; margin:16px 0;
      border-radius:12px; }}
    a {{ display:inline-block; margin-right:20px; color:#5e81ac; }}
  </style>
</head>
<body>
  <h1>Successful episode visualization examples</h1>
  <p>Three segmentation visualizations and three complete episode
  visualizations. No recovery/failure episode is included.</p>
  {cards}
</body>
</html>
"""
        ),
        encoding="utf-8",
    )


def package_visualizations(
    source_root: Path | str,
    zip_path: Path | str,
) -> dict[str, int | str]:
    """Create and verify a portable archive of all visualization assets."""

    source = Path(source_root)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    Path("visualizations") / path.relative_to(source),
                )
    with zipfile.ZipFile(destination) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"corrupt ZIP member: {corrupt}")
        names = archive.namelist()
    counts = {
        "segmentation_svg_count": sum(
            name.startswith("visualizations/segmentation/")
            and name.endswith(".svg")
            for name in names
        ),
        "complete_episode_html_count": sum(
            name.startswith("visualizations/episodes/episode_")
            and name.endswith(".html")
            for name in names
        ),
        "complete_camera_video_count": sum(
            name.endswith(".mp4") for name in names
        ),
    }
    if counts != {
        "segmentation_svg_count": 3,
        "complete_episode_html_count": 3,
        "complete_camera_video_count": 9,
    }:
        raise ValueError(f"unexpected visualization ZIP contents: {counts}")
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "file_count": len(names),
        **counts,
    }
