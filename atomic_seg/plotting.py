"""Dependency-light SVG diagnostic plots."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .features import arm_components


COLORS = {
    "x": "#2563eb",
    "y": "#16a34a",
    "z": "#dc2626",
    "gripper": "#9333ea",
    "energy": "#ea580c",
    "boundary": "#111827",
    "strong_boundary": "#b91c1c",
    "soft_boundary": "#7c3aed",
    "onset_boundary": "#0369a1",
    "reference": "#0f766e",
    "left": "#2563eb",
    "right": "#f97316",
}

PHASE_COLORS = {
    "still": "#e5e7eb",
    "move": "#93c5fd",
    "turn": "#c4b5fd",
    "lift": "#86efac",
    "lower": "#fca5a5",
    "grasp": "#f9a8d4",
    "close_gripper": "#f9a8d4",
    "open_gripper": "#fde68a",
    "gripper_decrease": "#f9a8d4",
    "gripper_increase": "#fde68a",
}


def _polyline(
    values: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
) -> str:
    if len(values) == 1:
        points = [(x0, y0 + height / 2.0)]
    else:
        span = max(maximum - minimum, 1e-12)
        points = [
            (
                x0 + width * index / (len(values) - 1),
                y0 + height * (1.0 - (float(value) - minimum) / span),
            )
            for index, value in enumerate(values)
        ]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_diagnostic_svg(
    path: Path | str,
    trajectory: np.ndarray,
    segments: Sequence[Dict[str, object]],
    active_arm: str,
    energy: np.ndarray,
    title: str,
    reference_segments: Sequence[Dict[str, object]] | None = None,
    boundary_evidence: Sequence[Dict[str, object]] | None = None,
    feature_bundle: Dict[str, object] | None = None,
    selection: Dict[str, object] | None = None,
    timestamps: np.ndarray | None = None,
) -> None:
    """Write an auditable dual-arm state plot.

    The diagnostic deliberately visualizes only state-derived signals.  The
    boundary labels (B1, B2, ...) map directly to ``boundary_evidence`` in the
    JSON, which is where the full provenance and onset-refinement statistics
    live.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 1500
    height = 1030
    left = 110
    plot_width = 1320
    panels = [
        ("Left xyz", 90, 145),
        ("Right xyz", 265, 145),
        ("Gripper raw value", 440, 100),
        ("State motion energy", 570, 100),
        ("Per-arm phase lane", 700, 90),
    ]
    left_position, _, left_gripper = arm_components(trajectory, "left")
    right_position, _, right_gripper = arm_components(trajectory, "right")
    frame_count = len(trajectory)
    if timestamps is not None and len(timestamps) > 1:
        fps = 1.0 / float(np.median(np.diff(timestamps)))
        clock_text = f"; estimated fps: {fps:.3f}"
    else:
        clock_text = ""
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="35" font-size="22" font-family="sans-serif">{escape(title)}</text>',
        (
            f'<text x="40" y="62" font-size="14" font-family="sans-serif">'
            f'legacy active-arm diagnostic: {escape(active_arm)}; '
            f'frames: {frame_count}{clock_text}; plotted decision inputs: state + timestamp'
            "</text>"
        ),
    ]
    for label, top, panel_height in panels:
        pieces.append(
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" '
            'fill="#fafafa" stroke="#d1d5db"/>'
        )
        pieces.append(
            f'<text x="15" y="{top + 20}" font-size="14" font-family="sans-serif">{label}</text>'
        )
    for position, top in ((left_position, 90), (right_position, 265)):
        pos_min = float(np.min(position))
        pos_max = float(np.max(position))
        for index, name in enumerate(("x", "y", "z")):
            points = _polyline(
                position[:, index],
                left,
                top,
                plot_width,
                145,
                pos_min,
                pos_max,
            )
            pieces.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{COLORS[name]}" stroke-width="1.7"/>'
            )

    grip_min = float(min(np.min(left_gripper), np.min(right_gripper)))
    grip_max = float(max(np.max(left_gripper), np.max(right_gripper)))
    for values, color in (
        (left_gripper, COLORS["left"]),
        (right_gripper, COLORS["right"]),
    ):
        points = _polyline(
            values, left, 440, plot_width, 100, grip_min, grip_max
        )
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )

    def arm_energy(arm: str) -> np.ndarray:
        if feature_bundle and arm in feature_bundle.get("arms", {}):
            return np.asarray(
                feature_bundle["arms"][arm]["energy_smooth"],
                dtype=np.float64,
            )
        if arm == active_arm:
            return np.asarray(energy, dtype=np.float64)
        return np.zeros(frame_count, dtype=np.float64)

    left_energy = arm_energy("left")
    right_energy = arm_energy("right")
    energy_min = float(min(np.min(left_energy), np.min(right_energy)))
    energy_max = float(max(np.max(left_energy), np.max(right_energy)))
    for values, color in (
        (left_energy, COLORS["left"]),
        (right_energy, COLORS["right"]),
    ):
        points = _polyline(
            values,
            left,
            570,
            plot_width,
            100,
            energy_min,
            energy_max,
        )
        pieces.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )

    def x_of(frame: int) -> float:
        return left + plot_width * frame / max(frame_count - 1, 1)

    def phase_runs(arm: str) -> Sequence[Dict[str, object]]:
        if not selection:
            return []
        timelines = selection.get("arm_timelines", {})
        if arm in timelines:
            return timelines[arm]
        diagnostics = selection.get("diagnostics", {})
        phase = diagnostics.get("phase", {}).get("per_arm", {}).get(arm, {})
        if phase:
            return phase.get("persistent_phase_runs", [])
        old_phase = (
            diagnostics.get("persistent_phases", {})
            .get("per_arm", {})
            .get(arm, {})
        )
        return old_phase.get("persistent_phase_runs", [])

    for arm, lane_y in (("left", 711), ("right", 752)):
        pieces.append(
            f'<text x="45" y="{lane_y + 18}" font-size="12" '
            f'font-family="sans-serif">{arm}</text>'
        )
        for run in phase_runs(arm):
            start = int(run["start_frame"])
            end = int(run["end_frame"])
            phase_name = str(run.get("phase", ""))
            x0 = x_of(start)
            x1 = x_of(end)
            color = PHASE_COLORS.get(phase_name, "#d1d5db")
            pieces.append(
                f'<rect x="{x0:.2f}" y="{lane_y}" '
                f'width="{max(x1 - x0, 1.0):.2f}" height="28" '
                f'fill="{color}" opacity="0.82" stroke="#ffffff"/>'
            )
            if x1 - x0 > 34:
                pieces.append(
                    f'<text x="{(x0 + x1) / 2.0:.2f}" '
                    f'y="{lane_y + 18}" text-anchor="middle" '
                    'font-size="10" font-family="sans-serif">'
                    f'{escape(phase_name)}</text>'
                )

    evidence_by_frame = {
        int(item["frame"]): item for item in (boundary_evidence or [])
    }
    boundary_summaries = []
    for boundary_index, segment in enumerate(segments[:-1], start=1):
        boundary = int(segment["end_frame"]) + 1
        evidence = evidence_by_frame.get(boundary)
        evidence_class = (
            str(evidence.get("evidence_class")) if evidence else ""
        )
        onset_refined = bool(
            evidence
            and (
                evidence.get("boundary_policy")
                == "earliest_persistent_successor_trend_onset"
                or "successor_trend_onset"
                in evidence.get("evidence_types", [])
            )
        )
        if onset_refined and evidence_class != "strong":
            color = COLORS["onset_boundary"]
            dash = "2,3"
            marker = "O"
        elif evidence_class == "strong":
            color = COLORS["strong_boundary"]
            dash = "7,3"
            marker = "S"
        elif evidence_class == "soft":
            color = COLORS["soft_boundary"]
            dash = "3,4"
            marker = "P"
        else:
            color = COLORS["boundary"]
            dash = "5,4"
            marker = "B"
        x = x_of(boundary)
        pieces.append(
            f'<line x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="800" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
        )
        if evidence:
            arms = "".join(
                "L" if arm == "left" else "R"
                for arm in evidence.get("source_arms", [])
            ) or "?"
            pieces.append(
                f'<text x="{x + 3:.2f}" y="102" font-size="11" '
                f'font-family="sans-serif" fill="{color}">'
                f'B{boundary_index}:{arms}-{marker}</text>'
            )
            transition = ""
            if evidence.get("previous_phase") or evidence.get("current_phase"):
                transition = (
                    f" {evidence.get('previous_phase', '?')}→"
                    f"{evidence.get('current_phase', '?')}"
                )
            boundary_summaries.append(
                (
                    f"B{boundary_index}=f{boundary} "
                    f"{arms}-{marker}{transition} "
                    f"{evidence.get('evidence_type', '')}"
                )
            )
        else:
            boundary_summaries.append(
                f"B{boundary_index}=f{boundary} no-evidence-record"
            )
    if reference_segments:
        for segment in reference_segments[:-1]:
            boundary = int(segment["end_frame"]) + 1
            x = x_of(boundary)
            pieces.append(
                f'<line x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="800" '
                f'stroke="{COLORS["reference"]}" stroke-width="2" opacity="0.7"/>'
            )

    label_y = 825
    for segment in segments:
        center = (int(segment["start_frame"]) + int(segment["end_frame"])) / 2.0
        left_label = str(segment.get("left_action", ""))
        right_label = str(segment.get("right_action", ""))
        relation = str(segment.get("coordination_relation", ""))
        label = f"L:{left_label} | R:{right_label}"
        if relation:
            label += f" | {relation}"
        pieces.append(
            f'<text x="{x_of(center):.2f}" y="{label_y}" text-anchor="middle" '
            f'font-size="10" font-family="sans-serif">{escape(label)}</text>'
        )

    for tick in range(0, 6):
        frame = int(round((frame_count - 1) * tick / 5.0))
        x = x_of(frame)
        if timestamps is not None:
            tick_label = f"f{frame} / {float(timestamps[frame]):.2f}s"
        else:
            tick_label = f"f{frame}"
        pieces.append(
            f'<line x1="{x:.2f}" y1="790" x2="{x:.2f}" y2="798" '
            'stroke="#374151"/>'
        )
        pieces.append(
            f'<text x="{x:.2f}" y="855" text-anchor="middle" '
            f'font-size="10" font-family="sans-serif">{tick_label}</text>'
        )

    legend_y = 885
    legend_items = [
        (COLORS["strong_boundary"], "S strong state event"),
        (COLORS["soft_boundary"], "P persistent phase boundary"),
        (COLORS["onset_boundary"], "O successor-trend onset"),
        (COLORS["reference"], "green weak reference"),
        (COLORS["left"], "left-arm signal"),
        (COLORS["right"], "right-arm signal"),
    ]
    legend_x = 55
    for color, text in legend_items:
        pieces.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        pieces.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" '
            f'font-size="11" font-family="sans-serif">{escape(text)}</text>'
        )
        legend_x += 220

    pieces.append(
        '<text x="45" y="918" font-size="12" font-family="sans-serif" '
        'font-weight="bold">Boundary audit key (full details in JSON):</text>'
    )
    for index, summary in enumerate(boundary_summaries[:12]):
        column = index % 3
        row = index // 3
        pieces.append(
            f'<text x="{45 + column * 485}" y="{942 + row * 19}" '
            f'font-size="10" font-family="monospace">{escape(summary)}</text>'
        )
    if len(boundary_summaries) > 12:
        pieces.append(
            f'<text x="45" y="1018" font-size="10" '
            f'font-family="sans-serif">... {len(boundary_summaries) - 12} '
            "more boundaries; inspect boundary_evidence in JSON.</text>"
        )
    pieces.append("</svg>")
    output.write_text("\n".join(pieces), encoding="utf-8")
