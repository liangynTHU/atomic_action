"""Non-authoritative boundary proposals for recovery trajectories.

Recovery semantics are not identifiable from robot state alone.  This module
therefore proposes review points for arbitrary-D joint or end-pose streams; it
never emits training-ready labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RecoveryProposalConfig:
    min_segment_seconds: float = 0.25
    max_segment_seconds: float = 3.0
    smoothing_seconds: float = 0.10
    direction_window_seconds: float = 0.18
    direction_cosine_threshold: float = 0.35
    pause_energy_quantile: float = 0.25

    def __post_init__(self) -> None:
        if self.min_segment_seconds <= 0:
            raise ValueError("min_segment_seconds must be positive")
        if self.max_segment_seconds <= self.min_segment_seconds:
            raise ValueError("max_segment_seconds must exceed min_segment_seconds")
        if self.smoothing_seconds <= 0 or self.direction_window_seconds <= 0:
            raise ValueError("proposal windows must be positive")
        if not -1.0 <= self.direction_cosine_threshold <= 1.0:
            raise ValueError("direction cosine threshold must be in [-1,1]")
        if not 0.0 < self.pause_energy_quantile < 1.0:
            raise ValueError("pause_energy_quantile must be in (0,1)")


def _validate(
    states: np.ndarray,
    timestamps: np.ndarray,
    state_layout: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("recovery proposal states must have shape [T,D], T>=2")
    if len(times) != len(values):
        raise ValueError("timestamps must have the same length as states")
    if len(state_layout) != values.shape[1]:
        raise ValueError("state_layout dimension mismatch")
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise ValueError("recovery proposal inputs must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return values, times


def _odd_frames(seconds: float, dt: float, minimum: int = 3) -> int:
    frames = max(minimum, int(round(seconds / dt)))
    return frames if frames % 2 == 1 else frames + 1


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) <= 2:
        return values.copy()
    width = min(width, len(values) if len(values) % 2 else len(values) - 1)
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")


def _candidate(
    frame: int,
    score: float,
    evidence_type: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frame": int(frame),
        "score": float(score),
        "evidence_types": [evidence_type],
        "details": details,
    }


def _merge_candidates(
    candidates: Sequence[dict[str, Any]],
    minimum_separation: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda value: int(value["frame"]))
    groups: list[list[dict[str, Any]]] = []
    for value in ordered:
        separated = (
            not groups
            or int(value["frame"]) - int(groups[-1][-1]["frame"]) > minimum_separation
        )
        if separated:
            groups.append([value])
        else:
            groups[-1].append(value)
    merged: list[dict[str, Any]] = []
    for group in groups:
        selected = max(group, key=lambda value: float(value["score"]))
        evidence_types = sorted(
            {evidence for value in group for evidence in value["evidence_types"]}
        )
        merged.append(
            {
                **selected,
                "evidence_types": evidence_types,
                "merged_candidate_count": len(group),
            }
        )
    return merged


def _enforce_max_length(
    boundaries: list[int],
    energy: np.ndarray,
    max_frames: int,
    min_frames: int,
) -> list[int]:
    result = [boundaries[0]]
    for right in boundaries[1:]:
        left = result[-1]
        while right - left > max_frames:
            nominal = left + max_frames
            radius = max(min_frames, max_frames // 4)
            low = max(left + min_frames, nominal - radius)
            high = min(right - min_frames, nominal + radius)
            if high <= low:
                split = nominal
            else:
                split = low + int(np.argmin(energy[low : high + 1]))
            result.append(int(split))
            left = int(split)
        result.append(int(right))
    return sorted(set(result))


def propose_recovery_segments(
    states: np.ndarray,
    timestamps: np.ndarray,
    state_layout: Sequence[str],
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    config: RecoveryProposalConfig | None = None,
) -> dict[str, Any]:
    """Propose review boundaries without assigning semantic subtask truth."""

    values, times = _validate(states, timestamps, state_layout)
    selected = config or RecoveryProposalConfig()
    end = len(values) - 1 if end_frame is None else int(end_frame)
    start = int(start_frame)
    if start < 0 or end >= len(values) or end <= start:
        raise ValueError("invalid proposal frame range")
    local_states = values[start : end + 1]
    local_times = times[start : end + 1]
    dt = float(np.median(np.diff(local_times)))
    min_frames = max(2, int(round(selected.min_segment_seconds / dt)))
    max_frames = max(min_frames + 1, int(round(selected.max_segment_seconds / dt)))
    smooth_frames = _odd_frames(selected.smoothing_seconds, dt)
    direction_frames = max(
        2,
        int(round(selected.direction_window_seconds / dt)),
    )

    deltas = np.diff(local_states, axis=0)
    delta_times = np.diff(local_times)
    velocity = deltas / delta_times[:, None]
    scale = np.quantile(np.abs(velocity), 0.90, axis=0)
    nonzero = scale[scale > 1e-8]
    fallback = float(np.median(nonzero)) if len(nonzero) else 1.0
    scale = np.where(scale > 1e-8, scale, fallback)
    normalized = velocity / scale
    raw_energy = np.linalg.norm(normalized, axis=1) / np.sqrt(values.shape[1])
    energy = np.concatenate(([raw_energy[0]], raw_energy))
    energy = _smooth(energy, smooth_frames)
    pause_threshold = float(np.quantile(energy, selected.pause_energy_quantile))
    active_threshold = max(
        pause_threshold * 1.5,
        float(np.quantile(energy, 0.65)) * 0.35,
        1e-6,
    )
    active = energy > active_threshold

    candidates: list[dict[str, Any]] = []
    for frame in range(1, len(local_states)):
        if bool(active[frame]) != bool(active[frame - 1]):
            candidates.append(
                _candidate(
                    frame,
                    1.0 + abs(float(energy[frame] - energy[frame - 1])),
                    "motion_activity_transition",
                    {
                        "before_active": bool(active[frame - 1]),
                        "after_active": bool(active[frame]),
                    },
                )
            )

    for frame in range(direction_frames, len(velocity) - direction_frames + 1):
        before = np.mean(
            normalized[frame - direction_frames : frame],
            axis=0,
        )
        after = np.mean(normalized[frame : frame + direction_frames], axis=0)
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        if before_norm < 0.10 or after_norm < 0.10:
            continue
        cosine = float(np.dot(before, after) / (before_norm * after_norm))
        if cosine < selected.direction_cosine_threshold:
            candidates.append(
                _candidate(
                    frame,
                    1.0 - cosine,
                    "joint_velocity_direction_change",
                    {"cosine": cosine},
                )
            )

    gripper_indices = [
        index
        for index, name in enumerate(state_layout)
        if "gripper" in str(name).lower()
    ]
    for index in gripper_indices:
        magnitude = np.abs(deltas[:, index])
        positive = magnitude[magnitude > 1e-8]
        if not len(positive):
            continue
        threshold = max(float(np.quantile(positive, 0.65)), 1e-4)
        changing = magnitude >= threshold
        padded = np.concatenate(([False], changing, [False]))
        for point in np.flatnonzero(padded[1:] != padded[:-1]):
            frame = min(int(point), len(local_states) - 1)
            candidates.append(
                _candidate(
                    frame,
                    2.0,
                    "gripper_transition",
                    {
                        "state_name": str(state_layout[index]),
                        "threshold": threshold,
                    },
                )
            )

    merged = _merge_candidates(candidates, max(1, min_frames // 2))
    internal = [
        value
        for value in merged
        if min_frames <= int(value["frame"]) <= len(local_states) - min_frames
    ]
    local_boundaries = (
        [0] + [int(value["frame"]) for value in internal] + [len(local_states)]
    )
    local_boundaries = _enforce_max_length(
        sorted(set(local_boundaries)),
        energy,
        max_frames,
        min_frames,
    )
    evidence_by_frame = {int(value["frame"]): value for value in internal}
    segments = []
    for index, (left, right) in enumerate(
        zip(local_boundaries[:-1], local_boundaries[1:])
    ):
        boundary = evidence_by_frame.get(left)
        segments.append(
            {
                "proposal_index": index,
                "start_frame": start + int(left),
                "end_frame": start + int(right) - 1,
                "duration_seconds": float(
                    local_times[int(right) - 1] - local_times[int(left)] + dt
                ),
                "candidate_label": "REVIEW_REQUIRED",
                "start_boundary_evidence": boundary,
            }
        )
    return {
        "schema": "recovery_segment_proposals",
        "authoritative": False,
        "review_required": True,
        "training_eligible": False,
        "method": "generic_state_motion_proposals",
        "warning": (
            "These boundaries describe motion changes only. They do not "
            "establish intended subtasks, outcomes, mismatches, or recovery "
            "semantics and must not be used directly for CoT training."
        ),
        "state_layout": list(state_layout),
        "frame_range": [start, end],
        "median_dt": dt,
        "config": asdict(selected),
        "diagnostics": {
            "pause_energy_threshold": pause_threshold,
            "active_energy_threshold": active_threshold,
            "raw_candidate_count": len(candidates),
            "merged_candidate_count": len(merged),
        },
        "boundaries": [start + value for value in local_boundaries[1:-1]],
        "segments": segments,
    }
