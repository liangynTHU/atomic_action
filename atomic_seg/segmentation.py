"""State-only atomic segmentation for dual-arm robot episodes.

Boundary decisions use only the 16-D end-pose state and timestamps. Images,
task text and action targets are deliberately downstream annotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import SegmentationConfig
from .features import (
    arm_components,
    contiguous_runs,
    gripper_events,
    local_motion_statistics,
    moving_average,
    normalize_direction,
    segment_motion_summary,
)
from .geometry import quaternion_log_delta


ARM_NAMES = ("left", "right")
KINEMATIC_PHASES = {"still", "move", "turn", "lift", "lower"}
STATE_LAYOUT = (
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
)

STATE_ONLY_DECISION_POLICY: Dict[str, object] = {
    "boundary_decision_inputs": [
        "observation.state.left.xyz",
        "observation.state.left.orientation",
        "observation.state.left.gripper",
        "observation.state.right.xyz",
        "observation.state.right.orientation",
        "observation.state.right.gripper",
        "timestamp",
    ],
    "excluded_from_boundary_decisions": [
        "image",
        "video",
        "action",
        "task_text",
        "instruction",
        "object_label",
        "success_label",
    ],
    "boundary_lock_after_state_segmentation": True,
}


@dataclass(frozen=True)
class TemporalParameters:
    sample_period_seconds: float
    fps_estimate: float
    min_segment_frames: int
    pause_min_frames: int
    boundary_merge_frames: int
    phase_window_frames: int
    phase_min_frames: int
    phase_bridge_frames: int
    onset_search_frames: int
    onset_confirm_frames: int
    cross_arm_sync_frames: int
    coordination_min_segment_frames: int
    exchange_search_frames: int
    gripper_pre_window_frames: int
    gripper_post_window_frames: int
    energy_smooth_frames: int


def segments_from_boundaries(
    boundaries: Sequence[int],
    frame_count: int,
) -> List[Dict[str, int]]:
    """Convert boundary starts into disjoint inclusive frame intervals."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_count == 0:
        return []
    cuts = sorted(
        {int(value) for value in boundaries if 0 < int(value) < frame_count}
    )
    starts = [0, *cuts]
    ends = [value - 1 for value in cuts] + [frame_count - 1]
    return [
        {"start_frame": start, "end_frame": end}
        for start, end in zip(starts, ends)
        if end >= start
    ]


def boundaries_from_segments(
    segments: Sequence[Mapping[str, object]],
) -> List[int]:
    return [
        int(segment["end_frame"]) + 1
        for segment in segments[:-1]
    ]


def _validate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("trajectory must have shape [frames, 16]")
    if len(values) == 0:
        raise ValueError("trajectory must contain at least one frame")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory must contain only finite values")
    return values


def _timestamps_or_default(
    timestamps: np.ndarray | Sequence[float] | None,
    frame_count: int,
    default_period: float,
) -> np.ndarray:
    if timestamps is None:
        return np.arange(frame_count, dtype=np.float64) * float(
            default_period
        )
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(values) != frame_count:
        raise ValueError(
            "timestamps must have the same length as the trajectory"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps must be finite")
    if frame_count > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return values


def _seconds_to_frames(
    seconds: float,
    sample_period: float,
    minimum: int = 1,
) -> int:
    return max(minimum, int(round(float(seconds) / sample_period)))


def resolve_temporal_parameters(
    timestamps: np.ndarray | Sequence[float] | None,
    frame_count: int,
    config: SegmentationConfig,
) -> Tuple[np.ndarray, TemporalParameters]:
    values = _timestamps_or_default(
        timestamps,
        frame_count,
        config.default_sample_period_seconds,
    )
    sample_period = (
        float(np.median(np.diff(values)))
        if frame_count > 1
        else float(config.default_sample_period_seconds)
    )
    if sample_period <= 0.0:
        raise ValueError("sample period must be positive")
    temporal = TemporalParameters(
        sample_period_seconds=sample_period,
        fps_estimate=1.0 / sample_period,
        min_segment_frames=_seconds_to_frames(
            config.min_segment_seconds,
            sample_period,
        ),
        pause_min_frames=_seconds_to_frames(
            config.pause_min_seconds,
            sample_period,
        ),
        boundary_merge_frames=_seconds_to_frames(
            config.boundary_merge_seconds,
            sample_period,
        ),
        phase_window_frames=_seconds_to_frames(
            config.phase_window_seconds,
            sample_period,
            minimum=3,
        ),
        phase_min_frames=_seconds_to_frames(
            config.phase_min_seconds,
            sample_period,
        ),
        phase_bridge_frames=_seconds_to_frames(
            config.phase_bridge_seconds,
            sample_period,
        ),
        onset_search_frames=_seconds_to_frames(
            config.onset_search_seconds,
            sample_period,
        ),
        onset_confirm_frames=_seconds_to_frames(
            config.onset_confirm_seconds,
            sample_period,
            minimum=2,
        ),
        cross_arm_sync_frames=_seconds_to_frames(
            config.cross_arm_sync_seconds,
            sample_period,
        ),
        coordination_min_segment_frames=_seconds_to_frames(
            config.coordination_min_segment_seconds,
            sample_period,
        ),
        exchange_search_frames=_seconds_to_frames(
            config.exchange_search_seconds,
            sample_period,
        ),
        gripper_pre_window_frames=_seconds_to_frames(
            config.gripper_pre_window_seconds,
            sample_period,
        ),
        gripper_post_window_frames=_seconds_to_frames(
            config.gripper_post_window_seconds,
            sample_period,
        ),
        energy_smooth_frames=_seconds_to_frames(
            0.10,
            sample_period,
        ),
    )
    return values, temporal


def _normalize_gripper(
    values: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(raw, [5.0, 95.0])
    span = float(high - low)
    normalized = (
        np.zeros_like(raw)
        if span <= 1e-9
        else np.clip((raw - low) / span, 0.0, 1.0)
    )
    return normalized, {
        "low": float(low),
        "high": float(high),
        "span": span,
    }


def _phase_activity_scores(
    linear_velocity: np.ndarray,
    angular_speed: np.ndarray,
    energy: np.ndarray,
    vertical_direction: np.ndarray,
    config: SegmentationConfig,
) -> Dict[str, np.ndarray]:
    vertical = normalize_direction(vertical_direction)
    vertical_speed = linear_velocity @ vertical
    horizontal_velocity = (
        linear_velocity - vertical_speed[:, None] * vertical[None, :]
    )
    horizontal_speed = np.linalg.norm(horizontal_velocity, axis=1)
    translation_speed = np.linalg.norm(linear_velocity, axis=1)
    return {
        "still": 1.0
        / (
            1.0
            + energy
            / max(config.state_pause_energy_threshold, 1e-12)
        ),
        "move": horizontal_speed
        / max(config.onset_translation_speed, 1e-12),
        "turn": np.maximum(
            angular_speed / max(config.onset_rotation_speed, 1e-12)
            - 0.35
            * translation_speed
            / max(config.onset_translation_speed, 1e-12),
            0.0,
        ),
        "lift": np.maximum(vertical_speed, 0.0)
        / max(config.onset_vertical_speed, 1e-12),
        "lower": np.maximum(-vertical_speed, 0.0)
        / max(config.onset_vertical_speed, 1e-12),
    }


def extract_state_features(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> Dict[str, object]:
    frame_count = len(trajectory)
    dt = np.diff(timestamps)
    arms: Dict[str, Dict[str, object]] = {}
    vertical_direction = np.asarray(
        config.vertical_direction,
        dtype=np.float64,
    )
    for arm in ARM_NAMES:
        position, quaternion, gripper_raw = arm_components(trajectory, arm)
        gripper_unit, normalization = _normalize_gripper(gripper_raw)
        linear_velocity = np.zeros((frame_count, 3), dtype=np.float64)
        angular_velocity = np.zeros((frame_count, 3), dtype=np.float64)
        gripper_rate = np.zeros(frame_count, dtype=np.float64)
        if frame_count > 1:
            linear_velocity[1:] = np.diff(position, axis=0) / dt[:, None]
            angular_velocity[1:] = (
                quaternion_log_delta(quaternion[:-1], quaternion[1:])
                / dt[:, None]
            )
            gripper_rate[1:] = np.diff(gripper_unit) / dt
        translation_speed = np.linalg.norm(linear_velocity, axis=1)
        angular_speed = np.linalg.norm(angular_velocity, axis=1)
        energy = (
            translation_speed
            / max(config.translation_speed_scale, 1e-12)
            + angular_speed
            / max(config.rotation_speed_scale, 1e-12)
            + np.abs(gripper_rate)
            / max(config.gripper_rate_scale, 1e-12)
        )
        energy_smooth = moving_average(
            energy,
            temporal.energy_smooth_frames,
        )
        arms[arm] = {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper_raw,
            "gripper_unit": gripper_unit,
            "gripper_normalization": normalization,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "gripper_rate": gripper_rate,
            "translation_speed": translation_speed,
            "angular_speed": angular_speed,
            "energy": energy,
            "energy_smooth": energy_smooth,
            "phase_scores": _phase_activity_scores(
                linear_velocity,
                angular_speed,
                energy,
                vertical_direction,
                config,
            ),
        }
    return {
        "timestamps": timestamps,
        "sample_period_seconds": temporal.sample_period_seconds,
        "fps_estimate": temporal.fps_estimate,
        "arms": arms,
    }


def _candidate(
    frame: int,
    arm: str,
    evidence_type: str,
    evidence_class: str,
    strength: float,
    timestamps: np.ndarray,
    **extra: object,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "frame": int(frame),
        "time_seconds": float(timestamps[int(frame)]),
        "source_arm": arm,
        "source_arms": [arm],
        "evidence_type": evidence_type,
        "evidence_types": [evidence_type],
        "evidence_class": evidence_class,
        "strength": float(strength),
        "supported_by_both_arms": False,
    }
    record.update(extra)
    return record


def _merge_short_boolean_gaps(
    mask: np.ndarray,
    max_gap: int,
    blocking_mask: np.ndarray | None = None,
) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    blocked = (
        np.zeros_like(output)
        if blocking_mask is None
        else np.asarray(blocking_mask, dtype=bool)
    )
    if max_gap <= 0:
        return output
    for start, end in contiguous_runs(~output):
        if (
            end - start + 1 <= max_gap
            and start > 0
            and end + 1 < len(output)
            and output[start - 1]
            and output[end + 1]
            and not bool(np.any(blocked[start : end + 1]))
        ):
            output[start : end + 1] = True
    return output


def _gripper_trend_events(
    gripper_unit: np.ndarray,
    gripper_rate: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for direction, sign in (("increase", 1.0), ("decrease", -1.0)):
        active = (
            sign * np.asarray(gripper_rate)
            >= config.gripper_event_rate_threshold
        )
        opposite = (
            sign * np.asarray(gripper_rate)
            <= -config.gripper_event_rate_threshold
        )
        active = _merge_short_boolean_gaps(
            active,
            temporal.phase_bridge_frames,
            blocking_mask=opposite,
        )
        for start, end in contiguous_runs(active):
            baseline = max(0, start - 1)
            delta = float(gripper_unit[end] - gripper_unit[baseline])
            if abs(delta) < config.gripper_event_min_delta:
                continue
            events.append(
                {
                    "onset_frame": int(start),
                    "done_frame": int(end),
                    "direction": direction,
                    "delta_normalized": delta,
                }
            )
    return sorted(events, key=lambda item: int(item["onset_frame"]))


def _pause_evidence(
    arm_bundle: Mapping[str, object],
    arm: str,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    energy = np.asarray(arm_bundle["energy_smooth"])
    pauses = [
        run
        for run in contiguous_runs(
            energy < config.state_pause_energy_threshold
        )
        if run[1] - run[0] + 1 >= temporal.pause_min_frames
    ]
    candidates: List[Dict[str, object]] = []
    for start, end in pauses:
        if start <= 1 or end >= len(energy) - 2:
            continue
        frame = start + int(np.argmin(energy[start : end + 1]))
        depth = max(
            0.0,
            (
                config.state_pause_energy_threshold - float(energy[frame])
            )
            / max(config.state_pause_energy_threshold, 1e-12),
        )
        candidates.append(
            _candidate(
                frame,
                arm,
                "state_motion_energy_valley",
                "strong",
                2.0 + depth,
                timestamps,
                pause_run=[int(start), int(end)],
                pause_duration_seconds=float(
                    timestamps[end] - timestamps[start]
                ),
                energy=float(energy[frame]),
            )
        )
    return candidates, pauses


def _gripper_evidence(
    arm_bundle: Mapping[str, object],
    arm: str,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    energy = np.asarray(arm_bundle["energy_smooth"])
    events = _gripper_trend_events(
        np.asarray(arm_bundle["gripper_unit"]),
        np.asarray(arm_bundle["gripper_rate"]),
        temporal,
        config,
    )
    candidates: List[Dict[str, object]] = []
    for event_index, event in enumerate(events):
        onset = int(event["onset_frame"])
        done = int(event["done_frame"])
        direction = str(event["direction"])
        before_start = max(1, onset - temporal.gripper_pre_window_frames)
        before_stop = max(before_start + 1, onset + 1)
        before = before_start + int(
            np.argmin(energy[before_start:before_stop])
        )
        after_start = done
        after_stop = min(
            len(energy),
            done + temporal.gripper_post_window_frames + 1,
        )
        after = after_start + int(
            np.argmin(energy[after_start:after_stop])
        )
        common = {
            "event_index": event_index,
            "gripper_direction": direction,
            "event_interval": [onset, done],
            "gripper_delta_normalized": float(
                event["delta_normalized"]
            ),
        }
        candidates.extend(
            [
                _candidate(
                    before,
                    arm,
                    f"pre_gripper_{direction}_stabilization",
                    "strong",
                    2.5
                    + max(
                        0.0,
                        config.state_pause_energy_threshold
                        - float(energy[before]),
                    ),
                    timestamps,
                    energy=float(energy[before]),
                    **common,
                ),
                _candidate(
                    after,
                    arm,
                    f"post_gripper_{direction}_stabilization",
                    "strong",
                    2.5
                    + max(
                        0.0,
                        config.state_pause_energy_threshold
                        - float(energy[after]),
                    ),
                    timestamps,
                    energy=float(energy[after]),
                    **common,
                ),
            ]
        )
    return candidates, events


def _phase_from_statistics(
    statistics: Mapping[str, object],
    config: SegmentationConfig,
) -> str:
    translation = float(statistics["translation_norm"])
    path_length = float(statistics["path_length"])
    rotation = float(statistics["accumulated_rotation"])
    vertical_displacement = float(statistics["vertical_displacement"])
    vertical_ratio = float(statistics["vertical_ratio"])
    if (
        translation < config.still_translation_threshold
        and path_length < config.still_path_threshold
        and rotation < config.still_rotation_threshold
    ):
        return "still"
    if (
        abs(vertical_displacement)
        >= config.vertical_displacement_threshold
        and vertical_ratio >= config.vertical_ratio_threshold
    ):
        return "lift" if vertical_displacement > 0.0 else "lower"
    rotation_score = rotation / max(config.rotation_window_scale, 1e-12)
    translation_score = translation / max(
        config.translation_window_scale,
        1e-12,
    )
    if rotation_score > config.rotation_dominance_ratio * max(
        translation_score,
        1e-6,
    ):
        return "turn"
    return "move"


def _raw_label_runs(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    output: List[Dict[str, object]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            output.append(
                {
                    "start_frame": start,
                    "end_frame": index - 1,
                    "phase": labels[start],
                }
            )
            start = index
    return output


def _persistent_phase_runs(
    labels: Sequence[str],
    min_frames: int,
) -> List[Dict[str, object]]:
    runs = _raw_label_runs(labels)
    changed = True
    while changed and len(runs) >= 3:
        changed = False
        for index in range(1, len(runs) - 1):
            run = runs[index]
            length = int(run["end_frame"]) - int(run["start_frame"]) + 1
            if (
                length < min_frames
                and runs[index - 1]["phase"] == runs[index + 1]["phase"]
            ):
                runs[index - 1]["end_frame"] = runs[index + 1][
                    "end_frame"
                ]
                del runs[index : index + 2]
                changed = True
                break
    output: List[Dict[str, object]] = []
    for run in runs:
        length = int(run["end_frame"]) - int(run["start_frame"]) + 1
        if length >= min_frames or not output:
            output.append(dict(run))
        else:
            output[-1]["end_frame"] = run["end_frame"]
    if len(output) >= 2:
        first_length = (
            int(output[0]["end_frame"])
            - int(output[0]["start_frame"])
            + 1
        )
        if first_length < min_frames:
            output[1]["start_frame"] = output[0]["start_frame"]
            output = output[1:]
    merged: List[Dict[str, object]] = []
    for run in output:
        if merged and merged[-1]["phase"] == run["phase"]:
            merged[-1]["end_frame"] = run["end_frame"]
        else:
            merged.append(dict(run))
    return merged


def _phase_evidence(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {"per_arm": {}}
    vertical = normalize_direction(
        np.asarray(config.vertical_direction, dtype=np.float64)
    )
    for arm in ARM_NAMES:
        statistics = [
            local_motion_statistics(
                trajectory,
                arm,
                frame,
                temporal.phase_window_frames,
                vertical,
            )
            for frame in range(len(trajectory))
        ]
        labels = [
            _phase_from_statistics(item, config)
            for item in statistics
        ]
        runs = _persistent_phase_runs(
            labels,
            temporal.phase_min_frames,
        )
        arm_candidates: List[Dict[str, object]] = []
        for previous, current in zip(runs, runs[1:]):
            previous_phase = str(previous["phase"])
            current_phase = str(current["phase"])
            if (
                previous_phase == current_phase
                or previous_phase not in KINEMATIC_PHASES
                or current_phase not in KINEMATIC_PHASES
            ):
                continue
            boundary = int(current["start_frame"])
            previous_duration = (
                int(previous["end_frame"])
                - int(previous["start_frame"])
                + 1
            )
            current_duration = (
                int(current["end_frame"])
                - int(current["start_frame"])
                + 1
            )
            previous_stats = statistics[int(previous["end_frame"])]
            current_stats = statistics[int(current["start_frame"])]
            contrast = abs(
                float(current_stats["vertical_ratio"])
                - float(previous_stats["vertical_ratio"])
            ) + abs(
                float(current_stats["accumulated_rotation"])
                - float(previous_stats["accumulated_rotation"])
            ) / max(config.rotation_window_scale, 1e-12)
            candidate = _candidate(
                boundary,
                arm,
                "persistent_state_phase_transition",
                "soft",
                1.0 + min(contrast, 2.0),
                timestamps,
                previous_phase=previous_phase,
                current_phase=current_phase,
                previous_duration_frames=previous_duration,
                current_duration_frames=current_duration,
                boundary_policy="persistent_phase_run_start",
            )
            candidates.append(candidate)
            arm_candidates.append(candidate)
        diagnostics["per_arm"][arm] = {
            "raw_phase_labels": labels,
            "persistent_phase_runs": runs,
            "soft_candidate_count": len(arm_candidates),
        }
    return candidates, diagnostics


def _collect_state_evidence(
    trajectory: np.ndarray,
    features: Mapping[str, object],
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmentationConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    strong: List[Dict[str, object]] = []
    strong_diagnostics: Dict[str, object] = {"per_arm": {}}
    for arm in ARM_NAMES:
        arm_bundle = features["arms"][arm]
        pauses, pause_runs = _pause_evidence(
            arm_bundle,
            arm,
            timestamps,
            temporal,
            config,
        )
        grippers, gripper_events_found = _gripper_evidence(
            arm_bundle,
            arm,
            timestamps,
            temporal,
            config,
        )
        strong.extend(pauses)
        strong.extend(grippers)
        strong_diagnostics["per_arm"][arm] = {
            "pause_runs": [list(run) for run in pause_runs],
            "gripper_events": gripper_events_found,
            "strong_candidate_count": len(pauses) + len(grippers),
        }
    soft, phase_diagnostics = _phase_evidence(
        trajectory,
        timestamps,
        temporal,
        config,
    )
    return strong, soft, {
        "strong": strong_diagnostics,
        "phase": phase_diagnostics,
    }


def successor_trend_onset(
    previous_scores: np.ndarray,
    current_scores: np.ndarray,
    nominal_frame: int,
    search_frames: int,
    confirm_frames: int,
    *,
    low_threshold: float = 0.25,
    sustained_threshold: float = 0.55,
) -> Tuple[int, Dict[str, object]]:
    """Choose the earliest sustained successor trend near a proposal."""

    previous = np.asarray(previous_scores, dtype=np.float64)
    current = np.asarray(current_scores, dtype=np.float64)
    frame_count = min(len(previous), len(current))
    nominal = int(np.clip(nominal_frame, 0, max(frame_count - 1, 0)))
    low = max(1, nominal - int(search_frames))
    high = min(frame_count - 1, nominal + int(search_frames))
    confirm = max(2, int(confirm_frames))
    chosen = nominal
    found = False
    for frame in range(low, high + 1):
        future = current[frame : min(frame_count, frame + confirm)]
        if len(future) < 2:
            continue
        before = current[max(0, frame - confirm) : frame]
        future_mean = float(np.mean(future))
        before_mean = float(np.mean(before)) if len(before) else 0.0
        sustained_fraction = float(np.mean(future >= low_threshold))
        rising = (
            float(current[frame]) >= low_threshold
            and future_mean >= sustained_threshold
            and sustained_fraction >= 0.67
            and (
                before_mean < sustained_threshold
                or future_mean >= before_mean + 0.15
            )
        )
        if rising:
            chosen = frame
            found = True
            break
    before_slice = slice(max(0, chosen - confirm), chosen)
    after_slice = slice(chosen, min(frame_count, chosen + confirm))
    old_before = (
        float(np.mean(previous[before_slice]))
        if chosen > 0
        else float(previous[chosen])
    )
    old_after = float(np.mean(previous[after_slice]))
    new_before = (
        float(np.mean(current[before_slice]))
        if chosen > 0
        else float(current[chosen])
    )
    new_after = float(np.mean(current[after_slice]))
    diagnostics = {
        "nominal_frame": nominal,
        "refined_frame": int(chosen),
        "onset_found": found,
        "previous_score_before": old_before,
        "previous_score_after": old_after,
        "successor_score_before": new_before,
        "successor_score_after": new_after,
        "previous_trend_falling": old_after < old_before - 0.05,
        "successor_trend_rising": new_after > new_before + 0.05,
    }
    diagnostics["competing_trends_detected"] = bool(
        diagnostics["previous_trend_falling"]
        and diagnostics["successor_trend_rising"]
    )
    return int(chosen), diagnostics


def _refine_soft_evidence(
    candidates: Sequence[Dict[str, object]],
    features: Mapping[str, object],
    timestamps: np.ndarray,
    temporal: TemporalParameters,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for candidate in candidates:
        record = dict(candidate)
        arm = str(record["source_arm"])
        previous_phase = str(record.get("previous_phase", "still"))
        current_phase = str(record.get("current_phase", "move"))
        scores = features["arms"][arm]["phase_scores"]
        nominal = int(record["frame"])
        onset, diagnostics = successor_trend_onset(
            np.asarray(scores[previous_phase]),
            np.asarray(scores[current_phase]),
            nominal,
            temporal.onset_search_frames,
            temporal.onset_confirm_frames,
            low_threshold=0.45 if current_phase == "still" else 0.25,
        )
        record.update(
            {
                "original_frame": nominal,
                "frame": onset,
                "time_seconds": float(timestamps[onset]),
                "original_evidence_type": record["evidence_type"],
                "evidence_type": "successor_trend_onset",
                "evidence_types": sorted(
                    {
                        *record.get("evidence_types", []),
                        "successor_trend_onset",
                    }
                ),
                "boundary_policy": (
                    "earliest_persistent_successor_trend_onset"
                ),
                "onset_refinement": diagnostics,
            }
        )
        output.append(record)
    return output


def _combine_cluster(
    cluster: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    strong = [
        candidate
        for candidate in cluster
        if candidate["evidence_class"] == "strong"
    ]
    priority = strong or list(cluster)
    if strong:
        representative = max(
            priority,
            key=lambda candidate: (
                float(candidate["strength"]),
                -int(candidate["frame"]),
            ),
        )
        weights = np.asarray(
            [
                max(float(candidate["strength"]), 1e-6)
                for candidate in priority
            ]
        )
        frames = np.asarray(
            [int(candidate["frame"]) for candidate in priority],
            dtype=np.float64,
        )
        frame = int(round(float(np.average(frames, weights=weights))))
    else:
        representative = min(
            priority,
            key=lambda candidate: int(candidate["frame"]),
        )
        frame = int(representative["frame"])
    source_arms = sorted(
        {
            str(arm)
            for candidate in cluster
            for arm in candidate.get(
                "source_arms",
                [candidate["source_arm"]],
            )
        }
    )
    evidence_types = sorted(
        {
            str(evidence_type)
            for candidate in cluster
            for evidence_type in candidate.get(
                "evidence_types",
                [candidate["evidence_type"]],
            )
        }
    )
    return {
        **dict(representative),
        "frame": frame,
        "source_arm": (
            source_arms[0] if len(source_arms) == 1 else "both"
        ),
        "source_arms": source_arms,
        "evidence_types": evidence_types,
        "evidence_class": "strong" if strong else "soft",
        "strength": float(
            max(float(candidate["strength"]) for candidate in cluster)
        ),
        "supported_by_both_arms": set(source_arms) == set(ARM_NAMES),
        "support_count": len(cluster),
        "support": [dict(candidate) for candidate in cluster],
    }


def _filter_evidence(
    candidates: Sequence[Dict[str, object]],
    frame_count: int,
    temporal: TemporalParameters,
    *,
    min_segment_frames: int | None = None,
    merge_radius_frames: int | None = None,
) -> List[Dict[str, object]]:
    minimum = (
        temporal.min_segment_frames
        if min_segment_frames is None
        else int(min_segment_frames)
    )
    merge_radius = (
        temporal.boundary_merge_frames
        if merge_radius_frames is None
        else max(0, int(merge_radius_frames))
    )
    valid = [
        dict(candidate)
        for candidate in candidates
        if minimum <= int(candidate["frame"]) <= frame_count - minimum
    ]
    if not valid:
        return []
    ordered = sorted(valid, key=lambda item: int(item["frame"]))
    clusters: List[List[Dict[str, object]]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        if (
            int(candidate["frame"])
            - int(clusters[-1][-1]["frame"])
            <= merge_radius
        ):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    output: List[Dict[str, object]] = []
    for candidate in [_combine_cluster(cluster) for cluster in clusters]:
        frame = int(candidate["frame"])
        if output and frame - int(output[-1]["frame"]) < minimum:
            output[-1] = _combine_cluster(
                [
                    *output[-1].get("support", [output[-1]]),
                    *candidate.get("support", [candidate]),
                ]
            )
        else:
            output.append(candidate)
    return output


def _evidence_signature(
    candidate: Mapping[str, object],
) -> Tuple[str, str]:
    if candidate.get("gripper_direction"):
        return ("gripper", str(candidate["gripper_direction"]))
    if candidate.get("current_phase"):
        return ("phase", str(candidate["current_phase"]))
    if "energy_valley" in str(candidate.get("evidence_type", "")):
        return ("pause", "still")
    return ("other", str(candidate.get("evidence_type", "")))


def group_bimanual_evidence(
    per_arm_evidence: Mapping[
        str,
        Sequence[Dict[str, object]],
    ],
    temporal: TemporalParameters,
) -> List[Dict[str, object]]:
    """Derive a coordination view without erasing per-arm boundaries."""

    left = [dict(item) for item in per_arm_evidence.get("left", [])]
    right = [dict(item) for item in per_arm_evidence.get("right", [])]
    used_right: set[int] = set()
    groups: List[Dict[str, object]] = []
    for left_item in left:
        available = [
            (index, right_item)
            for index, right_item in enumerate(right)
            if index not in used_right
            and abs(
                int(right_item["frame"]) - int(left_item["frame"])
            )
            <= temporal.cross_arm_sync_frames
        ]
        if not available:
            groups.append(left_item)
            continue
        right_index, right_item = min(
            available,
            key=lambda pair: abs(
                int(pair[1]["frame"]) - int(left_item["frame"])
            ),
        )
        left_signature = _evidence_signature(left_item)
        right_signature = _evidence_signature(right_item)
        compatible = left_signature == right_signature
        separation = abs(
            int(right_item["frame"]) - int(left_item["frame"])
        )
        if compatible or separation <= temporal.coordination_min_segment_frames:
            used_right.add(right_index)
            combined = _combine_cluster([left_item, right_item])
            combined["coordination_type"] = (
                f"synchronous_{left_signature[0]}"
                if compatible
                else "near_simultaneous_mixed_evidence"
            )
            combined["arm_time_delta_frames"] = separation
            groups.append(combined)
        else:
            groups.append(left_item)
    groups.extend(
        item
        for index, item in enumerate(right)
        if index not in used_right
    )
    for item in groups:
        item.setdefault("coordination_type", "single_arm")
    return sorted(groups, key=lambda item: int(item["frame"]))


def _direction_action(
    direction: str,
) -> str:
    return f"gripper_{direction}"


def _label_arm_segment(
    trajectory: np.ndarray,
    start: int,
    end: int,
    arm: str,
    config: SegmentationConfig,
) -> Tuple[str, Dict[str, float | list]]:
    summary = segment_motion_summary(trajectory, start, end, arm)
    gripper = arm_components(trajectory, arm)[2]
    overlapping = [
        event
        for event in gripper_events(
            gripper,
            threshold=config.gripper_change_threshold,
        )
        if int(event["start_frame"]) <= end
        and int(event["end_frame"]) >= start
    ]
    if overlapping:
        return _direction_action(str(overlapping[0]["direction"])), summary
    displacement = np.asarray(summary["displacement"], dtype=np.float64)
    vertical = normalize_direction(
        np.asarray(config.vertical_direction, dtype=np.float64)
    )
    vertical_displacement = float(np.dot(displacement, vertical))
    horizontal_displacement = float(
        np.linalg.norm(
            displacement - vertical_displacement * vertical
        )
    )
    translation = float(summary["translation_norm"])
    rotation = float(summary["rotation_path"])
    if translation < 0.005 and rotation < 0.03:
        label = "still"
    elif (
        abs(vertical_displacement) > max(0.015, 0.60 * translation)
        and abs(vertical_displacement) > horizontal_displacement
    ):
        label = "lift" if vertical_displacement > 0.0 else "lower"
    elif rotation / 0.10 > 4.5 * max(translation / 0.08, 1e-6):
        label = "turn"
    else:
        label = "move"
    summary = {
        **summary,
        "vertical_displacement": vertical_displacement,
        "horizontal_displacement": horizontal_displacement,
    }
    return label, summary


def _label_segments(
    trajectory: np.ndarray,
    segments: Sequence[Mapping[str, object]],
    config: SegmentationConfig,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        labels: Dict[str, str] = {}
        summaries: Dict[str, Dict[str, float | list]] = {}
        for arm in ARM_NAMES:
            labels[arm], summaries[arm] = _label_arm_segment(
                trajectory,
                start,
                end,
                arm,
                config,
            )
        moving = [arm for arm in ARM_NAMES if labels[arm] != "still"]
        if not moving:
            label = "still"
        elif len(moving) == 1:
            label = labels[moving[0]]
        elif labels["left"] == labels["right"]:
            label = labels["left"]
        else:
            label = f"{labels['left']}+{labels['right']}"
        output.append(
            {
                "segment_index": index,
                "start_frame": start,
                "end_frame": end,
                "frame_count": end - start + 1,
                "label": label,
                "left_action": labels["left"],
                "right_action": labels["right"],
                "arm_labels": labels,
                "kinematics": summaries,
            }
        )
    return output


def _arm_timeline(
    trajectory: np.ndarray,
    arm: str,
    evidence: Sequence[Mapping[str, object]],
    config: SegmentationConfig,
) -> List[Dict[str, object]]:
    raw_segments = segments_from_boundaries(
        [int(item["frame"]) for item in evidence],
        len(trajectory),
    )
    labelled = _label_segments(trajectory, raw_segments, config)
    prefix = "L" if arm == "left" else "R"
    return [
        {
            "primitive_id": f"{prefix}{index}",
            "arm": arm,
            "start_frame": int(segment["start_frame"]),
            "end_frame": int(segment["end_frame"]),
            "phase": str(segment[f"{arm}_action"]),
            "kinematics": segment["kinematics"][arm],
        }
        for index, segment in enumerate(labelled)
    ]


def _primitive_at_frame(
    timeline: Sequence[Mapping[str, object]],
    frame: int,
) -> Mapping[str, object] | None:
    for primitive in timeline:
        if (
            int(primitive["start_frame"])
            <= frame
            <= int(primitive["end_frame"])
        ):
            return primitive
    return None


def _attach_coordination_context(
    segments: Sequence[Dict[str, object]],
    timelines: Mapping[str, Sequence[Mapping[str, object]]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for segment in segments:
        record = dict(segment)
        center = (
            int(record["start_frame"]) + int(record["end_frame"])
        ) // 2
        left = _primitive_at_frame(timelines.get("left", []), center)
        right = _primitive_at_frame(timelines.get("right", []), center)
        left_action = str(record["left_action"])
        right_action = str(record["right_action"])
        if left_action == right_action == "still":
            relation = "both_still"
        elif left_action != "still" and right_action == "still":
            relation = "left_only"
        elif left_action == "still" and right_action != "still":
            relation = "right_only"
        elif left_action == right_action:
            relation = "dual_same_phase"
        else:
            relation = "dual_different_phase"
        record.update(
            {
                "coordination_relation": relation,
                "left_primitive_id": (
                    left.get("primitive_id") if left else None
                ),
                "right_primitive_id": (
                    right.get("primitive_id") if right else None
                ),
            }
        )
        output.append(record)
    return output


def _event_relations(
    diagnostics: Mapping[str, object],
    temporal: TemporalParameters,
    timestamps: np.ndarray,
) -> Dict[str, List[Dict[str, object]]]:
    events = {
        arm: diagnostics["strong"]["per_arm"][arm]["gripper_events"]
        for arm in ARM_NAMES
    }
    synchronous: List[Dict[str, object]] = []
    opposing: List[Dict[str, object]] = []
    sequential: List[Dict[str, object]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: List[Tuple[int, int, int]] = []
    for left_index, left_event in enumerate(events["left"]):
        for right_index, right_event in enumerate(events["right"]):
            if left_event["direction"] == right_event["direction"]:
                pairs.append(
                    (
                        abs(
                            int(right_event["onset_frame"])
                            - int(left_event["onset_frame"])
                        ),
                        left_index,
                        right_index,
                    )
                )
    for separation, left_index, right_index in sorted(pairs):
        if left_index in used_left or right_index in used_right:
            continue
        if separation > temporal.cross_arm_sync_frames:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        synchronous.append(
            {
                "left_frame": int(
                    events["left"][left_index]["onset_frame"]
                ),
                "right_frame": int(
                    events["right"][right_index]["onset_frame"]
                ),
                "direction": events["left"][left_index]["direction"],
                "separation_frames": separation,
            }
        )
    for left_index, left_event in enumerate(events["left"]):
        for right_index, right_event in enumerate(events["right"]):
            left_frame = int(left_event["onset_frame"])
            right_frame = int(right_event["onset_frame"])
            separation = abs(right_frame - left_frame)
            if separation > temporal.exchange_search_frames:
                continue
            record = {
                "left_frame": left_frame,
                "right_frame": right_frame,
                "left_direction": left_event["direction"],
                "right_direction": right_event["direction"],
                "separation_frames": separation,
                "separation_seconds": abs(
                    float(timestamps[right_frame])
                    - float(timestamps[left_frame])
                ),
            }
            if left_event["direction"] != right_event["direction"]:
                opposing.append(record)
            elif (
                left_index not in used_left
                and right_index not in used_right
                and separation > temporal.cross_arm_sync_frames
            ):
                sequential.append(record)
    return {
        "synchronous_dual_gripper_events": synchronous,
        "opposing_gripper_transition_candidates": opposing,
        "sequential_dual_gripper_events": sequential,
    }


def segment_episode(
    trajectory: np.ndarray,
    timestamps: np.ndarray | Sequence[float] | None = None,
    config: SegmentationConfig | None = None,
) -> Dict[str, object]:
    """Segment one successful dual-arm episode into atomic motion blocks."""

    values = _validate_trajectory(trajectory)
    selected_config = config or SegmentationConfig()
    timestamp_values, temporal = resolve_temporal_parameters(
        timestamps,
        len(values),
        selected_config,
    )
    features = extract_state_features(
        values,
        timestamp_values,
        temporal,
        selected_config,
    )
    strong, soft, diagnostics = _collect_state_evidence(
        values,
        features,
        timestamp_values,
        temporal,
        selected_config,
    )
    refined_soft = _refine_soft_evidence(
        soft,
        features,
        timestamp_values,
        temporal,
    )
    per_arm_evidence: Dict[str, List[Dict[str, object]]] = {}
    arm_timelines: Dict[str, List[Dict[str, object]]] = {}
    for arm in ARM_NAMES:
        candidates = [
            candidate
            for candidate in [*strong, *refined_soft]
            if arm in candidate.get("source_arms", [])
        ]
        evidence = _filter_evidence(
            candidates,
            len(values),
            temporal,
        )
        for item in evidence:
            item["time_seconds"] = float(
                timestamp_values[int(item["frame"])]
            )
        per_arm_evidence[arm] = evidence
        arm_timelines[arm] = _arm_timeline(
            values,
            arm,
            evidence,
            selected_config,
        )
    coordination = group_bimanual_evidence(
        per_arm_evidence,
        temporal,
    )
    coordination = _filter_evidence(
        coordination,
        len(values),
        temporal,
        min_segment_frames=temporal.coordination_min_segment_frames,
        merge_radius_frames=0,
    )
    for item in coordination:
        item["time_seconds"] = float(
            timestamp_values[int(item["frame"])]
        )
    boundaries = [int(item["frame"]) for item in coordination]
    segments = _label_segments(
        values,
        segments_from_boundaries(boundaries, len(values)),
        selected_config,
    )
    segments = _attach_coordination_context(segments, arm_timelines)
    return {
        "schema": "atomic_episode_segmentation",
        "method": "factorized_bimanual_state_segmentation",
        "frame_count": int(len(values)),
        "state_layout": list(STATE_LAYOUT),
        "decision_input_policy": dict(STATE_ONLY_DECISION_POLICY),
        "temporal_parameters": asdict(temporal),
        "boundaries": boundaries,
        "boundary_evidence": coordination,
        "per_arm_boundaries": {
            arm: [int(item["frame"]) for item in evidence]
            for arm, evidence in per_arm_evidence.items()
        },
        "per_arm_boundary_evidence": per_arm_evidence,
        "arm_timelines": arm_timelines,
        "segments": segments,
        "diagnostics": {
            **diagnostics,
            "onset_policy": (
                "earliest sustained successor trend near each phase proposal"
            ),
            "bimanual_policy": {
                "per_arm_boundaries_are_primary": True,
                "joint_timeline_is_derived": True,
                "global_active_arm_is_not_used": True,
            },
            **_event_relations(
                diagnostics,
                temporal,
                timestamp_values,
            ),
        },
        "_state_features": features,
    }


def serializable_segmentation(
    result: Mapping[str, object],
) -> Dict[str, object]:
    """Drop internal NumPy feature arrays before JSON serialization."""

    return {
        key: value
        for key, value in result.items()
        if not str(key).startswith("_")
    }
