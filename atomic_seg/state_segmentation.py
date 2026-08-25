"""Timestamp-aware, state-only segmentation for V6--V8.

The decision path in this module intentionally consumes only robot state and
timestamps.  Images, videos, task text, and ``action_config`` are downstream
annotation context; they never move a V6--V8 boundary.

Version progression:

* V6: physical-time state evidence, with the V5-style joint evidence fusion.
* V7: V6 plus successor-trend-onset boundary refinement.
* V8: V7 plus factorized left/right primitive streams and an explicit
  bimanual coordination view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .features import (
    ARM_OFFSETS,
    arm_components,
    contiguous_runs,
    local_motion_statistics,
    normalize_direction,
)
from .geometry import quaternion_log_delta
from .segmentation import (
    KINEMATIC_PHASES,
    SegmenterConfig,
    filter_merge_evidence,
    label_segments,
    persistent_phase_runs,
    segments_from_boundaries,
)


ARM_NAMES = ("left", "right")


@dataclass(frozen=True)
class TemporalParameters:
    """Frame counts derived from physical durations and the episode clock."""

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
        "action_config",
        "task_text",
        "instruction",
        "object_label",
    ],
    "downstream_annotation_context": [
        "image",
        "video",
        "action_config",
        "task_text",
        "data_config",
    ],
    "boundary_lock_after_state_segmentation": True,
    "note": (
        "Visual or metadata context may generate captions/instructions after "
        "segmentation, but it cannot alter V6--V8 state-derived boundaries."
    ),
}


def _timestamps_or_default(
    timestamps: np.ndarray | Sequence[float] | None,
    frame_count: int,
    default_period: float,
) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float64)
    if timestamps is None:
        return np.arange(frame_count, dtype=np.float64) * float(
            default_period
        )
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(values) != frame_count:
        raise ValueError(
            "timestamps must have the same length as the trajectory "
            f"({len(values)} != {frame_count})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps must be finite")
    if frame_count > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return values


def _seconds_to_frames(
    seconds: float, sample_period: float, minimum: int = 1
) -> int:
    return max(minimum, int(round(float(seconds) / sample_period)))


def resolve_temporal_parameters(
    timestamps: np.ndarray | Sequence[float] | None,
    frame_count: int,
    config: SegmenterConfig,
) -> Tuple[np.ndarray, TemporalParameters]:
    """Resolve all V6+ durations from timestamps instead of fixed FPS."""

    values = _timestamps_or_default(
        timestamps, frame_count, config.default_sample_period_seconds
    )
    if frame_count > 1:
        sample_period = float(np.median(np.diff(values)))
    else:
        sample_period = float(config.default_sample_period_seconds)
    if sample_period <= 0.0:
        raise ValueError("sample period must be positive")
    parameters = TemporalParameters(
        sample_period_seconds=sample_period,
        fps_estimate=1.0 / sample_period,
        min_segment_frames=_seconds_to_frames(
            config.min_segment_seconds, sample_period
        ),
        pause_min_frames=_seconds_to_frames(
            config.pause_min_seconds, sample_period
        ),
        boundary_merge_frames=_seconds_to_frames(
            config.boundary_merge_seconds, sample_period
        ),
        phase_window_frames=_seconds_to_frames(
            config.phase_window_seconds, sample_period, minimum=3
        ),
        phase_min_frames=_seconds_to_frames(
            config.phase_min_seconds, sample_period
        ),
        phase_bridge_frames=_seconds_to_frames(
            config.phase_bridge_seconds, sample_period
        ),
        onset_search_frames=_seconds_to_frames(
            config.onset_search_seconds, sample_period
        ),
        onset_confirm_frames=_seconds_to_frames(
            config.onset_confirm_seconds, sample_period, minimum=2
        ),
        cross_arm_sync_frames=_seconds_to_frames(
            config.cross_arm_sync_seconds, sample_period
        ),
        coordination_min_segment_frames=_seconds_to_frames(
            config.coordination_min_segment_seconds,
            sample_period,
            minimum=1,
        ),
        exchange_search_frames=_seconds_to_frames(
            config.exchange_search_seconds, sample_period
        ),
        gripper_pre_window_frames=_seconds_to_frames(
            config.gripper_pre_window_seconds, sample_period
        ),
        gripper_post_window_frames=_seconds_to_frames(
            config.gripper_post_window_seconds, sample_period
        ),
        energy_smooth_frames=_seconds_to_frames(
            0.10, sample_period, minimum=1
        ),
    )
    return values, parameters


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) <= 1:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _normalize_gripper(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize amplitude without assigning open/closed semantics."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values.copy(), {"low": 0.0, "high": 0.0, "span": 0.0}
    low, high = np.percentile(values, [5.0, 95.0])
    span = float(high - low)
    if span <= 1e-9:
        normalized = np.zeros_like(values)
    else:
        normalized = np.clip((values - low) / span, 0.0, 1.0)
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
    config: SegmenterConfig,
) -> Dict[str, np.ndarray]:
    vertical = normalize_direction(vertical_direction)
    vertical_speed = linear_velocity @ vertical
    horizontal_velocity = (
        linear_velocity - vertical_speed[:, None] * vertical[None, :]
    )
    horizontal_speed = np.linalg.norm(horizontal_velocity, axis=1)
    translation_speed = np.linalg.norm(linear_velocity, axis=1)
    move_score = horizontal_speed / max(
        config.onset_translation_speed, 1e-12
    )
    lift_score = np.maximum(vertical_speed, 0.0) / max(
        config.onset_vertical_speed, 1e-12
    )
    lower_score = np.maximum(-vertical_speed, 0.0) / max(
        config.onset_vertical_speed, 1e-12
    )
    rotation_base = angular_speed / max(
        config.onset_rotation_speed, 1e-12
    )
    translation_competition = translation_speed / max(
        config.onset_translation_speed, 1e-12
    )
    turn_score = np.maximum(
        rotation_base - 0.35 * translation_competition, 0.0
    )
    still_score = 1.0 / (
        1.0
        + energy / max(config.state_pause_energy_threshold, 1e-12)
    )
    return {
        "still": still_score,
        "move": move_score,
        "turn": turn_score,
        "lift": lift_score,
        "lower": lower_score,
    }


def extract_state_features(
    trajectory: np.ndarray,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmenterConfig,
) -> Dict[str, object]:
    """Physical-rate features used by V6--V8."""

    frame_count = len(trajectory)
    dt = np.diff(timestamps)
    arms: Dict[str, Dict[str, object]] = {}
    vertical_direction = np.asarray(
        config.vertical_direction, dtype=np.float64
    )
    for arm in ARM_NAMES:
        position, quaternion, gripper_raw = arm_components(trajectory, arm)
        gripper_unit, gripper_normalization = _normalize_gripper(
            gripper_raw
        )
        linear_velocity = np.zeros((frame_count, 3), dtype=np.float64)
        angular_velocity = np.zeros((frame_count, 3), dtype=np.float64)
        gripper_rate = np.zeros(frame_count, dtype=np.float64)
        if frame_count > 1:
            linear_velocity[1:] = (
                np.diff(position, axis=0) / dt[:, None]
            )
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
            + angular_speed / max(config.rotation_speed_scale, 1e-12)
            + np.abs(gripper_rate)
            / max(config.gripper_rate_scale, 1e-12)
        )
        energy_smooth = _moving_average(
            energy, temporal.energy_smooth_frames
        )
        phase_scores = _phase_activity_scores(
            linear_velocity,
            angular_speed,
            energy,
            vertical_direction,
            config,
        )
        arms[arm] = {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper_raw,
            "gripper_unit": gripper_unit,
            "gripper_normalization": gripper_normalization,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "gripper_rate": gripper_rate,
            "translation_speed": translation_speed,
            "angular_speed": angular_speed,
            "energy": energy,
            "energy_smooth": energy_smooth,
            "phase_scores": phase_scores,
        }
    return {
        "timestamps": timestamps,
        "sample_period_seconds": temporal.sample_period_seconds,
        "fps_estimate": temporal.fps_estimate,
        "arms": arms,
    }


def _state_candidate(
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
    false_runs = contiguous_runs(~output)
    for start, end in false_runs:
        length = end - start + 1
        if (
            length <= max_gap
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
    config: SegmenterConfig,
) -> List[Dict[str, object]]:
    """Direction-neutral gripper events in normalized amplitude units."""

    events: List[Dict[str, object]] = []
    for direction, direction_sign in (
        ("increase", 1.0),
        ("decrease", -1.0),
    ):
        active = (
            direction_sign * np.asarray(gripper_rate)
            >= config.gripper_event_rate_threshold
        )
        opposite_active = (
            direction_sign * np.asarray(gripper_rate)
            <= -config.gripper_event_rate_threshold
        )
        active = _merge_short_boolean_gaps(
            active,
            temporal.phase_bridge_frames,
            blocking_mask=opposite_active,
        )
        for start, end in contiguous_runs(active):
            if start <= 0:
                baseline = 0
            else:
                baseline = start - 1
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
    arm_bundle: Dict[str, object],
    arm: str,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmenterConfig,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    energy = np.asarray(arm_bundle["energy_smooth"])
    pause_runs = [
        run
        for run in contiguous_runs(
            energy < config.state_pause_energy_threshold
        )
        if run[1] - run[0] + 1 >= temporal.pause_min_frames
    ]
    candidates: List[Dict[str, object]] = []
    for start, end in pause_runs:
        if start <= 1 or end >= len(energy) - 2:
            continue
        frame = start + int(np.argmin(energy[start : end + 1]))
        depth = max(
            0.0,
            (
                config.state_pause_energy_threshold
                - float(energy[frame])
            )
            / max(config.state_pause_energy_threshold, 1e-12),
        )
        candidates.append(
            _state_candidate(
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
    return candidates, pause_runs


def _gripper_evidence(
    arm_bundle: Dict[str, object],
    arm: str,
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmenterConfig,
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
            "semantic_gripper_state": (
                "unresolved_without_polarity_metadata"
            ),
        }
        candidates.append(
            _state_candidate(
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
            )
        )
        candidates.append(
            _state_candidate(
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
            )
        )
    return candidates, events


def _phase_from_statistics(
    statistics: Dict[str, float | list], config: SegmenterConfig
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
        config.translation_window_scale, 1e-12
    )
    if rotation_score > config.rotation_dominance_ratio * max(
        translation_score, 1e-6
    ):
        return "turn"
    return "move"


def _phase_runs_for_arm(
    trajectory: np.ndarray,
    arm: str,
    temporal: TemporalParameters,
    config: SegmenterConfig,
) -> Tuple[List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    vertical = normalize_direction(
        np.asarray(config.vertical_direction, dtype=np.float64)
    )
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
        _phase_from_statistics(frame_statistics, config)
        for frame_statistics in statistics
    ]
    frame_config = replace(
        config, phase_min_frames=temporal.phase_min_frames
    )
    runs = persistent_phase_runs(labels, frame_config)
    return labels, runs, statistics


def _phase_evidence(
    trajectory: np.ndarray,
    state_bundle: Dict[str, object],
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmenterConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {"per_arm": {}}
    for arm in ARM_NAMES:
        labels, runs, statistics = _phase_runs_for_arm(
            trajectory, arm, temporal, config
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
            if (
                previous_duration < temporal.phase_min_frames
                or current_duration < temporal.phase_min_frames
            ):
                continue
            previous_stats = statistics[int(previous["end_frame"])]
            current_stats = statistics[int(current["start_frame"])]
            contrast = abs(
                float(current_stats["vertical_ratio"])
                - float(previous_stats["vertical_ratio"])
            ) + abs(
                float(current_stats["accumulated_rotation"])
                - float(previous_stats["accumulated_rotation"])
            ) / max(config.rotation_window_scale, 1e-12)
            candidate = _state_candidate(
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


def collect_state_evidence(
    trajectory: np.ndarray,
    state_bundle: Dict[str, object],
    timestamps: np.ndarray,
    temporal: TemporalParameters,
    config: SegmenterConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    strong: List[Dict[str, object]] = []
    strong_diagnostics: Dict[str, object] = {"per_arm": {}}
    for arm in ARM_NAMES:
        arm_bundle = state_bundle["arms"][arm]
        pause_candidates, pause_runs = _pause_evidence(
            arm_bundle, arm, timestamps, temporal, config
        )
        gripper_candidates, gripper_events = _gripper_evidence(
            arm_bundle, arm, timestamps, temporal, config
        )
        strong.extend(pause_candidates)
        strong.extend(gripper_candidates)
        strong_diagnostics["per_arm"][arm] = {
            "pause_runs": [list(run) for run in pause_runs],
            "gripper_events": gripper_events,
            "strong_candidate_count": len(pause_candidates)
            + len(gripper_candidates),
        }
    soft, phase_diagnostics = _phase_evidence(
        trajectory, state_bundle, timestamps, temporal, config
    )
    return strong, soft, {
        "strong": strong_diagnostics,
        "phase": phase_diagnostics,
    }


def _frame_config(
    config: SegmenterConfig, temporal: TemporalParameters
) -> SegmenterConfig:
    return replace(
        config,
        min_segment_frames=temporal.min_segment_frames,
        pause_min_frames=temporal.pause_min_frames,
        boundary_merge_radius=temporal.boundary_merge_frames,
        phase_window=temporal.phase_window_frames,
        phase_min_frames=temporal.phase_min_frames,
        phase_bridge_frames=temporal.phase_bridge_frames,
    )


def _with_boundary_times(
    evidence: Sequence[Dict[str, object]], timestamps: np.ndarray
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for item in evidence:
        record = dict(item)
        frame = int(record["frame"])
        record["time_seconds"] = float(timestamps[frame])
        output.append(record)
    return output


def _base_selection(
    method: str,
    temporal: TemporalParameters,
    strong: Sequence[Dict[str, object]],
    soft: Sequence[Dict[str, object]],
    fused: Sequence[Dict[str, object]],
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    boundaries = [int(candidate["frame"]) for candidate in fused]
    return {
        "method": method,
        "decision_input_policy": dict(STATE_ONLY_DECISION_POLICY),
        "temporal_parameters": asdict(temporal),
        "strong_candidates": [dict(item) for item in strong],
        "soft_candidates": [dict(item) for item in soft],
        "boundary_evidence": [dict(item) for item in fused],
        "chosen_boundaries": boundaries,
        "chosen_segments": len(boundaries) + 1,
        "diagnostics": diagnostics,
    }


def version6_time_normalized_state_segmentation(
    trajectory: np.ndarray,
    timestamps: np.ndarray | Sequence[float] | None,
    config: SegmenterConfig,
) -> Dict[str, object]:
    """V6: state-only evidence with physical-time thresholds."""

    timestamp_values, temporal = resolve_temporal_parameters(
        timestamps, len(trajectory), config
    )
    state_bundle = extract_state_features(
        trajectory, timestamp_values, temporal, config
    )
    strong, soft, diagnostics = collect_state_evidence(
        trajectory, state_bundle, timestamp_values, temporal, config
    )
    fused = filter_merge_evidence(
        [*strong, *soft],
        len(trajectory),
        _frame_config(config, temporal),
    )
    fused = _with_boundary_times(fused, timestamp_values)
    selection = _base_selection(
        "time_normalized_state_evidence_segmentation",
        temporal,
        strong,
        soft,
        fused,
        diagnostics,
    )
    return {
        "segments": segments_from_boundaries(
            selection["chosen_boundaries"], len(trajectory)
        ),
        "selection": selection,
        "_state_bundle": state_bundle,
    }


def successor_trend_onset_from_scores(
    previous_scores: np.ndarray,
    current_scores: np.ndarray,
    nominal_frame: int,
    search_frames: int,
    confirm_frames: int,
    *,
    low_threshold: float = 0.25,
    sustained_threshold: float = 0.55,
) -> Tuple[int, Dict[str, object]]:
    """Choose the start of the successor trend, not the score crossover.

    This directly handles the overlapping-transition case where the successor
    activity rises while the predecessor activity decays.  A candidate onset
    must have a low-but-real current score, sustained future support, and a
    rise relative to the immediately preceding window.  If no onset is
    observable inside the search window, the persistent-run boundary is kept.
    """

    previous = np.asarray(previous_scores, dtype=np.float64)
    current = np.asarray(current_scores, dtype=np.float64)
    frame_count = min(len(previous), len(current))
    nominal = int(np.clip(nominal_frame, 0, max(frame_count - 1, 0)))
    lo = max(1, nominal - int(search_frames))
    hi = min(frame_count - 1, nominal + int(search_frames))
    confirm = max(2, int(confirm_frames))
    chosen = nominal
    found = False
    for frame in range(lo, hi + 1):
        future = current[frame : min(frame_count, frame + confirm)]
        if len(future) < 2:
            continue
        before = current[max(0, frame - confirm) : frame]
        future_mean = float(np.mean(future))
        before_mean = float(np.mean(before)) if len(before) else 0.0
        sustained_fraction = float(
            np.mean(future >= low_threshold)
        )
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


def refine_soft_evidence_to_onsets(
    soft_candidates: Sequence[Dict[str, object]],
    state_bundle: Dict[str, object],
    timestamps: np.ndarray,
    temporal: TemporalParameters,
) -> List[Dict[str, object]]:
    refined: List[Dict[str, object]] = []
    for candidate in soft_candidates:
        record = dict(candidate)
        arm = str(record["source_arm"])
        previous_phase = str(record.get("previous_phase", "still"))
        current_phase = str(record.get("current_phase", "move"))
        scores = state_bundle["arms"][arm]["phase_scores"]
        previous_scores = np.asarray(scores[previous_phase])
        current_scores = np.asarray(scores[current_phase])
        nominal = int(record["frame"])
        onset, onset_diagnostics = successor_trend_onset_from_scores(
            previous_scores,
            current_scores,
            nominal,
            temporal.onset_search_frames,
            temporal.onset_confirm_frames,
            low_threshold=0.45 if current_phase == "still" else 0.25,
            sustained_threshold=0.55,
        )
        record["original_frame"] = nominal
        record["frame"] = onset
        record["time_seconds"] = float(timestamps[onset])
        record["original_evidence_type"] = record["evidence_type"]
        record["evidence_type"] = "successor_trend_onset"
        record["evidence_types"] = sorted(
            {
                *record.get("evidence_types", []),
                "successor_trend_onset",
            }
        )
        record["boundary_policy"] = (
            "earliest_persistent_successor_trend_onset"
        )
        record["onset_refinement"] = onset_diagnostics
        refined.append(record)
    return refined


def _combine_state_cluster(
    cluster: Sequence[Dict[str, object]],
    *,
    prefer_successor_onset: bool,
) -> Dict[str, object]:
    strong = [
        candidate
        for candidate in cluster
        if candidate["evidence_class"] == "strong"
    ]
    priority_pool = strong or list(cluster)
    if strong:
        representative = max(
            priority_pool,
            key=lambda candidate: (
                float(candidate["strength"]),
                -int(candidate["frame"]),
            ),
        )
        weights = np.asarray(
            [
                max(float(candidate["strength"]), 1e-6)
                for candidate in priority_pool
            ]
        )
        frames = np.asarray(
            [int(candidate["frame"]) for candidate in priority_pool],
            dtype=np.float64,
        )
        frame = int(round(float(np.average(frames, weights=weights))))
    elif prefer_successor_onset:
        representative = min(
            priority_pool, key=lambda candidate: int(candidate["frame"])
        )
        frame = int(representative["frame"])
    else:
        representative = max(
            priority_pool, key=lambda candidate: float(candidate["strength"])
        )
        frame = int(representative["frame"])
    source_arms = sorted(
        {
            str(arm)
            for candidate in cluster
            for arm in candidate.get(
                "source_arms", [candidate["source_arm"]]
            )
        }
    )
    evidence_types = sorted(
        {
            str(evidence_type)
            for candidate in cluster
            for evidence_type in candidate.get(
                "evidence_types", [candidate["evidence_type"]]
            )
        }
    )
    classes = {
        str(candidate["evidence_class"]) for candidate in cluster
    }
    return {
        **dict(representative),
        "frame": frame,
        "source_arm": (
            source_arms[0] if len(source_arms) == 1 else "both"
        ),
        "source_arms": source_arms,
        "evidence_types": evidence_types,
        "evidence_class": "strong" if "strong" in classes else "soft",
        "strength": float(
            max(float(candidate["strength"]) for candidate in cluster)
        ),
        "supported_by_both_arms": set(source_arms) == set(ARM_NAMES),
        "support_count": len(cluster),
        "support": [dict(candidate) for candidate in cluster],
    }


def _onset_aware_filter(
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
    merged = [
        _combine_state_cluster(
            cluster, prefer_successor_onset=True
        )
        for cluster in clusters
    ]
    output: List[Dict[str, object]] = []
    for candidate in merged:
        frame = int(candidate["frame"])
        if output and frame - int(output[-1]["frame"]) < minimum:
            output[-1] = _combine_state_cluster(
                [
                    *output[-1].get("support", [output[-1]]),
                    *candidate.get("support", [candidate]),
                ],
                prefer_successor_onset=True,
            )
        else:
            output.append(candidate)
    return output


def version7_onset_aware_state_segmentation(
    trajectory: np.ndarray,
    timestamps: np.ndarray | Sequence[float] | None,
    config: SegmenterConfig,
) -> Dict[str, object]:
    """V7: V6 plus successor-onset boundary placement."""

    timestamp_values, temporal = resolve_temporal_parameters(
        timestamps, len(trajectory), config
    )
    state_bundle = extract_state_features(
        trajectory, timestamp_values, temporal, config
    )
    strong, soft, diagnostics = collect_state_evidence(
        trajectory, state_bundle, timestamp_values, temporal, config
    )
    onset_soft = refine_soft_evidence_to_onsets(
        soft, state_bundle, timestamp_values, temporal
    )
    fused = _onset_aware_filter(
        [*strong, *onset_soft], len(trajectory), temporal
    )
    fused = _with_boundary_times(fused, timestamp_values)
    diagnostics = {
        **diagnostics,
        "onset_policy": {
            "rule": (
                "When successor activity rises while predecessor activity "
                "falls, cut at the earliest sustained successor onset, not "
                "at the score crossover or predecessor completion."
            ),
            "search_frames": temporal.onset_search_frames,
            "confirm_frames": temporal.onset_confirm_frames,
        },
    }
    selection = _base_selection(
        "successor_onset_aware_state_segmentation",
        temporal,
        strong,
        onset_soft,
        fused,
        diagnostics,
    )
    return {
        "segments": segments_from_boundaries(
            selection["chosen_boundaries"], len(trajectory)
        ),
        "selection": selection,
        "_state_bundle": state_bundle,
    }


def _evidence_signature(
    candidate: Dict[str, object]
) -> Tuple[str, str]:
    if candidate.get("gripper_direction"):
        return ("gripper", str(candidate["gripper_direction"]))
    if candidate.get("current_phase"):
        return ("phase", str(candidate["current_phase"]))
    if "energy_valley" in str(candidate.get("evidence_type", "")):
        return ("pause", "still")
    return ("other", str(candidate.get("evidence_type", "")))


def _cross_arm_compatibility(
    left: Dict[str, object], right: Dict[str, object]
) -> Tuple[bool, str]:
    left_signature = _evidence_signature(left)
    right_signature = _evidence_signature(right)
    if left_signature == right_signature:
        return True, f"synchronous_{left_signature[0]}"
    if left_signature[0] == right_signature[0] == "phase":
        return False, "different_phase_onsets"
    if left_signature[0] == right_signature[0] == "gripper":
        return False, "opposing_gripper_trends"
    return False, "mixed_cross_arm_evidence"


def group_bimanual_evidence(
    per_arm_evidence: Dict[str, Sequence[Dict[str, object]]],
    temporal: TemporalParameters,
) -> List[Dict[str, object]]:
    """Create a joint coordination view without erasing per-arm boundaries."""

    left = [dict(item) for item in per_arm_evidence.get("left", [])]
    right = [dict(item) for item in per_arm_evidence.get("right", [])]
    used_right: set[int] = set()
    groups: List[List[Dict[str, object]]] = []
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
            groups.append([left_item])
            continue
        right_index, right_item = min(
            available,
            key=lambda pair: abs(
                int(pair[1]["frame"]) - int(left_item["frame"])
            ),
        )
        compatible, relation = _cross_arm_compatibility(
            left_item, right_item
        )
        separation = abs(
            int(right_item["frame"]) - int(left_item["frame"])
        )
        if (
            compatible
            or separation <= temporal.coordination_min_segment_frames
        ):
            used_right.add(right_index)
            combined = _combine_state_cluster(
                [left_item, right_item],
                prefer_successor_onset=True,
            )
            combined["coordination_type"] = (
                relation
                if compatible
                else "near_simultaneous_mixed_evidence"
            )
            combined["arm_time_delta_frames"] = separation
            groups.append([combined])
        else:
            groups.append([left_item])
    for index, right_item in enumerate(right):
        if index not in used_right:
            groups.append([right_item])
    output: List[Dict[str, object]] = []
    for group in groups:
        candidate = dict(group[0])
        candidate.setdefault("coordination_type", "single_arm")
        output.append(candidate)
    return sorted(output, key=lambda item: int(item["frame"]))


def _arm_timeline(
    trajectory: np.ndarray,
    arm: str,
    evidence: Sequence[Dict[str, object]],
    config: SegmenterConfig,
) -> List[Dict[str, object]]:
    boundaries = [int(item["frame"]) for item in evidence]
    raw_segments = segments_from_boundaries(boundaries, len(trajectory))
    labelled = neutralize_gripper_semantics(
        label_segments(trajectory, raw_segments, {}, config)
    )
    output: List[Dict[str, object]] = []
    prefix = "L" if arm == "left" else "R"
    for index, segment in enumerate(labelled):
        output.append(
            {
                "primitive_id": f"{prefix}{index}",
                "arm": arm,
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "phase": str(segment[f"{arm}_action"]),
                "kinematics": segment["kinematics"][arm],
            }
        )
    return output


def neutralize_gripper_semantics(
    segments: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Use direction-only labels for V6+ until polarity is annotated.

    The historical labeller assumes negative gripper delta means close.  That
    assumption is valid for the current RoboTwin sample but is not part of the
    state-only V6+ boundary contract.  We therefore expose direction labels in
    V6+ results and leave open/close/grasp semantics to downstream metadata or
    visual annotation.
    """

    mapping = {
        "close_gripper": "gripper_decrease",
        "grasp": "gripper_decrease",
        "open_gripper": "gripper_increase",
    }
    output: List[Dict[str, object]] = []
    for segment in segments:
        record = dict(segment)
        left = mapping.get(
            str(record.get("left_action", "still")),
            str(record.get("left_action", "still")),
        )
        right = mapping.get(
            str(record.get("right_action", "still")),
            str(record.get("right_action", "still")),
        )
        record["left_action"] = left
        record["right_action"] = right
        record["arm_labels"] = {"left": left, "right": right}
        moving = [
            arm
            for arm, label in (("left", left), ("right", right))
            if label != "still"
        ]
        if len(moving) == 0:
            record["label"] = "still"
        elif len(moving) == 1:
            record["label"] = left if moving[0] == "left" else right
        elif left == right:
            record["label"] = left
        else:
            record["label"] = f"{left}+{right}"
        output.append(record)
    return output


def _event_relations(
    diagnostics: Dict[str, object],
    temporal: TemporalParameters,
    timestamps: np.ndarray,
) -> Dict[str, List[Dict[str, object]]]:
    per_arm_events = {
        arm: diagnostics["strong"]["per_arm"][arm]["gripper_events"]
        for arm in ARM_NAMES
    }
    exchange: List[Dict[str, object]] = []
    sequential_same_direction: List[Dict[str, object]] = []
    synchronous_pairs: List[Dict[str, object]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    same_direction_pairs: List[Tuple[int, int, int]] = []
    for left_index, left_event in enumerate(per_arm_events["left"]):
        for right_index, right_event in enumerate(per_arm_events["right"]):
            if left_event["direction"] != right_event["direction"]:
                continue
            separation = abs(
                int(right_event["onset_frame"])
                - int(left_event["onset_frame"])
            )
            same_direction_pairs.append(
                (separation, left_index, right_index)
            )
    for separation, left_index, right_index in sorted(
        same_direction_pairs
    ):
        if left_index in used_left or right_index in used_right:
            continue
        if separation > temporal.cross_arm_sync_frames:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        left_event = per_arm_events["left"][left_index]
        right_event = per_arm_events["right"][right_index]
        synchronous_pairs.append(
            {
                "left_frame": int(left_event["onset_frame"]),
                "right_frame": int(right_event["onset_frame"]),
                "direction": left_event["direction"],
                "separation_frames": separation,
                "relation": "synchronous_dual_gripper_event",
            }
        )

    for left_index, left_event in enumerate(per_arm_events["left"]):
        for right_index, right_event in enumerate(per_arm_events["right"]):
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
                record["relation"] = "opposing_gripper_transition_candidate"
                record["semantic_role_status"] = (
                    "donor_recipient_unresolved_without_gripper_polarity_"
                    "and_contact_semantics"
                )
                exchange.append(record)
            elif (
                left_index not in used_left
                and right_index not in used_right
                and separation > temporal.cross_arm_sync_frames
            ):
                record["relation"] = (
                    "sequential_same_direction_dual_gripper_events"
                )
                sequential_same_direction.append(record)
    return {
        "synchronous_dual_gripper_events": synchronous_pairs,
        "cross_arm_exchange_candidates": exchange,
        "sequential_dual_gripper_events": sequential_same_direction,
    }


def version8_factorized_bimanual_state_segmentation(
    trajectory: np.ndarray,
    timestamps: np.ndarray | Sequence[float] | None,
    config: SegmenterConfig,
) -> Dict[str, object]:
    """V8: independent arm primitives plus a derived coordination timeline."""

    timestamp_values, temporal = resolve_temporal_parameters(
        timestamps, len(trajectory), config
    )
    state_bundle = extract_state_features(
        trajectory, timestamp_values, temporal, config
    )
    strong, soft, diagnostics = collect_state_evidence(
        trajectory, state_bundle, timestamp_values, temporal, config
    )
    onset_soft = refine_soft_evidence_to_onsets(
        soft, state_bundle, timestamp_values, temporal
    )
    per_arm_evidence: Dict[str, List[Dict[str, object]]] = {}
    arm_timelines: Dict[str, List[Dict[str, object]]] = {}
    for arm in ARM_NAMES:
        arm_candidates = [
            candidate
            for candidate in [*strong, *onset_soft]
            if arm in candidate.get("source_arms", [])
        ]
        fused = _onset_aware_filter(
            arm_candidates, len(trajectory), temporal
        )
        fused = _with_boundary_times(fused, timestamp_values)
        per_arm_evidence[arm] = fused
        arm_timelines[arm] = _arm_timeline(
            trajectory, arm, fused, config
        )
    coordination = group_bimanual_evidence(
        per_arm_evidence, temporal
    )
    coordination = _onset_aware_filter(
        coordination,
        len(trajectory),
        temporal,
        min_segment_frames=temporal.coordination_min_segment_frames,
        merge_radius_frames=0,
    )
    coordination = _with_boundary_times(coordination, timestamp_values)
    boundaries = [int(item["frame"]) for item in coordination]
    event_relations = _event_relations(
        diagnostics, temporal, timestamp_values
    )
    diagnostics = {
        **diagnostics,
        "onset_policy": {
            "rule": (
                "Each arm is refined to the earliest sustained successor "
                "onset before cross-arm coordination is derived."
            )
        },
        "bimanual_policy": {
            "per_arm_boundaries_are_primary": True,
            "joint_timeline_is_derived": True,
            "simultaneous_compatible_events_may_share_boundary": True,
            "staggered_or_different_events_remain_separate_when_duration_allows": True,
            "global_active_arm_is_not_used": True,
        },
        **event_relations,
    }
    selection = _base_selection(
        "factorized_bimanual_onset_aware_state_segmentation",
        temporal,
        strong,
        onset_soft,
        coordination,
        diagnostics,
    )
    selection["per_arm_boundaries"] = {
        arm: [int(item["frame"]) for item in evidence]
        for arm, evidence in per_arm_evidence.items()
    }
    selection["per_arm_boundary_evidence"] = per_arm_evidence
    selection["arm_timelines"] = arm_timelines
    selection["coordination_boundary_evidence"] = coordination
    selection["coordination_boundaries"] = boundaries
    return {
        "segments": segments_from_boundaries(boundaries, len(trajectory)),
        "selection": selection,
        "_state_bundle": state_bundle,
    }


def _primitive_at_frame(
    timeline: Sequence[Dict[str, object]], frame: int
) -> Dict[str, object] | None:
    for primitive in timeline:
        if (
            int(primitive["start_frame"])
            <= frame
            <= int(primitive["end_frame"])
        ):
            return primitive
    return None


def attach_bimanual_segment_context(
    segments: Sequence[Dict[str, object]],
    selection: Dict[str, object],
) -> List[Dict[str, object]]:
    """Attach V8 arm primitive references and coordination labels."""

    timelines = selection.get("arm_timelines", {})
    output: List[Dict[str, object]] = []
    for segment in segments:
        record = dict(segment)
        center = (
            int(record["start_frame"]) + int(record["end_frame"])
        ) // 2
        left_primitive = _primitive_at_frame(
            timelines.get("left", []), center
        )
        right_primitive = _primitive_at_frame(
            timelines.get("right", []), center
        )
        left_action = str(record.get("left_action", "still"))
        right_action = str(record.get("right_action", "still"))
        if left_action == "still" and right_action == "still":
            relation = "both_still"
        elif left_action != "still" and right_action == "still":
            relation = "left_only"
        elif left_action == "still" and right_action != "still":
            relation = "right_only"
        elif left_action == right_action:
            relation = "dual_same_phase"
        else:
            relation = "dual_different_phase"
        record["coordination_relation"] = relation
        record["left_primitive_id"] = (
            left_primitive.get("primitive_id")
            if left_primitive
            else None
        )
        record["right_primitive_id"] = (
            right_primitive.get("primitive_id")
            if right_primitive
            else None
        )
        output.append(record)
    return output
