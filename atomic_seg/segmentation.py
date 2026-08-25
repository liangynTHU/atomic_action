"""Shared primitives for the segmentation-method evolution.

V1--V5 are the original geometry/event/phase line.  V6--V8 live in
``state_segmentation.py`` and reuse the interval, evidence, and labelling
helpers defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .features import (
    ARM_OFFSETS,
    arm_components,
    contiguous_runs,
    gripper_events,
    local_motion_statistics,
    normalize_direction,
    segment_motion_summary,
)
from .geometry import huber, quaternion_angle, slerp


ARM_NAMES = ("left", "right")
KINEMATIC_PHASES = {"still", "move", "turn", "lift", "lower"}
GRIPPER_PHASES = {"close_gripper", "open_gripper"}


@dataclass(frozen=True)
class SegmenterConfig:
    min_segment_frames: int = 5
    max_segments: int = 20
    # Characteristic deviation / tolerance scales for reconstruction.
    position_error_scale: float = 0.01
    rotation_error_scale: float = 0.05
    gripper_error_scale: float = 0.10
    pause_threshold: float = 0.5
    pause_min_frames: int = 3
    boundary_merge_radius: int = 5
    phase_window: int = 9
    phase_min_frames: int = 5
    phase_bridge_frames: int = 2
    phase_search_after_gripper: bool = True
    gripper_change_threshold: float = 0.01
    gripper_trend_threshold: float = 0.05
    still_translation_threshold: float = 0.003
    still_path_threshold: float = 0.008
    still_rotation_threshold: float = 0.02
    vertical_ratio_threshold: float = 0.62
    vertical_displacement_threshold: float = 0.005
    rotation_dominance_ratio: float = 1.6
    rotation_window_scale: float = 0.05
    translation_window_scale: float = 0.03
    # World z is gravity-aligned in the current RoboTwin dataset.
    vertical_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Final method uses both arms; this mode is retained only as an ablation.
    arm_mode: str = "joint"
    # V6+ resolves all temporal thresholds from timestamps.  The frame-based
    # fields above remain unchanged so V1--V5 stay reproducible.
    default_sample_period_seconds: float = 0.02
    min_segment_seconds: float = 0.10
    pause_min_seconds: float = 0.06
    boundary_merge_seconds: float = 0.10
    phase_window_seconds: float = 0.18
    phase_min_seconds: float = 0.10
    phase_bridge_seconds: float = 0.04
    onset_search_seconds: float = 0.18
    onset_confirm_seconds: float = 0.06
    cross_arm_sync_seconds: float = 0.08
    coordination_min_segment_seconds: float = 0.04
    exchange_search_seconds: float = 1.50
    gripper_pre_window_seconds: float = 0.60
    gripper_post_window_seconds: float = 0.40
    # Physical-rate scales used by the state-only V6+ energy.  These are
    # characteristic scales, not semantic task thresholds.
    translation_speed_scale: float = 0.15
    rotation_speed_scale: float = 1.00
    gripper_rate_scale: float = 1.00
    state_pause_energy_threshold: float = 0.50
    gripper_event_rate_threshold: float = 0.50
    gripper_event_min_delta: float = 0.08
    onset_translation_speed: float = 0.08
    onset_vertical_speed: float = 0.05
    onset_rotation_speed: float = 0.40


def geometric_primitives_from_knots(
    knots: Sequence[int], frame_count: int
) -> List[Dict[str, int]]:
    """Create inclusive geometric primitives [kappa_{j-1}, kappa_j].

    Adjacent primitives share their geometric knot. These are approximation
    intervals, not exported frame-ownership intervals.
    """

    if frame_count <= 0:
        return []
    ordered = sorted({0, frame_count - 1, *(int(value) for value in knots)})
    if ordered[0] != 0 or ordered[-1] != frame_count - 1:
        raise ValueError("Geometric knots must span frames 0 through T-1.")
    return [
        {"start_knot": start, "end_knot": end}
        for start, end in zip(ordered[:-1], ordered[1:])
        if end > start
    ]


def segments_from_boundaries(
    boundaries: Sequence[int], frame_count: int
) -> List[Dict[str, int]]:
    """Export disjoint frame slices using half-open ownership boundaries.

    A boundary b means the previous exported segment owns frames through b-1
    and the next segment starts at frame b. This implementation convention is
    deliberately separate from the shared-knot geometric formulation.
    """

    cuts = sorted(
        {int(value) for value in boundaries if 0 < int(value) < frame_count}
    )
    starts = [0] + cuts
    ends = [value - 1 for value in cuts] + [frame_count - 1]
    return [
        {"start_frame": int(start), "end_frame": int(end)}
        for start, end in zip(starts, ends)
        if end >= start
    ]


def boundaries_from_segments(
    segments: Sequence[Dict[str, int]],
) -> List[int]:
    return [int(segment["end_frame"]) + 1 for segment in segments[:-1]]


def _linear_segment_cost(
    trajectory: np.ndarray,
    start_knot: int,
    end_knot: int,
    config: SegmenterConfig,
    include_gripper: bool = True,
) -> float:
    """Piecewise-geodesic reconstruction cost on inclusive knot endpoints."""

    count = end_knot - start_knot + 1
    fractions = np.linspace(0.0, 1.0, count)
    total = 0.0
    for offset in (0, 8):
        position = trajectory[:, offset : offset + 3]
        quaternion = trajectory[:, offset + 3 : offset + 7]
        gripper = trajectory[:, offset + 7]
        position_hat = (
            position[start_knot]
            + fractions[:, None]
            * (position[end_knot] - position[start_knot])
        )
        quaternion_hat = slerp(
            quaternion[start_knot], quaternion[end_knot], fractions
        )
        position_error = (
            np.linalg.norm(
                position[start_knot : end_knot + 1] - position_hat, axis=1
            )
            / config.position_error_scale
        )
        rotation_error = (
            quaternion_angle(
                quaternion[start_knot : end_knot + 1], quaternion_hat
            )
            / config.rotation_error_scale
        )
        total += float(np.sum(huber(position_error) + huber(rotation_error)))
        if include_gripper:
            gripper_hat = gripper[start_knot] + fractions * (
                gripper[end_knot] - gripper[start_knot]
            )
            gripper_error = (
                np.abs(
                    gripper[start_knot : end_knot + 1] - gripper_hat
                )
                / config.gripper_error_scale
            )
            total += float(np.sum(huber(gripper_error)))
    return total


def _geometric_cost_matrix(
    trajectory: np.ndarray, config: SegmenterConfig, include_gripper: bool
) -> np.ndarray:
    frame_count = len(trajectory)
    costs = np.full((frame_count, frame_count), np.inf)
    min_span = max(1, config.min_segment_frames - 1)
    for start_knot in range(frame_count - 1):
        for end_knot in range(
            start_knot + min_span, frame_count
        ):
            costs[start_knot, end_knot] = _linear_segment_cost(
                trajectory,
                start_knot,
                end_knot,
                config,
                include_gripper=include_gripper,
            )
    return costs


def _fixed_k_geometric_dynamic_programming(
    costs: np.ndarray, config: SegmenterConfig
) -> List[Tuple[float, List[int]]]:
    """Globally optimize reconstruction for every fixed number of primitives."""

    frame_count = costs.shape[0]
    min_span = max(1, config.min_segment_frames - 1)
    max_primitives = min(
        config.max_segments, (frame_count - 1) // min_span
    )
    objective = np.full((max_primitives + 1, frame_count), np.inf)
    previous = np.full(
        (max_primitives + 1, frame_count), -1, dtype=np.int64
    )
    objective[0, 0] = 0.0
    for primitive_count in range(1, max_primitives + 1):
        minimum_end = primitive_count * min_span
        for end_knot in range(minimum_end, frame_count):
            minimum_start = (primitive_count - 1) * min_span
            maximum_start = end_knot - min_span
            for start_knot in range(minimum_start, maximum_start + 1):
                candidate = (
                    objective[primitive_count - 1, start_knot]
                    + costs[start_knot, end_knot]
                )
                if candidate < objective[primitive_count, end_knot]:
                    objective[primitive_count, end_knot] = candidate
                    previous[primitive_count, end_knot] = start_knot
    solutions: List[Tuple[float, List[int]]] = []
    for primitive_count in range(1, max_primitives + 1):
        end_knot = frame_count - 1
        knots = [end_knot]
        count = primitive_count
        while count > 0:
            start_knot = int(previous[count, end_knot])
            if start_knot < 0:
                knots = []
                break
            knots.append(start_knot)
            end_knot = start_knot
            count -= 1
        if knots:
            solutions.append(
                (
                    float(objective[primitive_count, frame_count - 1]),
                    knots[::-1],
                )
            )
    return solutions


def _largest_curvature_choice(
    solutions: Sequence[Tuple[float, List[int]]],
) -> int:
    """Conservative, parameter-light elbow for the reconstruction curve."""

    errors = np.asarray([solution[0] for solution in solutions], dtype=np.float64)
    if len(errors) <= 2:
        return len(errors)
    log_error = np.log(np.maximum(errors, 1e-12))
    curvature = np.diff(log_error, n=2)
    usable = max(1, min(len(curvature), len(errors) // 2))
    return int(np.argmax(curvature[:usable]) + 2)


def _baseline_result(
    trajectory: np.ndarray,
    config: SegmenterConfig,
    include_gripper: bool,
    method: str,
) -> Dict[str, object]:
    costs = _geometric_cost_matrix(
        trajectory, config, include_gripper=include_gripper
    )
    solutions = _fixed_k_geometric_dynamic_programming(costs, config)
    chosen_count = _largest_curvature_choice(solutions)
    error, knots = solutions[chosen_count - 1]
    # Slicing ownership uses the internal shared knots as the next slice start.
    slicing_boundaries = knots[1:-1]
    return {
        "segments": segments_from_boundaries(
            slicing_boundaries, len(trajectory)
        ),
        "selection": {
            "method": method,
            "chosen_primitives": chosen_count,
            "chosen_error": error,
            "geometric_knots": knots,
            "geometric_primitives": geometric_primitives_from_knots(
                knots, len(trajectory)
            ),
            "frame_ownership_convention": (
                "half-open slicing: [b_j, b_{j+1}); geometric primitives "
                "remain inclusive and share knots"
            ),
            "error_curve": [float(item[0]) for item in solutions],
        },
    }


def version1_piecewise_geodesic_dp(
    trajectory: np.ndarray, config: SegmenterConfig
) -> Dict[str, object]:
    """Piecewise-geodesic trajectory approximation baseline."""

    return _baseline_result(
        trajectory,
        config,
        include_gripper=False,
        method="piecewise_geodesic_fixed_k_dp",
    )


def version2_geometry_aware_reconstruction_dp(
    trajectory: np.ndarray, config: SegmenterConfig
) -> Dict[str, object]:
    """Geometry-aware reconstruction baseline with gripper and robust loss."""

    return _baseline_result(
        trajectory,
        config,
        include_gripper=True,
        method="geometry_aware_robust_reconstruction_dp",
    )


# Backward-compatible function aliases for callers outside this repository.
version1_linear_dp = version1_piecewise_geodesic_dp
version2_robust_se3_gripper_dp = (
    version2_geometry_aware_reconstruction_dp
)


def _velocity_change_score(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        return np.zeros(len(features) + 1)
    difference = np.linalg.norm(np.diff(features, axis=0), axis=1)
    score = np.r_[0.0, difference, 0.0]
    if len(score) >= 5:
        score = np.convolve(score, np.ones(5) / 5.0, mode="same")
    return score


def _simple_merge_indices(
    boundaries: Iterable[int], score: np.ndarray, radius: int
) -> List[int]:
    result: List[int] = []
    for boundary in sorted({int(value) for value in boundaries}):
        if result and boundary - result[-1] < radius:
            if score[boundary] > score[result[-1]]:
                result[-1] = boundary
        else:
            result.append(boundary)
    return result


def version3_motion_change_points(
    trajectory: np.ndarray,
    feature_bundle: Dict[str, object],
    config: SegmenterConfig,
) -> Dict[str, object]:
    """Derivative-domain motion change-point ablation."""

    velocity = np.asarray(feature_bundle["dual_velocity_features"])
    score = _velocity_change_score(velocity)
    if len(score) < len(trajectory):
        score = np.pad(score, (0, len(trajectory) - len(score)))
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) + 1e-9
    threshold = median + 2.5 * mad
    candidates = []
    for index in range(2, len(trajectory) - 2):
        if (
            score[index] >= score[index - 1]
            and score[index] >= score[index + 1]
            and score[index] >= threshold
        ):
            candidates.append(index)
    candidates = _simple_merge_indices(
        candidates, score, max(config.min_segment_frames, 4)
    )
    boundaries = [
        boundary
        for boundary in candidates
        if boundary >= config.min_segment_frames
        and len(trajectory) - boundary >= config.min_segment_frames
    ]
    evidence = [
        {
            "frame": boundary,
            "source_arm": "both",
            "evidence_type": "motion_change_point",
            "evidence_class": "ablation",
            "strength": float(score[boundary]),
        }
        for boundary in boundaries
    ]
    return {
        "segments": segments_from_boundaries(boundaries, len(trajectory)),
        "selection": {
            "method": "motion_change_point_ablation",
            "threshold": threshold,
            "candidate_boundaries": candidates,
            "chosen_boundaries": boundaries,
            "boundary_evidence": evidence,
            "chosen_segments": len(boundaries) + 1,
        },
    }


version3_velocity_change_points = version3_motion_change_points


def _candidate(
    frame: int,
    source_arm: str,
    evidence_type: str,
    evidence_class: str,
    strength: float,
    **extra: object,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "frame": int(frame),
        "source_arm": source_arm,
        "source_arms": [source_arm],
        "evidence_type": evidence_type,
        "evidence_types": [evidence_type],
        "evidence_class": evidence_class,
        "strength": float(strength),
        "supported_by_both_arms": False,
    }
    record.update(extra)
    return record


def _pause_evidence_for_arm(
    feature_bundle: Dict[str, object],
    arm: str,
    config: SegmenterConfig,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    energy = np.asarray(feature_bundle["arms"][arm]["energy_smooth"])
    pause_runs = [
        run
        for run in contiguous_runs(energy < config.pause_threshold)
        if run[1] - run[0] + 1 >= config.pause_min_frames
    ]
    candidates: List[Dict[str, object]] = []
    for start, end in pause_runs:
        if start <= 2 or end >= len(energy) - 3:
            continue
        frame = start + int(np.argmin(energy[start : end + 1]))
        depth = max(
            0.0,
            (config.pause_threshold - float(energy[frame]))
            / max(config.pause_threshold, 1e-12),
        )
        candidates.append(
            _candidate(
                frame,
                arm,
                "sustained_motion_energy_valley",
                "strong",
                2.0 + depth,
                pause_run=[int(start), int(end)],
                energy=float(energy[frame]),
            )
        )
    return candidates, pause_runs


def _gripper_evidence_for_arm(
    feature_bundle: Dict[str, object],
    arm: str,
    config: SegmenterConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    arm_features = feature_bundle["arms"][arm]
    energy = np.asarray(arm_features["energy_smooth"])
    events = gripper_events(
        np.asarray(arm_features["gripper"]),
        threshold=config.gripper_change_threshold,
    )
    candidates: List[Dict[str, object]] = []
    for event_index, event in enumerate(events):
        kind = str(event["kind"])
        delta = abs(float(event["delta"]))
        event_start = int(event["start_frame"])
        event_end = int(event["end_frame"])
        before_start = max(1, event_start - 30)
        before_stop = event_start + 1
        before = before_start + int(
            np.argmin(energy[before_start:before_stop])
        )
        after_start = event_end
        after_stop = min(len(energy), after_start + 21)
        after = after_start + int(
            np.argmin(energy[after_start:after_stop])
        )
        candidates.extend(
            [
                _candidate(
                    before,
                    arm,
                    f"pre_{kind}_stabilization_valley",
                    "strong",
                    2.5
                    + max(
                        0.0,
                        config.pause_threshold - float(energy[before]),
                    ),
                    event_index=event_index,
                    related_gripper_event=kind,
                    event_interval=[event_start, event_end],
                    gripper_delta=float(event["delta"]),
                    energy=float(energy[before]),
                ),
                _candidate(
                    after,
                    arm,
                    f"post_{kind}_stabilization_valley",
                    "strong",
                    2.5
                    + max(
                        0.0,
                        config.pause_threshold - float(energy[after]),
                    ),
                    event_index=event_index,
                    related_gripper_event=kind,
                    event_interval=[event_start, event_end],
                    gripper_delta=float(event["delta"]),
                    energy=float(energy[after]),
                ),
            ]
        )
    return candidates, events


def _selected_arms(
    feature_bundle: Dict[str, object], config: SegmenterConfig
) -> List[str]:
    if config.arm_mode == "active":
        return [str(feature_bundle["active_arm"])]
    if config.arm_mode != "joint":
        raise ValueError("arm_mode must be either 'joint' or 'active'.")
    return list(ARM_NAMES)


def _collect_strong_evidence(
    feature_bundle: Dict[str, object], config: SegmenterConfig
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {
        "arm_mode": config.arm_mode,
        "per_arm": {},
    }
    for arm in _selected_arms(feature_bundle, config):
        pause_candidates, pause_runs = _pause_evidence_for_arm(
            feature_bundle, arm, config
        )
        gripper_candidates, events = _gripper_evidence_for_arm(
            feature_bundle, arm, config
        )
        candidates.extend(pause_candidates)
        candidates.extend(gripper_candidates)
        diagnostics["per_arm"][arm] = {
            "pause_runs": [list(run) for run in pause_runs],
            "gripper_events": events,
            "strong_candidate_count": len(pause_candidates)
            + len(gripper_candidates),
        }
    return candidates, diagnostics


def _phase_from_statistics(
    statistics: Dict[str, float | list], config: SegmenterConfig
) -> str:
    gripper_trend = float(statistics["gripper_trend"])
    translation = float(statistics["translation_norm"])
    path_length = float(statistics["path_length"])
    rotation = float(statistics["accumulated_rotation"])
    vertical_displacement = float(statistics["vertical_displacement"])
    vertical_ratio = float(statistics["vertical_ratio"])
    if gripper_trend <= -config.gripper_trend_threshold:
        return "close_gripper"
    if gripper_trend >= config.gripper_trend_threshold:
        return "open_gripper"
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
    rotation_score = rotation / config.rotation_window_scale
    translation_score = translation / config.translation_window_scale
    if rotation_score > config.rotation_dominance_ratio * max(
        translation_score, 1e-6
    ):
        return "turn"
    return "move"


def local_phase_sequence(
    trajectory: np.ndarray,
    arm: str,
    config: SegmenterConfig,
) -> Tuple[List[str], List[Dict[str, float | list]]]:
    vertical_direction = normalize_direction(
        np.asarray(config.vertical_direction, dtype=np.float64)
    )
    statistics = [
        local_motion_statistics(
            trajectory,
            arm,
            frame,
            config.phase_window,
            vertical_direction,
        )
        for frame in range(len(trajectory))
    ]
    labels = [
        _phase_from_statistics(frame_statistics, config)
        for frame_statistics in statistics
    ]
    return labels, statistics


def _raw_label_runs(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    runs: List[Dict[str, object]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append(
                {
                    "start_frame": start,
                    "end_frame": index - 1,
                    "phase": labels[start],
                }
            )
            start = index
    return runs


def _absorb_short_phase_runs(
    runs: List[Dict[str, object]], min_frames: int
) -> List[Dict[str, object]]:
    """Remove isolated A-B-A flicker and absorb remaining short runs."""

    runs = [dict(run) for run in runs]
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


def persistent_phase_runs(
    labels: Sequence[str], config: SegmenterConfig
) -> List[Dict[str, object]]:
    return _absorb_short_phase_runs(
        _raw_label_runs(labels), config.phase_min_frames
    )


def _meaningful_phase_transition(previous: str, current: str) -> bool:
    if previous == current:
        return False
    if previous in GRIPPER_PHASES or current in GRIPPER_PHASES:
        # Gripper transitions are already represented as strong evidence.
        return False
    if previous == "still" and current == "still":
        return False
    return previous in KINEMATIC_PHASES and current in KINEMATIC_PHASES


def _collect_soft_evidence(
    trajectory: np.ndarray,
    feature_bundle: Dict[str, object],
    config: SegmenterConfig,
    strong_candidates: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    diagnostics: Dict[str, object] = {"per_arm": {}}
    strong_frames_by_arm = {
        arm: [
            int(candidate["frame"])
            for candidate in strong_candidates
            if arm in candidate.get("source_arms", [])
        ]
        for arm in ARM_NAMES
    }
    gripper_end_by_arm = {arm: 0 for arm in ARM_NAMES}
    for arm in ARM_NAMES:
        events = gripper_events(
            np.asarray(feature_bundle["arms"][arm]["gripper"]),
            threshold=config.gripper_change_threshold,
        )
        gripper_end_by_arm[arm] = max(
            [int(event["end_frame"]) for event in events] or [0]
        )
    for arm in _selected_arms(feature_bundle, config):
        labels, statistics = local_phase_sequence(
            trajectory, arm, config
        )
        runs = persistent_phase_runs(labels, config)
        arm_candidates: List[Dict[str, object]] = []
        for previous, current in zip(runs, runs[1:]):
            previous_phase = str(previous["phase"])
            current_phase = str(current["phase"])
            boundary = int(current["start_frame"])
            if not _meaningful_phase_transition(
                previous_phase, current_phase
            ):
                continue
            if (
                config.phase_search_after_gripper
                and boundary <= gripper_end_by_arm[arm]
            ):
                continue
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
                previous_duration < config.phase_min_frames
                or current_duration < config.phase_min_frames
            ):
                continue
            if min(
                [
                    abs(boundary - frame)
                    for frame in strong_frames_by_arm[arm]
                ]
                or [10**9]
            ) < config.boundary_merge_radius:
                # It will not add a new boundary, but the final merge records
                # phase support if it falls in the temporal cluster.
                pass
            previous_stats = statistics[
                min(int(previous["end_frame"]), len(statistics) - 1)
            ]
            current_stats = statistics[
                min(int(current["start_frame"]), len(statistics) - 1)
            ]
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
                "persistent_phase_transition",
                "soft",
                1.0 + min(contrast, 2.0),
                previous_phase=previous_phase,
                current_phase=current_phase,
                previous_duration=previous_duration,
                current_duration=current_duration,
            )
            candidates.append(candidate)
            arm_candidates.append(candidate)
        diagnostics["per_arm"][arm] = {
            "persistent_phase_runs": runs,
            "soft_candidate_count": len(arm_candidates),
        }
    return candidates, diagnostics


def _combine_cluster(
    cluster: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    strong = [
        candidate
        for candidate in cluster
        if candidate["evidence_class"] == "strong"
    ]
    priority_pool = strong or list(cluster)
    representative = max(
        priority_pool,
        key=lambda candidate: (
            float(candidate["strength"]),
            -int(candidate["frame"]),
        ),
    )
    weights = np.asarray(
        [max(float(item["strength"]), 1e-6) for item in priority_pool]
    )
    frames = np.asarray(
        [int(item["frame"]) for item in priority_pool], dtype=np.float64
    )
    frame = int(round(float(np.average(frames, weights=weights))))
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
        "evidence_type": str(representative["evidence_type"]),
        "evidence_types": evidence_types,
        "evidence_class": "strong" if "strong" in classes else "soft",
        "strength": float(max(float(item["strength"]) for item in cluster)),
        "supported_by_both_arms": set(source_arms) == set(ARM_NAMES),
        "support_count": len(cluster),
        "support": [dict(candidate) for candidate in cluster],
    }


def temporal_merge_evidence(
    candidates: Sequence[Dict[str, object]], radius: int
) -> List[Dict[str, object]]:
    """Temporal NMS with strong-over-soft priority and provenance fusion."""

    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: int(item["frame"]))
    clusters: List[List[Dict[str, object]]] = [[dict(ordered[0])]]
    for candidate in ordered[1:]:
        if (
            int(candidate["frame"])
            - int(clusters[-1][-1]["frame"])
            <= radius
        ):
            clusters[-1].append(dict(candidate))
        else:
            clusters.append([dict(candidate)])
    return [_combine_cluster(cluster) for cluster in clusters]


def filter_merge_evidence(
    candidates: Sequence[Dict[str, object]],
    frame_count: int,
    config: SegmenterConfig,
) -> List[Dict[str, object]]:
    """FilterMerge(B_strong union B_soft) from the final formulation."""

    valid = [
        dict(candidate)
        for candidate in candidates
        if config.min_segment_frames
        <= int(candidate["frame"])
        <= frame_count - config.min_segment_frames
    ]
    merged = temporal_merge_evidence(
        valid, config.boundary_merge_radius
    )
    # Enforce minimum segment duration again after representative relocation.
    output: List[Dict[str, object]] = []
    for candidate in sorted(merged, key=lambda item: int(item["frame"])):
        frame = int(candidate["frame"])
        if frame < config.min_segment_frames:
            continue
        if frame_count - frame < config.min_segment_frames:
            continue
        if (
            output
            and frame - int(output[-1]["frame"])
            < config.min_segment_frames
        ):
            winner = _combine_cluster(
                [*output[-1].get("support", [output[-1]]), *candidate.get(
                    "support", [candidate]
                )]
            )
            output[-1] = winner
        else:
            output.append(candidate)
    return output


def version4_strong_event_segmentation(
    trajectory: np.ndarray,
    feature_bundle: Dict[str, object],
    config: SegmenterConfig,
) -> Dict[str, object]:
    """Event-aware kinematic segmentation using strong evidence only."""

    strong_candidates, diagnostics = _collect_strong_evidence(
        feature_bundle, config
    )
    fused = filter_merge_evidence(
        strong_candidates, len(trajectory), config
    )
    boundaries = [int(candidate["frame"]) for candidate in fused]
    return {
        "segments": segments_from_boundaries(boundaries, len(trajectory)),
        "selection": {
            "method": "strong_event_evidence_ablation",
            "arm_mode": config.arm_mode,
            "active_arm_ablation_choice": feature_bundle["active_arm"],
            "strong_candidates": strong_candidates,
            "soft_candidates": [],
            "boundary_evidence": fused,
            "chosen_boundaries": boundaries,
            "chosen_segments": len(boundaries) + 1,
            "diagnostics": diagnostics,
        },
    }


version4_event_aware = version4_strong_event_segmentation


def version5_event_phase_atomic_motion(
    trajectory: np.ndarray,
    feature_bundle: Dict[str, object],
    config: SegmenterConfig,
) -> Dict[str, object]:
    """Final event- and phase-aware atomic motion segmentation framework."""

    strong_candidates, strong_diagnostics = _collect_strong_evidence(
        feature_bundle, config
    )
    soft_candidates, phase_diagnostics = _collect_soft_evidence(
        trajectory, feature_bundle, config, strong_candidates
    )
    fused = filter_merge_evidence(
        [*strong_candidates, *soft_candidates],
        len(trajectory),
        config,
    )
    boundaries = [int(candidate["frame"]) for candidate in fused]
    return {
        "segments": segments_from_boundaries(boundaries, len(trajectory)),
        "selection": {
            "method": "event_and_phase_aware_atomic_motion_segmentation",
            "arm_mode": config.arm_mode,
            "active_arm_ablation_choice": feature_bundle["active_arm"],
            "strong_candidate_frames": sorted(
                {
                    int(candidate["frame"])
                    for candidate in strong_candidates
                }
            ),
            "soft_candidate_frames": sorted(
                {
                    int(candidate["frame"])
                    for candidate in soft_candidates
                }
            ),
            "strong_boundary_set": [
                int(candidate["frame"])
                for candidate in fused
                if candidate["evidence_class"] == "strong"
            ],
            "soft_boundary_set": [
                int(candidate["frame"])
                for candidate in fused
                if candidate["evidence_class"] == "soft"
            ],
            "strong_candidates": strong_candidates,
            "soft_candidates": soft_candidates,
            "boundary_evidence": fused,
            "chosen_boundaries": boundaries,
            "chosen_segments": len(boundaries) + 1,
            "filter_merge": {
                "minimum_segment_frames": config.min_segment_frames,
                "temporal_merge_radius": config.boundary_merge_radius,
                "phase_min_frames": config.phase_min_frames,
                "strong_over_soft_priority": True,
                "isolated_phase_transition_filter": True,
            },
            "diagnostics": {
                "strong_events": strong_diagnostics,
                "persistent_phases": phase_diagnostics,
            },
        },
    }


version5_hybrid_semantic = version5_event_phase_atomic_motion


def _segment_label_for_arm(
    trajectory: np.ndarray,
    start: int,
    end: int,
    arm: str,
    config: SegmenterConfig,
) -> Tuple[str, Dict[str, float | list]]:
    summary = segment_motion_summary(trajectory, start, end, arm)
    gripper = arm_components(trajectory, arm)[2]
    events = gripper_events(
        gripper, threshold=config.gripper_change_threshold
    )
    overlap_events = [
        event
        for event in events
        if int(event["start_frame"]) <= end
        and int(event["end_frame"]) >= start
    ]
    if overlap_events:
        label = str(overlap_events[0]["kind"])
        if (
            label == "close_gripper"
            and float(summary["translation_norm"]) > 0.03
        ):
            label = "grasp"
        return label, summary
    displacement = np.asarray(summary["displacement"], dtype=np.float64)
    vertical = normalize_direction(
        np.asarray(config.vertical_direction, dtype=np.float64)
    )
    vertical_displacement = float(np.dot(displacement, vertical))
    horizontal_displacement = float(
        np.linalg.norm(displacement - vertical_displacement * vertical)
    )
    translation = float(summary["translation_norm"])
    rotation = float(summary["rotation_path"])
    if translation < 0.005 and rotation < 0.03:
        label = "still"
    elif (
        abs(vertical_displacement)
        > max(0.015, 0.60 * translation)
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


def label_segments(
    trajectory: np.ndarray,
    segments: List[Dict[str, object]],
    feature_bundle: Dict[str, object],
    config: SegmenterConfig,
) -> List[Dict[str, object]]:
    """Attach per-arm kinematic labels; no full semantic-action claim."""

    labelled: List[Dict[str, object]] = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        arm_labels: Dict[str, str] = {}
        arm_summaries: Dict[str, Dict[str, float | list]] = {}
        for arm in ARM_NAMES:
            arm_labels[arm], arm_summaries[arm] = _segment_label_for_arm(
                trajectory, start, end, arm, config
            )
        moving_arms = [
            arm for arm in ARM_NAMES if arm_labels[arm] != "still"
        ]
        if len(moving_arms) == 1:
            label = arm_labels[moving_arms[0]]
        elif len(moving_arms) == 2:
            label = (
                arm_labels["left"]
                if arm_labels["left"] == arm_labels["right"]
                else f"{arm_labels['left']}+{arm_labels['right']}"
            )
        else:
            label = "still"
        record = dict(segment)
        record["label"] = label
        record["left_action"] = arm_labels["left"]
        record["right_action"] = arm_labels["right"]
        record["arm_labels"] = arm_labels
        record["kinematics"] = arm_summaries
        labelled.append(record)
    return labelled
