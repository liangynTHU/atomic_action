"""Kinematic feature extraction for 16D dual-arm end-pose trajectories.

The physical orientation belongs to SO(3). Unit quaternions in the input are
only its numerical representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .geometry import quaternion_angle, quaternion_log_delta


ARM_OFFSETS = {"left": 0, "right": 8}


@dataclass(frozen=True)
class FeatureConfig:
    smooth_window: int = 5
    translation_step_scale: float = 0.003
    rotation_step_scale: float = 0.02
    gripper_step_scale: float = 0.05


def arm_components(trajectory: np.ndarray, arm: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = ARM_OFFSETS[arm]
    return (
        trajectory[:, offset : offset + 3],
        trajectory[:, offset + 3 : offset + 7],
        trajectory[:, offset + 7],
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def per_arm_features(
    trajectory: np.ndarray, arm: str, config: FeatureConfig
) -> Dict[str, np.ndarray]:
    position, quaternion, gripper = arm_components(trajectory, arm)
    delta_position = np.diff(position, axis=0)
    delta_rotation_vector = quaternion_log_delta(quaternion[:-1], quaternion[1:])
    delta_rotation = np.linalg.norm(delta_rotation_vector, axis=1)
    delta_gripper = np.diff(gripper)
    translation_step = np.linalg.norm(delta_position, axis=1)
    energy_step = (
        translation_step / config.translation_step_scale
        + delta_rotation / config.rotation_step_scale
        + np.abs(delta_gripper) / config.gripper_step_scale
    )
    energy = np.r_[0.0, energy_step]
    energy_smooth = moving_average(energy, config.smooth_window)
    velocity_features = np.column_stack(
        [
            delta_position / config.translation_step_scale,
            delta_rotation_vector / config.rotation_step_scale,
            delta_gripper / config.gripper_step_scale,
        ]
    )
    for column in range(velocity_features.shape[1]):
        velocity_features[:, column] = moving_average(
            velocity_features[:, column], config.smooth_window
        )
    return {
        "position": position,
        "quaternion": quaternion,
        "gripper": gripper,
        "delta_position": delta_position,
        "delta_rotation": delta_rotation,
        "delta_rotation_vector": delta_rotation_vector,
        "delta_gripper": delta_gripper,
        "translation_step": translation_step,
        "energy": energy,
        "energy_smooth": energy_smooth,
        "velocity_features": velocity_features,
    }


def extract_features(
    trajectory: np.ndarray, config: FeatureConfig | None = None
) -> Dict[str, object]:
    config = config or FeatureConfig()
    arms = {
        arm: per_arm_features(trajectory, arm, config)
        for arm in ("left", "right")
    }
    total_energy = {
        arm: float(np.sum(arms[arm]["energy_smooth"])) for arm in arms
    }
    active_arm = max(total_energy, key=total_energy.get)
    dual_velocity = np.column_stack(
        [arms["left"]["velocity_features"], arms["right"]["velocity_features"]]
    )
    return {
        "config": config,
        "arms": arms,
        "active_arm": active_arm,
        "total_energy": total_energy,
        "dual_velocity_features": dual_velocity,
    }


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Direction vector must have non-zero norm.")
    return direction / norm


def local_motion_statistics(
    trajectory: np.ndarray,
    arm: str,
    frame: int,
    window: int,
    vertical_direction: np.ndarray,
) -> Dict[str, float | list]:
    """Compute windowed motion statistics around one frame.

    The window captures net displacement, path length, vertical/horizontal
    decomposition, accumulated rotation, and gripper trend. These statistics
    are used for phase classification; no single-frame derivative determines a
    phase label.
    """

    half = max(1, int(window) // 2)
    start = max(0, int(frame) - half)
    end = min(len(trajectory) - 1, int(frame) + half)
    position, quaternion, gripper = arm_components(trajectory, arm)
    displacement = position[end] - position[start]
    vertical = normalize_direction(vertical_direction)
    vertical_displacement = float(np.dot(displacement, vertical))
    horizontal_vector = displacement - vertical_displacement * vertical
    horizontal_displacement = float(np.linalg.norm(horizontal_vector))
    translation_norm = float(np.linalg.norm(displacement))
    path_length = float(
        np.sum(
            np.linalg.norm(
                np.diff(position[start : end + 1], axis=0), axis=1
            )
        )
    )
    accumulated_rotation = float(
        np.sum(
            quaternion_angle(
                quaternion[start + 1 : end + 1],
                quaternion[start:end],
            )
        )
    )
    gripper_trend = float(gripper[end] - gripper[start])
    return {
        "start_frame": start,
        "end_frame": end,
        "displacement": displacement.tolist(),
        "translation_norm": translation_norm,
        "path_length": path_length,
        "vertical_displacement": vertical_displacement,
        "vertical_ratio": abs(vertical_displacement)
        / max(translation_norm, 1e-12),
        "horizontal_displacement": horizontal_displacement,
        "horizontal_ratio": horizontal_displacement
        / max(translation_norm, 1e-12),
        "accumulated_rotation": accumulated_rotation,
        "gripper_trend": gripper_trend,
    }


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and bool(mask[index])
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def gripper_events(gripper: np.ndarray, threshold: float = 0.01) -> List[Dict[str, object]]:
    changes = np.diff(np.asarray(gripper, dtype=np.float64))
    moving = np.abs(changes) >= threshold
    events: List[Dict[str, object]] = []
    for start, end in contiguous_runs(moving):
        # changes[start:end+1] maps state frame start -> end+1.
        state_start = start
        state_end = end + 1
        delta = float(gripper[state_end] - gripper[state_start])
        events.append(
            {
                "start_frame": int(state_start),
                "end_frame": int(state_end),
                "delta": delta,
                "kind": "close_gripper" if delta < 0.0 else "open_gripper",
            }
        )
    return events


def segment_motion_summary(
    trajectory: np.ndarray, start: int, end: int, arm: str
) -> Dict[str, float | list]:
    position, quaternion, gripper = arm_components(trajectory, arm)
    displacement = position[end] - position[start]
    path_length = float(
        np.sum(np.linalg.norm(np.diff(position[start : end + 1], axis=0), axis=1))
    )
    rotation = float(
        np.sum(
            quaternion_angle(
                quaternion[start + 1 : end + 1], quaternion[start:end]
            )
        )
    )
    return {
        "displacement": displacement.tolist(),
        "translation_norm": float(np.linalg.norm(displacement)),
        "path_length": path_length,
        "rotation_path": rotation,
        "gripper_delta": float(gripper[end] - gripper[start]),
        "mean_gripper": float(np.mean(gripper[start : end + 1])),
    }
