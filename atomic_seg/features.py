"""NumPy-only kinematic features for 16-D dual-arm end-pose trajectories."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .geometry import quaternion_angle


ARM_OFFSETS = {"left": 0, "right": 8}


def arm_components(
    trajectory: np.ndarray,
    arm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz, quaternion-wxyz and gripper streams for one arm."""

    if arm not in ARM_OFFSETS:
        raise ValueError(f"unknown arm {arm!r}")
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("trajectory must have shape [frames, 16]")
    offset = ARM_OFFSETS[arm]
    return (
        values[:, offset : offset + 3],
        values[:, offset + 3 : offset + 7],
        values[:, offset + 7],
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a one-dimensional stream without changing its length."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("moving_average expects a one-dimensional array")
    if len(array) == 0 or window <= 1:
        return array.copy()
    width = min(int(window), len(array))
    if width <= 1:
        return array.copy()
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(array, kernel, mode="same")


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("direction vector must have non-zero norm")
    return vector / norm


def local_motion_statistics(
    trajectory: np.ndarray,
    arm: str,
    frame: int,
    window: int,
    vertical_direction: np.ndarray,
) -> Dict[str, float | list]:
    """Compute windowed displacement, path, rotation and gripper statistics."""

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
                np.diff(position[start : end + 1], axis=0),
                axis=1,
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
        "gripper_trend": float(gripper[end] - gripper[start]),
    }


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start = None
    for index in range(len(values) + 1):
        active = index < len(values) and bool(values[index])
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def gripper_events(
    gripper: np.ndarray,
    threshold: float = 0.01,
) -> List[Dict[str, object]]:
    """Return contiguous gripper changes in state-frame coordinates."""

    values = np.asarray(gripper, dtype=np.float64)
    changes = np.diff(values)
    moving = np.abs(changes) >= threshold
    events: List[Dict[str, object]] = []
    for start, end in contiguous_runs(moving):
        state_start = start
        state_end = end + 1
        delta = float(values[state_end] - values[state_start])
        events.append(
            {
                "start_frame": int(state_start),
                "end_frame": int(state_end),
                "delta": delta,
                "direction": "decrease" if delta < 0.0 else "increase",
            }
        )
    return events


def segment_motion_summary(
    trajectory: np.ndarray,
    start: int,
    end: int,
    arm: str,
) -> Dict[str, float | list]:
    position, quaternion, gripper = arm_components(trajectory, arm)
    displacement = position[end] - position[start]
    path_length = float(
        np.sum(
            np.linalg.norm(
                np.diff(position[start : end + 1], axis=0),
                axis=1,
            )
        )
    )
    rotation = float(
        np.sum(
            quaternion_angle(
                quaternion[start + 1 : end + 1],
                quaternion[start:end],
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
