"""Quaternion utilities used by the segmentation core."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    return quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1e-12,
    )


def quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).copy()
    quaternion[..., 1:] *= -1.0
    return quaternion


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(left), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quaternion_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = normalize_quaternion(left)
    b = normalize_quaternion(right)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def quaternion_log_delta(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """Return rotation vectors for relative rotations before^-1 -> after."""

    origin = normalize_quaternion(before)
    target = normalize_quaternion(after)
    relative = quaternion_multiply(target, quaternion_conjugate(origin))
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(
        norm,
        np.clip(relative[..., 0], -1.0, 1.0),
    )
    return (
        vector
        / np.maximum(norm[..., None], 1e-12)
        * angle[..., None]
    )
