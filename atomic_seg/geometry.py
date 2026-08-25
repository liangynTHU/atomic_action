"""Small NumPy-only quaternion and robust-loss utilities."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_quaternion(a)
    b = normalize_quaternion(b)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def quaternion_log_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation vectors for relative rotations a^-1 -> b (wxyz input)."""

    a = normalize_quaternion(a)
    b = normalize_quaternion(b)
    relative = quaternion_multiply(b, quaternion_conjugate(a))
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(norm, np.clip(relative[..., 0], -1.0, 1.0))
    return vector / np.maximum(norm[..., None], 1e-12) * angle[..., None]


def slerp(q0: np.ndarray, q1: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    q0 = normalize_quaternion(np.asarray(q0, dtype=np.float64))
    q1 = normalize_quaternion(np.asarray(q1, dtype=np.float64))
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    u = np.asarray(fractions, dtype=np.float64)[:, None]
    if dot > 0.9995:
        return normalize_quaternion((1.0 - u) * q0 + u * q1)
    theta = np.arccos(dot)
    return (
        np.sin((1.0 - u) * theta) / np.sin(theta) * q0
        + np.sin(u * theta) / np.sin(theta) * q1
    )


def huber(value: np.ndarray, delta: float = 1.5) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    magnitude = np.abs(value)
    return np.where(
        magnitude <= delta,
        0.5 * value * value,
        delta * (magnitude - 0.5 * delta),
    )
