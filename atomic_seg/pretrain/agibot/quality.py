"""Validate dual-arm measurements; assume a stationary base for the 16D scope."""

import numpy as np

from .config import BASE_AUXILIARY_FIELDS


def inspect_numeric(row, arrays, settings):
    audit = {"whole_episode": True, "trimmed_rows": 0, "reasons": [],
             "checks": {"numeric": "running", "base": "assumed_stationary_16d", "video": "not_checked"},
             "base": {"policy": settings.base_policy, "action_dimension": 16,
                      "stationary_assumed": True, "measured_stationarity_verified": False,
                      "auxiliary_fields_used": False}}

    def reject(code, **evidence):
        audit["reasons"].append({"code": code, **evidence})

    n = row["length"]
    if n < 3:
        reject("insufficient_measured_states", count=n)
    for name, feature in row["features"].items():
        if name in BASE_AUXILIARY_FIELDS:
            continue
        if name not in arrays:
            reject("missing_numeric_field", field=name)
            continue
        a = arrays[name]
        shape = (n,) if feature["shape"] == [1] and name in {"timestamp", "frame_index", "episode_index", "index", "task_index"} else (n, *feature["shape"])
        if a.shape != shape:
            reject("numeric_shape_mismatch", field=name, expected=list(shape), actual=list(a.shape))
        elif not np.isfinite(a).all():
            reject("invalid_numeric_values", field=name)
    if audit["reasons"]:
        audit["checks"]["numeric"] = "failed"
        return audit
    required = {"frame_index", "timestamp", "index", "episode_index", "observation.states.end.position",
                "observation.states.end.orientation", "observation.states.effector.position"}
    if not required <= set(arrays):
        reject("missing_required_numeric_fields", fields=sorted(required - set(arrays)))
        audit["checks"]["numeric"] = "failed"
        return audit
    for name, shape in {
        "observation.states.end.position": (n, 2, 3),
        "observation.states.end.orientation": (n, 2, 4),
        "observation.states.effector.position": (n, 2),
    }.items():
        if arrays[name].shape != shape:
            reject("invalid_16d_dual_arm_shape", field=name, expected=list(shape),
                   actual=list(arrays[name].shape))
    if audit["reasons"]:
        audit["checks"]["numeric"] = "failed"
        return audit
    if not np.array_equal(arrays["frame_index"], np.arange(n)):
        reject("frame_gap_duplicate_or_disorder")
    first = row["metadata"]["dataset_from_index"]
    if not np.array_equal(arrays["index"], np.arange(first, first + n)) or row["metadata"]["dataset_to_index"] != first+n:
        reject("dataset_index_gap_or_length_mismatch")
    if not np.all(arrays["episode_index"] == row["episode_index"]):
        reject("episode_index_mismatch")
    skew = np.abs(arrays["timestamp"] - np.arange(n) / settings.fps)
    if np.any(skew > settings.time_tolerance_s) or np.any(np.diff(arrays["timestamp"]) <= 0):
        reject("timestamp_gap_or_disorder", max_skew_s=float(skew.max()))
    norms = np.linalg.norm(arrays["observation.states.end.orientation"], axis=-1)
    if np.any(abs(norms - 1) > settings.quaternion_norm_tolerance):
        reject("invalid_end_quaternion", first_frame=int(np.argwhere(abs(norms-1) > settings.quaternion_norm_tolerance)[0, 0]))
    raw_grip = arrays["observation.states.effector.position"]
    g = (settings.gripper_closed - raw_grip) / (settings.gripper_closed - settings.gripper_open)
    if np.any(g < -settings.gripper_tolerance) or np.any(g > 1+settings.gripper_tolerance):
        reject("gripper_outside_calibrated_range", raw_min=raw_grip.min(axis=0).tolist(), raw_max=raw_grip.max(axis=0).tolist())
    audit["checks"]["numeric"] = "failed" if audit["reasons"] else "passed"
    audit["numeric"] = {"source_rows": n, "reference_action_rows": n-1, "max_timestamp_skew_s": float(skew.max())}
    return audit
