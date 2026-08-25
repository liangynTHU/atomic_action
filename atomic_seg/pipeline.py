"""Top-level orchestration shared by the CLI and tests."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .evaluation import boundary_metrics, label_agreement
from .features import FeatureConfig, extract_features
from .segmentation import (
    SegmenterConfig,
    label_segments,
    version1_piecewise_geodesic_dp,
    version2_geometry_aware_reconstruction_dp,
    version3_motion_change_points,
    version4_strong_event_segmentation,
    version5_event_phase_atomic_motion,
)
from .state_segmentation import (
    attach_bimanual_segment_context,
    neutralize_gripper_semantics,
    version6_time_normalized_state_segmentation,
    version7_onset_aware_state_segmentation,
    version8_factorized_bimanual_state_segmentation,
)


VERSION_NAMES = {
    1: "piecewise_geodesic_approximation",
    2: "geometry_aware_reconstruction",
    3: "motion_change_point_ablation",
    4: "strong_event_segmentation",
    5: "event_phase_atomic_motion",
    6: "time_normalized_state_evidence",
    7: "successor_onset_state_transitions",
    8: "factorized_bimanual_state_motion",
}


VERSION_ROLES = {
    1: "piecewise-geodesic trajectory approximation baseline",
    2: "geometry-aware reconstruction baseline",
    3: "motion change-point ablation",
    4: "strong event-evidence ablation",
    5: "event- and phase-aware frame-threshold baseline",
    6: "timestamp-normalized state-only segmentation",
    7: "state-only segmentation with successor-trend-onset boundaries",
    8: "final factorized bimanual state-only atomic motion segmentation",
}


def run_version(
    trajectory: np.ndarray,
    version: int,
    reference: Dict[str, object] | None = None,
    segmenter_config: SegmenterConfig | None = None,
    feature_config: FeatureConfig | None = None,
    timestamps: np.ndarray | None = None,
) -> Dict[str, object]:
    segmenter_config = segmenter_config or SegmenterConfig()
    feature_bundle = extract_features(trajectory, feature_config)
    if version == 1:
        result = version1_piecewise_geodesic_dp(
            trajectory, segmenter_config
        )
    elif version == 2:
        result = version2_geometry_aware_reconstruction_dp(
            trajectory, segmenter_config
        )
    elif version == 3:
        result = version3_motion_change_points(
            trajectory, feature_bundle, segmenter_config
        )
    elif version == 4:
        result = version4_strong_event_segmentation(
            trajectory, feature_bundle, segmenter_config
        )
    elif version == 5:
        result = version5_event_phase_atomic_motion(
            trajectory, feature_bundle, segmenter_config
        )
    elif version == 6:
        result = version6_time_normalized_state_segmentation(
            trajectory, timestamps, segmenter_config
        )
    elif version == 7:
        result = version7_onset_aware_state_segmentation(
            trajectory, timestamps, segmenter_config
        )
    elif version == 8:
        result = version8_factorized_bimanual_state_segmentation(
            trajectory, timestamps, segmenter_config
        )
    else:
        raise ValueError(f"Unknown version {version}; expected 1..8")
    labelled = label_segments(
        trajectory,
        list(result["segments"]),
        feature_bundle,
        segmenter_config,
    )
    if version >= 6:
        labelled = neutralize_gripper_semantics(labelled)
    if version == 8:
        labelled = attach_bimanual_segment_context(
            labelled, result["selection"]
        )
    reference_segments = reference.get("segments") if reference else None
    output = {
        "version": version,
        "version_name": VERSION_NAMES[version],
        "version_role": VERSION_ROLES[version],
        "frame_count": int(len(trajectory)),
        "method_scope": "kinematic_atomic_motion_segmentation",
        "active_arm": feature_bundle["active_arm"],
        "total_energy": feature_bundle["total_energy"],
        "selection": result["selection"],
        "segments": labelled,
        "boundary_metrics": boundary_metrics(
            labelled, reference_segments, tolerance=5
        ),
        "label_metrics": label_agreement(labelled, reference_segments),
        "evaluation_note": (
            "Weak-reference metrics measure agreement with an existing "
            "automatic segmentation, not ground-truth accuracy."
        ),
        "_feature_bundle": feature_bundle,
    }
    if "_state_bundle" in result:
        output["_state_bundle"] = result["_state_bundle"]
    return output
