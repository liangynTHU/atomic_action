"""Configuration for state-only atomic episode segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SegmentationConfig:
    """Physical-time thresholds for dual-arm atomic motion segmentation."""

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

    translation_speed_scale: float = 0.15
    rotation_speed_scale: float = 1.00
    gripper_rate_scale: float = 1.00
    state_pause_energy_threshold: float = 0.50
    gripper_event_rate_threshold: float = 0.50
    gripper_event_min_delta: float = 0.08
    onset_translation_speed: float = 0.08
    onset_vertical_speed: float = 0.05
    onset_rotation_speed: float = 0.40

    still_translation_threshold: float = 0.003
    still_path_threshold: float = 0.008
    still_rotation_threshold: float = 0.02
    vertical_ratio_threshold: float = 0.62
    vertical_displacement_threshold: float = 0.005
    rotation_dominance_ratio: float = 1.6
    rotation_window_scale: float = 0.05
    translation_window_scale: float = 0.03
    gripper_change_threshold: float = 0.01

    vertical_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    gripper_decrease_means_close: bool = True
