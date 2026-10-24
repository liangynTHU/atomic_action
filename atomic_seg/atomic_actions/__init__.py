"""Stable public entrypoint for atomic action segmentation.

The pretraining segmentation algorithm and its future pipeline are unchanged.
"""

from atomic_seg.config import SegmentationConfig
from atomic_seg.segmentation import segment_episode, serializable_segmentation
from atomic_seg.still_policy import plan_still_segment_groups

__all__ = [
    "SegmentationConfig",
    "segment_episode",
    "serializable_segmentation",
    "plan_still_segment_groups",
]
