"""Atomic episode segmentation and successful-demonstration data tooling."""

from .config import SegmentationConfig
from .segmentation import segment_episode

__all__ = ["SegmentationConfig", "segment_episode"]
