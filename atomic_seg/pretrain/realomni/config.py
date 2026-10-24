"""Semantic settings are frozen separately from node addresses and concurrency."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Settings:
    source_root: str = "/apdcephfs_gy7/share_303588738/peterrao/data/umi/10Kh_RealOmin_OpenData"
    model: str = "Qwen/Qwen3.8-Flash-Next"
    logical_shards: int = 256
    pose_hz: float = 30.0
    camera_hz: float = 30.0
    gripper_hz: float = 50.0
    imu_hz: float = 200.0
    # More than 1.5 nominal periods means a missing sample, not normal clock phase.
    maximum_gap_periods: float = 1.5
    maximum_boundary_skew_periods: float = 1.5
    maximum_alignment_age_periods: float = 1.5
    gripper_scale_m: float = 0.103
    batch_size: int = 4
    temperature: float = 0.1
    max_tokens: int = 2200
    content_attempts: int = 3
    service_attempts: int = 6
    image_width: int = 640
    image_height: int = 520
    jpeg_quality: int = 90
    maximum_episode_rows: int = 200_000
    # Legacy strict mode remains reproducible; the CLI selects trim_edges_1s.
    quality_policy: str = "strict"
    maximum_edge_trim_seconds: float = 1.0
    length_policy: dict = field(default_factory=lambda: {
        "minimum_frames": None, "preferred_minimum_frames": 32,
        "preferred_maximum_frames": 64, "maximum_merged_frames": 128,
    })

    def as_dict(self):
        return asdict(self)

    def __post_init__(self):
        import math

        for name in ("pose_hz", "camera_hz", "gripper_hz", "imu_hz", "gripper_scale_m",
                     "maximum_gap_periods", "maximum_boundary_skew_periods",
                     "maximum_alignment_age_periods"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.logical_shards < 1 or self.maximum_episode_rows < 3:
            raise ValueError("invalid shard count or episode row limit")
        if self.gripper_scale_m != 0.103:
            raise ValueError("RealOmni aperture normalization is fixed at 0.103 m")
        if self.quality_policy not in {"strict", "trim_edges_1s"}:
            raise ValueError("unknown RealOmni quality policy")
        if self.maximum_edge_trim_seconds != 1.0:
            raise ValueError("the agreed edge trim limit is one second per edge")
        from atomic_seg.pretrain.length_policy import LengthPolicy

        LengthPolicy(**self.length_policy)


OUTPUT_ROOT = "/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/RealOmni"
TERMINAL = {"rejected", "segmented", "complete", "review"}
