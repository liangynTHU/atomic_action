"""Versioned physical and filtering contract. Runtime topology is separate."""

from dataclasses import asdict, dataclass, field

SOURCE_ROOT = "/apdcephfs_gy2/share_303214312/leoyizhang/datasets/agibot_beta_v3"
OUTPUT_ROOT = "/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot"
MODEL = "glm-5.3-flash-ioa"
API_BASE = "http://127.0.0.1:15721/tencent/v1"
CAMERAS = ("head", "head_center_fisheye", "head_left_fisheye", "head_right_fisheye",
           "hand_left", "hand_right", "back_left_fisheye", "back_right_fisheye")
STUDENT_CAMERAS = ("head", "hand_left", "hand_right")


@dataclass(frozen=True)
class Settings:
    source_root: str = SOURCE_ROOT
    fps: int = 30
    cameras: tuple = CAMERAS
    student_cameras: tuple = STUDENT_CAMERAS
    # Encoder openness proxy, NOT millimetres of jaw opening. Fixed approximate
    # dataset-level anchors, visually checked against raw recordings. These are
    # representation anchors, not a manufacturer-certified physical calibration.
    gripper_open: float = 35.0
    gripper_closed: float = 120.0
    gripper_tolerance: float = 0.06
    base_command_tolerance: float = 1e-6
    base_translation_tolerance_m: float = 1e-4
    base_rotation_tolerance_rad: float = 1e-4
    quaternion_norm_tolerance: float = 0.01
    time_tolerance_s: float = 0.001
    video_tolerance_s: float = 0.001
    model: str = MODEL
    batch_size: int = 4
    temperature: float = 0.1
    max_tokens: int = 8192
    reasoning_effort: str = "xhigh"
    semantic_attempts: int = 3
    object_grounding: bool = False
    length_policy: dict = field(default_factory=lambda: {
        "minimum_frames": None, "preferred_minimum_frames": 32,
        "preferred_maximum_frames": 64, "maximum_merged_frames": 128,
    })

    def __post_init__(self):
        if self.gripper_closed <= self.gripper_open or self.fps != 30:
            raise ValueError("invalid AgiBot calibration or time contract")
        if not set(self.student_cameras) <= set(self.cameras):
            raise ValueError("student cameras must be inspected source cameras")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")
        if self.batch_size < 1 or self.max_tokens < 1 or self.semantic_attempts < 1:
            raise ValueError("batch size, max tokens and semantic attempts must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def as_dict(self):
        return asdict(self)
