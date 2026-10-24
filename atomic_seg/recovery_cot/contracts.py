"""Compatibility identifiers and native recovery layout."""

from pathlib import Path

CODE_VERSION = "robotwin_recovery_cot_quality_20260930"
MANIFEST_SCHEMA = "robotwin_recovery_v28_source_manifest_v1"
ANNOTATION_SCHEMA = "robotwin_recovery_v28_causal_block_audit_v1"
RANK_SCHEMA = "robotwin_recovery_v28_rank_generation_v1"
VLA_SCHEMA = "robotwin_vla_atomic_episode_v28_v1"
VLM_FORMAT_VERSION = "inclusive_action_endpoint_image_history_v3_plus_v28_cot_v2"
DEFAULT_SOURCE_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/datasets/robotwin_recovery"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/datasets/"
    "robotwin_recovery_cot"
)
DEFAULT_REFERENCE_DRIVER = Path(
    "/mnt/lyn/workspace/FastWAM/tools/robotwin_cot_v28_full.py"
)
DEFAULT_MODEL = "Qwen/Qwen3.8-Flash-Next"
CAMERAS = (
    ("head", "observation/head_camera/rgb"),
    ("left_wrist", "observation/images/cam_left_wrist/rgb_jpeg"),
    ("right_wrist", "observation/images/cam_right_wrist/rgb_jpeg"),
)
STATE_LAYOUT = (
    "left_x",
    "left_y",
    "left_z",
    "left_qw",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qw",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_gripper",
)
CHUNK_SIZE = 8
FORBIDDEN_VLA_KEYS = {
    "cot",
    "reasoning",
    "chain_of_thought",
    "assistant_response",
    "reasoning_provenance",
    "task_chain",
}
VALID_PRIMITIVES = {
    "still",
    "move",
    "turn",
    "lift",
    "lower",
    "close_gripper",
    "open_gripper",
}
