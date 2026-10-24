"""Keep source-specific physical conventions in both teacher and student prompts."""

VERSION = "pretrain_noncot_same_action_20261021"
STUDENT_VERSION = "robotwin_aligned_noncot_20261012"

SAME_ACTION_GROUP_CONTRACT = (
    "Adjacent primitives are merged only when both left_action and right_action match. "
    "Recorded absorbed both-still spans are idle context for that action. "
    "Describe the supplied action for each arm across the interval. Do not append a "
    "distinct action phase such as approach, lift, placement or withdrawal to an "
    "open/close instruction. Source annotation transitions are coarse noisy context; "
    "image and measured motion take precedence. "
)

ACTION_STARTS = {
    "move": "Move", "turn": "Rotate", "lift": "Lift", "lower": "Lower",
    "open_gripper": "Open the gripper", "close_gripper": "Close the gripper", "still": "Hold",
}

PROFILES = {
    "realomni_umi": {
        "robot": "two handheld DAS grippers in RealOmni Open; robot0 is left and robot1 is right",
        "state": "16 values ordered left xyz, quaternion wxyz, gripper, then right xyz, quaternion wxyz, gripper. Positions are in meters in the recorded VIO world frame. Quaternions are normalized and sign-continuous. Measured gripper aperture is divided by 0.103 m, approximately 0=closed and 1=open. Quaternion component differences are not Euler angles or a relative quaternion; zero gripper delta means unchanged.",
        "action": "Reference action[t] is the adjacent measured end-effector state[t+1], an absolute target, not a recorded controller command. guide_action is the 16D final reference action[end] minus current state[start], equivalently measured_state[end+1] minus measured_state[start]. There are N-1 real transitions for N measurements; the terminal measurement is retained without inventing an outgoing action. This reference is not evidence of task success.",
        "vision": "Two gripper cameras named left_wrist and right_wrist. Student observations are causal; future frames are offline teacher evidence only. This is human collection: no chassis or walking classification is performed. Do not infer a shared stationary base from gripper motion.",
    },
    "oxe_bridge": {
        "robot": "a single-arm WidowX robot in the OXE Bridge dataset",
        "state": "7 values: [x,y,z,roll,pitch,yaw,gripper]. Position is in meters in the source robot frame; angles are radians. Gripper aperture is approximately 1=open, 0=closed (measurement noise can exceed these limits).",
        "action": "Native action has world_vector (3 translation increments), rotation_delta (3 angle increments), open_gripper (absolute boolean target), and terminate_episode (preserved separately). guide_action has 7 components: the final commanded pose/aperture minus the current observed state. The commanded endpoint is state[end] plus the native final delta for xyz/rpy and the final absolute open_gripper target. It is not a sum of absolute positions or an observed success claim.",
        "vision": "There is one external camera named image. There is no second arm or wrist view. Screen-left is not robot-left. The original instruction can contain shorthand and scene descriptions; use the images to resolve the manipulated object, without inventing objects.",
    },
    "abc_yam": {
        "robot": "a bimanual YAM station in ABC-130k",
        "state": "The 14 values are [left_joint1,left_joint2,left_joint3,left_joint4,left_joint5,left_joint6,left_gripper,right_joint1,right_joint2,right_joint3,right_joint4,right_joint5,right_joint6,right_gripper]. Components 1 through 6 are the left arm joint angles, component 7 its gripper aperture, components 8 through 13 the right arm joint angles, and component 14 its gripper aperture. Joint angles are radians, ordered from base to wrist for each arm; gripper aperture is 0=closed, 1=open. These are joint coordinates, not Cartesian positions or quaternions. Zero joint/gripper delta means unchanged, not an absolute closed gripper. Optional measured end-effector transforms are separate row-major 4x4 matrices in the source station world frame, meters, right-handed Z-up.",
        "action": "Native actions are absolute commanded joint positions and gripper apertures in the same 14-value order. guide_action is final commanded 14-D action minus current observed 14-D state. Never interpret joint deltas as centimeters or infer Cartesian up/down from their signs. A command is a target, not proof of its physical completion.",
        "vision": "Use the actual camera topic labels: top or top_left/top_right, left_wrist, right_wrist. Left and right refer to physical robot arms, not screen sides. Streams have independent timestamps; the adapter supplies observations at or before the planning timestamp. Source subtask annotations, when present, are coarse task context, not guaranteed atomic action labels or proof of success.",
    },
}


def profile_text(profile: str) -> str:
    p = PROFILES[profile]
    return "\n".join([f"Robot: {p['robot']}.", p["state"], p["action"], p["vision"]])


def student_system(profile: str) -> str:
    from atomic_seg.prompt_contract import (
        NEXT_ATOMIC_SUBTASK, RECOMPUTE_GUIDE, CHUNK_COUNT_CONTRACT,
        SINGLE_ARM_CONTRACT, MULTI_ARM_CONTRACT, PREVIOUS_COMMAND_CONTRACT,
        NO_THINKING_OUTPUT_CONTRACT,
    )

    physical = {
        "realomni_umi": (
            "You are the high-level planner for a bimanual manipulation task. ",
            "The 16-D state and guide_action use left xyz, quaternion wxyz, gripper, "
            "then right xyz, quaternion wxyz, gripper. Positions are in meters in the "
            "recorded VIO world frame. Orientation arrays are named quat_wxyz. "
            "Gripper is measured aperture divided by 0.103 m, approximately 1=open and 0=closed. ",
            16,
            "The target endpoint is the adjacent measured state following the final inclusive "
            "reference action assigned to the subtask. Each reference action is the next "
            "measured absolute end-effector state. Quaternion entries in guide_action are "
            "component differences between sign-continuous unit quaternions, not Euler "
            "angles or a relative rotation quaternion. A zero gripper delta means unchanged aperture. ",
            MULTI_ARM_CONTRACT,
        ),
        "oxe_bridge": (
            "You are the high-level planner for a WidowX single-arm robot. ",
            "The 7-D robot state and guide_action use the order "
            "[x,y,z,roll,pitch,yaw,gripper]. xyz values are end-effector positions "
            "in meters in the source robot frame. Orientation uses Euler angles "
            "in radians in roll,pitch,yaw order. Gripper is approximately 1=open "
            "and 0=closed. Orientation arrays in the prompt are named rpy. ",
            7,
            "The endpoint is the commanded state for the final inclusive action "
            "assigned to that atomic subtask. Translation and rotation commands "
            "are increments applied to the observed pose at that action, while "
            "the gripper command specifies the absolute target aperture. "
            "Euler entries are component differences in the recorded angle representation. ",
            SINGLE_ARM_CONTRACT,
        ),
        "abc_yam": (
            "You are the high-level planner for a YAM dual-arm robot. ",
            "The 14-D robot state and guide_action use the order "
            "[left_joint1,left_joint2,left_joint3,left_joint4,left_joint5,left_joint6,left_gripper,"
            "right_joint1,right_joint2,right_joint3,right_joint4,right_joint5,right_joint6,right_gripper]. "
            "Joint positions are angles in radians, ordered from base to wrist for each arm. "
            "Gripper is approximately 1=open and 0=closed. Joint arrays in the prompt "
            "are named joint_positions. These values are joint coordinates rather than "
            "Cartesian positions or quaternions. Components 1 through 6 and 8 through 13 "
            "are the left and right joint angles respectively; components 7 and 14 "
            "are the corresponding gripper apertures. ",
            14,
            "The endpoint is the absolute commanded joint positions and gripper apertures "
            "for the final inclusive action assigned to that atomic subtask. Joint entries "
            "are component differences in radians, not end-effector displacement. "
            "A zero gripper delta means no change in aperture, not a closed gripper. "
            "Do not infer Cartesian motion direction from the signs of joint deltas. ",
            MULTI_ARM_CONTRACT,
        ),
    }
    if profile not in physical:
        raise ValueError(f"unsupported student profile {profile!r}")
    identity, state, dimension, endpoint, arms = physical[profile]
    return (
        identity + NEXT_ATOMIC_SUBTASK + state
        + f"guide_action is the remaining component-wise {dimension}-D state delta from the current "
        "planning origin to the target endpoint of the active atomic subtask, "
        "target_endpoint_state - current_origin_state. "
        + endpoint + RECOMPUTE_GUIDE + CHUNK_COUNT_CONTRACT + arms
        + PREVIOUS_COMMAND_CONTRACT + NO_THINKING_OUTPUT_CONTRACT
    )


def teacher_system(profile: str, *, grouped: bool = False) -> str:
    text = (
        "Annotate short robot motion segments with grounded atomic-subtask labels.\n"
        + profile_text(profile)
        + "\nThe segment boundaries are fixed by numerical motion segmentation. "
        "You see start/middle/end frames ONLY as an offline annotator. The student "
        "will see only current/past observations. Use the whole segment to name the "
        "intended physical motion. Do not write observations, outcomes, retrospective "
        "success statements, or reasoning into sub_task. Respect the supplied "
        "kinematic labels and gripper deltas. If an object is occluded use a neutral "
        "description, never invent its color, identity, contact or grasp security. "
        "Do not translate joint-space move labels into lift/lower without visual evidence. "
        "A still segment means hold the current pose, not automatically task complete. "
        "Each segment includes required_clause_starts. Follow those exact starts. "
        "For a single arm write one imperative starting with the required verb. "
        "For a bimanual robot write exactly 'Left arm: <imperative>. Right arm: <imperative>.' "
        "Each arm's imperative must begin with its own required verb, even for Hold. "
        "Never swap sides, omit a required gripper action, or add another action phase "
        "with 'then'. An open/close clause describes that gripper command and its "
        "intended object; do not assign the other arm's movement to it. "
        "Return JSON only: {\"segments\":[{\"segment_index\":0,\"sub_task\":\"...\"}]}. "
        "Return every requested index once in order, each sub_task a concise English "
        "imperative, maximum 45 words. No other keys, reasoning or thought tags. "
        "Do not generate boundaries, numerical actions or duration values."
    )
    if grouped:
        text = text.replace("short robot motion segments", "robot same-action motion intervals")
        text = text.replace("start/middle/end frames", "five chronological frames")
        text = text.replace("maximum 45 words", "maximum 70 words")
        text += (
            " Each interval groups contiguous primitive_segments and supplies motion_sequence for each arm. "
            + SAME_ACTION_GROUP_CONTRACT
        )
    return text
