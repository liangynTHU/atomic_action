import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atomic_seg.generation import (
    ACTION_EPISODE_SCHEMA,
    COT_DATASET_SCHEMA,
    TemplateReasoner,
    build_action_episode,
    validate_action_episode,
    validate_cot_row,
)
from atomic_seg.io import SuccessEpisode, load_success_manifest


def synthetic_source(frame_count: int = 40):
    state = np.zeros((frame_count, 16), dtype=np.float64)
    state[:, 3] = 1.0
    state[:, 7] = 1.0
    state[:, 11] = 1.0
    state[:, 15] = 1.0
    state[5:25, 0] = np.linspace(0.0, 0.2, 20)
    state[25:, 0] = 0.2
    action = state.copy()
    action[:-1] = state[1:]
    return {
        "observation.state": state,
        "action": action,
        "timestamp": np.arange(frame_count) * 0.02,
    }


class GenerationTests(unittest.TestCase):
    def test_manifest_rejects_non_success_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {"episode_index": 3, "success": False}
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "only accepts fully correct episodes",
            ):
                load_success_manifest(path)

    def test_action_episode_contains_no_cot_fields(self):
        record = build_action_episode(
            SuccessEpisode(
                episode_index=0,
                task="Move the object.",
                success=True,
            ),
            synthetic_source(),
        )
        self.assertEqual(record["schema"], ACTION_EPISODE_SCHEMA)
        validate_action_episode(record)
        serialized = json.dumps(record)
        for forbidden in (
            '"cot"',
            '"chain_of_thought"',
            '"reasoning"',
            '"assistant_response"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_template_reasoner_is_pre_action_and_calibrated(self):
        reasoner = TemplateReasoner()
        request = type(
            "Request",
            (),
            {
                "task": "Lift the bottle.",
                "history_actions": (),
                "target": {
                    "sub_task": "Move the left arm toward the bottle."
                },
            },
        )()
        value = reasoner.generate(request)
        self.assertIn("pre-action", value)
        self.assertIn("does not claim contact", value)

    def test_valid_cot_row_embeds_reasoning_only_in_assistant(self):
        target = {
            "guide_action": [0.0] * 16,
            "primary_action_verb": "left:move, right:still",
            "left_action": "move",
            "right_action": "still",
            "sub_task": "Move the left arm while keeping the right arm still.",
            "num_chunks": 1,
        }
        row = {
            "schema": COT_DATASET_SCHEMA,
            "sample_id": "episode_000000__segment_000",
            "episode_id": "episode_000000",
            "episode_index": 0,
            "segment_index": 0,
            "task": "Lift the bottle.",
            "success": True,
            "images": ["a.jpg", "b.jpg", "c.jpg"],
            "image_roles": [
                "current_cam_high",
                "current_cam_left_wrist",
                "current_cam_right_wrist",
            ],
            "image_frames": [0, 0, 0],
            "current_frame": 0,
            "history_actions": [],
            "target": target,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image><image><image>\nTask: Lift the bottle.\n"
                        "Current evidence only."
                    ),
                },
                {
                    "from": "gpt",
                    "value": (
                        "<think>Move closer before interaction.</think>\n"
                        '{"guide_action":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'
                        '"num_chunks":1,'
                        '"primary_action_verb":"left:move, right:still",'
                        '"sub_task":"Move the left arm while keeping the right arm still."}'
                    ),
                },
            ],
        }
        validate_cot_row(row)
        self.assertNotIn(
            target["sub_task"],
            row["conversations"][0]["value"],
        )


if __name__ == "__main__":
    unittest.main()
