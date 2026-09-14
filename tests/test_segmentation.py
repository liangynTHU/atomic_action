import unittest

import numpy as np

from atomic_seg import SegmentationConfig, segment_episode
from atomic_seg.features import moving_average
from atomic_seg.geometry import quaternion_angle
from atomic_seg.segmentation import (
    group_bimanual_evidence,
    resolve_temporal_parameters,
    segments_from_boundaries,
    successor_trend_onset,
)


def synthetic_dual_trajectory(frame_count: int = 100) -> np.ndarray:
    trajectory = np.zeros((frame_count, 16), dtype=np.float64)
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = 1.0
    trajectory[:, 11] = 1.0
    trajectory[:, 15] = 1.0
    return trajectory


class SegmentationTests(unittest.TestCase):
    def test_moving_average_preserves_short_length(self):
        values = np.array([1.0])
        averaged = moving_average(values, window=5)
        self.assertEqual(averaged.shape, values.shape)
        np.testing.assert_allclose(averaged, values)

    def test_quaternion_angle_is_sign_invariant(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(quaternion_angle(q, -q)), 0.0)

    def test_boundaries_create_disjoint_slices(self):
        self.assertEqual(
            segments_from_boundaries([5, 10], 15),
            [
                {"start_frame": 0, "end_frame": 4},
                {"start_frame": 5, "end_frame": 9},
                {"start_frame": 10, "end_frame": 14},
            ],
        )

    def test_temporal_parameters_follow_episode_clock(self):
        config = SegmentationConfig(phase_window_seconds=0.20)
        _, at_50hz = resolve_temporal_parameters(
            np.arange(101) * 0.02,
            101,
            config,
        )
        _, at_100hz = resolve_temporal_parameters(
            np.arange(201) * 0.01,
            201,
            config,
        )
        self.assertEqual(at_50hz.phase_window_frames, 10)
        self.assertEqual(at_100hz.phase_window_frames, 20)

    def test_successor_onset_precedes_score_crossover(self):
        previous = np.r_[
            np.full(10, 2.0),
            np.linspace(2.0, 0.0, 10),
            np.zeros(10),
        ]
        current = np.r_[
            np.zeros(10),
            np.linspace(0.3, 2.0, 10),
            np.full(10, 2.0),
        ]
        onset, diagnostics = successor_trend_onset(
            previous,
            current,
            nominal_frame=15,
            search_frames=8,
            confirm_frames=3,
        )
        self.assertLess(onset, 15)
        self.assertTrue(diagnostics["onset_found"])
        self.assertTrue(diagnostics["competing_trends_detected"])

    def test_nearby_different_arm_phases_stay_separate(self):
        _, temporal = resolve_temporal_parameters(
            np.arange(80) * 0.02,
            80,
            SegmentationConfig(),
        )

        def evidence(frame: int, arm: str, phase: str):
            return {
                "frame": frame,
                "source_arm": arm,
                "source_arms": [arm],
                "evidence_type": "successor_trend_onset",
                "evidence_types": ["successor_trend_onset"],
                "evidence_class": "soft",
                "strength": 1.0,
                "current_phase": phase,
                "previous_phase": "still",
                "supported_by_both_arms": False,
            }

        grouped = group_bimanual_evidence(
            {
                "left": [
                    evidence(20, "left", "lift"),
                    evidence(40, "left", "move"),
                ],
                "right": [
                    evidence(21, "right", "lift"),
                    evidence(43, "right", "turn"),
                ],
            },
            temporal,
        )
        self.assertEqual(
            [int(item["frame"]) for item in grouped],
            [20, 40, 43],
        )
        self.assertEqual(grouped[0]["source_arm"], "both")

    def test_segment_episode_is_state_only_and_factorized(self):
        trajectory = synthetic_dual_trajectory()
        trajectory[5:30, 0] = np.linspace(0.0, 0.25, 25)
        trajectory[30:, 0] = 0.25
        trajectory[55:80, 8] = np.linspace(0.0, 0.25, 25)
        trajectory[80:, 8] = 0.25
        trajectory[25:36, 7] = np.linspace(1.0, 0.0, 11)
        trajectory[36:, 7] = 0.0
        trajectory[70:81, 15] = np.linspace(1.0, 0.0, 11)
        trajectory[81:, 15] = 0.0
        result = segment_episode(
            trajectory,
            np.arange(100) * 0.02,
        )
        policy = result["decision_input_policy"]
        self.assertIn("video", policy["excluded_from_boundary_decisions"])
        self.assertIn("action", policy["excluded_from_boundary_decisions"])
        self.assertEqual(
            set(result["per_arm_boundaries"]),
            {"left", "right"},
        )
        self.assertGreater(len(result["segments"]), 1)

if __name__ == "__main__":
    unittest.main()
