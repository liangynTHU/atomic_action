import unittest

import numpy as np

from atomic_seg.evaluation import boundary_metrics
from atomic_seg.features import (
    gripper_events,
    local_motion_statistics,
    moving_average,
)
from atomic_seg.geometry import quaternion_angle, slerp
from atomic_seg.segmentation import (
    SegmenterConfig,
    geometric_primitives_from_knots,
    persistent_phase_runs,
    segments_from_boundaries,
    temporal_merge_evidence,
    version5_event_phase_atomic_motion,
)
from atomic_seg.features import extract_features
from atomic_seg.state_segmentation import (
    group_bimanual_evidence,
    resolve_temporal_parameters,
    successor_trend_onset_from_scores,
    version6_time_normalized_state_segmentation,
    version8_factorized_bimanual_state_segmentation,
)


def synthetic_dual_trajectory(frame_count=40):
    trajectory = np.zeros((frame_count, 16), dtype=np.float64)
    trajectory[:, 3] = 1.0
    trajectory[:, 11] = 1.0
    trajectory[:, 7] = 1.0
    trajectory[:, 15] = 1.0
    return trajectory


class GeometryTests(unittest.TestCase):
    def test_quaternion_angle_sign_invariance(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(quaternion_angle(q, -q)), 0.0)

    def test_slerp_endpoints(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        path = slerp(q0, q1, np.array([0.0, 1.0]))
        np.testing.assert_allclose(path[0], q0, atol=1e-7)
        self.assertAlmostEqual(float(quaternion_angle(path[1], q1)), 0.0)


class SegmentationTests(unittest.TestCase):
    def test_moving_average_preserves_very_short_episode_length(self):
        values = np.array([1.0])
        averaged = moving_average(values, window=5)
        self.assertEqual(averaged.shape, values.shape)
        np.testing.assert_allclose(averaged, values)

    def test_geometric_knots_share_endpoints(self):
        primitives = geometric_primitives_from_knots([0, 5, 10], 11)
        self.assertEqual(
            primitives,
            [
                {"start_knot": 0, "end_knot": 5},
                {"start_knot": 5, "end_knot": 10},
            ],
        )
        self.assertEqual(
            primitives[0]["end_knot"], primitives[1]["start_knot"]
        )

    def test_frame_boundaries_are_disjoint_slices(self):
        segments = segments_from_boundaries([5, 10], 15)
        self.assertEqual(
            segments,
            [
                {"start_frame": 0, "end_frame": 4},
                {"start_frame": 5, "end_frame": 9},
                {"start_frame": 10, "end_frame": 14},
            ],
        )

    def test_gripper_event(self):
        gripper = np.array([1.0, 1.0, 0.8, 0.4, 0.0, 0.0])
        events = gripper_events(gripper)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "close_gripper")
        self.assertEqual(events[0]["start_frame"], 1)
        self.assertEqual(events[0]["end_frame"], 4)

    def test_windowed_vertical_statistics_use_configured_direction(self):
        trajectory = synthetic_dual_trajectory(11)
        trajectory[:, 0] = np.linspace(0.0, 1.0, 11)
        statistics = local_motion_statistics(
            trajectory,
            "left",
            frame=5,
            window=9,
            vertical_direction=np.array([1.0, 0.0, 0.0]),
        )
        self.assertGreater(float(statistics["vertical_ratio"]), 0.99)
        self.assertLess(float(statistics["horizontal_ratio"]), 0.01)

    def test_persistence_filters_isolated_phase_flicker(self):
        config = SegmenterConfig(phase_min_frames=3)
        labels = ["move"] * 5 + ["turn"] + ["move"] * 5
        runs = persistent_phase_runs(labels, config)
        self.assertEqual(
            runs,
            [{"start_frame": 0, "end_frame": 10, "phase": "move"}],
        )

    def test_strong_evidence_has_priority_over_soft(self):
        strong = {
            "frame": 10,
            "source_arm": "left",
            "source_arms": ["left"],
            "evidence_type": "close_gripper",
            "evidence_types": ["close_gripper"],
            "evidence_class": "strong",
            "strength": 3.0,
            "supported_by_both_arms": False,
        }
        soft = {
            "frame": 12,
            "source_arm": "right",
            "source_arms": ["right"],
            "evidence_type": "persistent_phase_transition",
            "evidence_types": ["persistent_phase_transition"],
            "evidence_class": "soft",
            "strength": 9.0,
            "supported_by_both_arms": False,
        }
        merged = temporal_merge_evidence([soft, strong], radius=5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["evidence_class"], "strong")
        self.assertIn("close_gripper", merged[0]["evidence_types"])
        self.assertIn(
            "persistent_phase_transition", merged[0]["evidence_types"]
        )

    def test_joint_v5_retains_both_arm_sources(self):
        trajectory = synthetic_dual_trajectory(50)
        trajectory[:20, 0] = np.linspace(0.0, 0.2, 20)
        trajectory[20:, 0] = 0.2
        trajectory[:25, 8] = 0.0
        trajectory[25:, 8] = np.linspace(0.0, 0.2, 25)
        trajectory[15:21, 7] = np.linspace(1.0, 0.0, 6)
        trajectory[21:, 7] = 0.0
        trajectory[30:36, 15] = np.linspace(1.0, 0.0, 6)
        trajectory[36:, 15] = 0.0
        config = SegmenterConfig(
            arm_mode="joint",
            min_segment_frames=4,
            phase_min_frames=4,
        )
        result = version5_event_phase_atomic_motion(
            trajectory, extract_features(trajectory), config
        )
        sources = {
            arm
            for candidate in [
                *result["selection"]["strong_candidates"],
                *result["selection"]["soft_candidates"],
            ]
            for arm in candidate["source_arms"]
        }
        self.assertEqual(sources, {"left", "right"})
        self.assertEqual(
            set(
                result["selection"]["diagnostics"]["strong_events"][
                    "per_arm"
                ]
            ),
            {"left", "right"},
        )

    def test_v6_temporal_parameters_follow_episode_clock(self):
        config = SegmenterConfig(
            phase_window_seconds=0.20,
            min_segment_seconds=0.10,
        )
        _, at_50hz = resolve_temporal_parameters(
            np.arange(101) * 0.02, 101, config
        )
        _, at_100hz = resolve_temporal_parameters(
            np.arange(201) * 0.01, 201, config
        )
        self.assertEqual(at_50hz.phase_window_frames, 10)
        self.assertEqual(at_100hz.phase_window_frames, 20)
        self.assertEqual(at_50hz.min_segment_frames, 5)
        self.assertEqual(at_100hz.min_segment_frames, 10)

    def test_v6_boundary_policy_excludes_visual_and_action_config(self):
        trajectory = synthetic_dual_trajectory(40)
        result = version6_time_normalized_state_segmentation(
            trajectory,
            np.arange(40) * 0.02,
            SegmenterConfig(),
        )
        policy = result["selection"]["decision_input_policy"]
        self.assertEqual(
            policy["boundary_decision_inputs"][-1], "timestamp"
        )
        self.assertIn(
            "action_config", policy["excluded_from_boundary_decisions"]
        )
        self.assertIn("video", policy["excluded_from_boundary_decisions"])
        self.assertTrue(policy["boundary_lock_after_state_segmentation"])

    def test_successor_onset_precedes_competing_score_crossover(self):
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
        nominal_crossover = 15
        onset, diagnostics = successor_trend_onset_from_scores(
            previous,
            current,
            nominal_frame=nominal_crossover,
            search_frames=8,
            confirm_frames=3,
        )
        self.assertLess(onset, nominal_crossover)
        self.assertGreaterEqual(onset, 10)
        self.assertTrue(diagnostics["onset_found"])
        self.assertTrue(diagnostics["successor_trend_rising"])
        self.assertTrue(diagnostics["previous_trend_falling"])
        self.assertTrue(diagnostics["competing_trends_detected"])

    def test_v8_cross_arm_fusion_preserves_different_nearby_events(self):
        config = SegmenterConfig(
            cross_arm_sync_seconds=0.08,
            coordination_min_segment_seconds=0.04,
        )
        _, temporal = resolve_temporal_parameters(
            np.arange(80) * 0.02, 80, config
        )

        def evidence(frame, arm, current_phase):
            return {
                "frame": frame,
                "source_arm": arm,
                "source_arms": [arm],
                "evidence_type": "successor_trend_onset",
                "evidence_types": ["successor_trend_onset"],
                "evidence_class": "soft",
                "strength": 1.0,
                "current_phase": current_phase,
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
            [int(item["frame"]) for item in grouped], [20, 40, 43]
        )
        self.assertEqual(grouped[0]["source_arm"], "both")
        self.assertEqual(
            grouped[0]["coordination_type"], "synchronous_phase"
        )
        self.assertEqual(grouped[1]["source_arm"], "left")
        self.assertEqual(grouped[2]["source_arm"], "right")

    def test_v8_outputs_factorized_arm_timelines(self):
        trajectory = synthetic_dual_trajectory(100)
        trajectory[5:30, 0] = np.linspace(0.0, 0.25, 25)
        trajectory[30:, 0] = 0.25
        trajectory[55:80, 8] = np.linspace(0.0, 0.25, 25)
        trajectory[80:, 8] = 0.25
        trajectory[25:36, 7] = np.linspace(1.0, 0.0, 11)
        trajectory[36:, 7] = 0.0
        trajectory[70:81, 15] = np.linspace(1.0, 0.0, 11)
        trajectory[81:, 15] = 0.0
        result = version8_factorized_bimanual_state_segmentation(
            trajectory,
            np.arange(100) * 0.02,
            SegmenterConfig(),
        )
        selection = result["selection"]
        self.assertEqual(set(selection["arm_timelines"]), {"left", "right"})
        self.assertGreater(len(selection["arm_timelines"]["left"]), 1)
        self.assertGreater(len(selection["arm_timelines"]["right"]), 1)
        self.assertTrue(
            all(
                item["source_arm"] in {"left", "both"}
                for item in selection["per_arm_boundary_evidence"]["left"]
            )
        )
        self.assertTrue(
            all(
                item["source_arm"] in {"right", "both"}
                for item in selection["per_arm_boundary_evidence"]["right"]
            )
        )


class EvaluationTests(unittest.TestCase):
    def test_weak_reference_metrics_include_over_under_counts(self):
        predicted = segments_from_boundaries([5, 10], 15)
        reference = segments_from_boundaries([6], 15)
        metrics = boundary_metrics(predicted, reference, tolerance=1)
        self.assertEqual(
            metrics["reference_type"], "weak_segmentation_reference"
        )
        self.assertEqual(metrics["over_segmentation_count"], 1)
        self.assertEqual(metrics["under_segmentation_count"], 0)
        self.assertEqual(metrics["segment_count_error"], 1)


if __name__ == "__main__":
    unittest.main()
