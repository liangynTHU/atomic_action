"""Agreement metrics against a weak segmentation reference.

These metrics are not ground-truth segmentation accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def internal_boundaries(
    segments: Sequence[Dict[str, object]],
) -> List[int]:
    return [int(segment["end_frame"]) + 1 for segment in segments[:-1]]


def _greedy_boundary_matching(
    predicted: Sequence[int],
    reference: Sequence[int],
    tolerance: int,
) -> Tuple[List[Dict[str, int]], List[int], List[int]]:
    candidate_pairs = sorted(
        (
            abs(predicted_index - reference_index),
            predicted_position,
            reference_position,
        )
        for predicted_position, predicted_index in enumerate(predicted)
        for reference_position, reference_index in enumerate(reference)
        if abs(predicted_index - reference_index) <= tolerance
    )
    used_predicted = set()
    used_reference = set()
    matches: List[Dict[str, int]] = []
    for distance, predicted_position, reference_position in candidate_pairs:
        if (
            predicted_position in used_predicted
            or reference_position in used_reference
        ):
            continue
        used_predicted.add(predicted_position)
        used_reference.add(reference_position)
        matches.append(
            {
                "predicted": int(predicted[predicted_position]),
                "reference": int(reference[reference_position]),
                "distance": int(distance),
            }
        )
    unmatched_predicted = [
        int(boundary)
        for index, boundary in enumerate(predicted)
        if index not in used_predicted
    ]
    unmatched_reference = [
        int(boundary)
        for index, boundary in enumerate(reference)
        if index not in used_reference
    ]
    return matches, unmatched_predicted, unmatched_reference


def boundary_metrics(
    predicted_segments: Sequence[Dict[str, object]],
    reference_segments: Sequence[Dict[str, object]] | None,
    tolerance: int = 5,
) -> Dict[str, object] | None:
    """Tolerance-based agreement with the existing weak reference."""

    if not reference_segments:
        return None
    predicted = internal_boundaries(predicted_segments)
    reference = internal_boundaries(reference_segments)
    matches, false_positive, false_negative = _greedy_boundary_matching(
        predicted, reference, tolerance
    )
    true_positive = len(matches)
    precision = (
        true_positive / len(predicted) if predicted else float(not reference)
    )
    recall = true_positive / len(reference) if reference else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    distances = [match["distance"] for match in matches]
    count_error = len(predicted_segments) - len(reference_segments)
    return {
        "reference_type": "weak_segmentation_reference",
        "interpretation": (
            "agreement with the existing weak segmentation reference; "
            "not ground-truth accuracy"
        ),
        "tolerance_frames": int(tolerance),
        "predicted_boundaries": predicted,
        "reference_boundaries": reference,
        "matches": matches,
        "matched": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_boundaries": false_positive,
        "false_negative_boundaries": false_negative,
        "over_segmentation_count": len(false_positive),
        "under_segmentation_count": len(false_negative),
        "predicted_segment_count": len(predicted_segments),
        "reference_segment_count": len(reference_segments),
        "segment_count_error": int(count_error),
        "absolute_segment_count_error": abs(int(count_error)),
        "mean_absolute_boundary_error": (
            float(np.mean(distances)) if distances else None
        ),
        "max_absolute_boundary_error": (
            int(max(distances)) if distances else None
        ),
    }


def label_agreement(
    predicted_segments: Sequence[Dict[str, object]],
    reference_segments: Sequence[Dict[str, object]] | None,
) -> Dict[str, object] | None:
    """Majority-overlap label agreement with a weak reference."""

    if not reference_segments:
        return None
    reference_labels = []
    reference_ranges = []
    for segment in reference_segments:
        left = str(segment.get("left_action", "still"))
        right = str(segment.get("right_action", "still"))
        label = left if left != "still" else right
        reference_labels.append(label)
        reference_ranges.append(
            (int(segment["start_frame"]), int(segment["end_frame"]))
        )
    correct = 0
    details = []
    for segment in predicted_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        overlaps = [
            max(0, min(end, ref_end) - max(start, ref_start) + 1)
            for ref_start, ref_end in reference_ranges
        ]
        index = int(np.argmax(overlaps))
        predicted = str(segment.get("label", ""))
        reference = reference_labels[index]
        normalized_predicted = (
            "grasp" if predicted in {"close_gripper", "grasp"} else predicted
        )
        normalized_reference = (
            "grasp" if reference in {"close_gripper", "grasp"} else reference
        )
        match = normalized_predicted == normalized_reference
        correct += int(match)
        details.append(
            {
                "predicted": predicted,
                "reference": reference,
                "overlap_frames": overlaps[index],
                "match": match,
            }
        )
    return {
        "reference_type": "weak_segmentation_reference",
        "interpretation": (
            "majority-overlap label agreement with a weak reference; "
            "not semantic-action ground-truth accuracy"
        ),
        "segment_majority_accuracy": correct / len(predicted_segments)
        if predicted_segments
        else 0.0,
        "details": details,
    }
