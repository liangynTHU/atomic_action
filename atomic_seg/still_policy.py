"""Post-process atomic timelines so idle holds do not become action fragments.

The state segmenter may find real stationary intervals between motions.  Those
intervals are useful boundary evidence, but an initial or intermediate
``left:still, right:still`` interval is not a useful standalone policy target.
This module assigns such intervals to a neighbouring atomic action while
retaining one sufficiently long terminal hold as the optional final stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SegmentGroup:
    """One output segment assembled from one or more input segments."""

    owner_index: int
    source_indices: tuple[int, ...]
    start_frame: int
    end_frame: int
    absorbed_still_indices: tuple[int, ...]
    absorbed_still_roles: tuple[str, ...]


def is_both_still(segment: Mapping[str, Any]) -> bool:
    return (
        str(segment["left_action"]) == "still"
        and str(segment["right_action"]) == "still"
    )


def _frame_count(segment: Mapping[str, Any]) -> int:
    return int(segment["end_frame"]) - int(segment["start_frame"]) + 1


def plan_still_segment_groups(
    segments: Sequence[Mapping[str, Any]],
    *,
    minimum_terminal_still_frames: int = 8,
) -> list[SegmentGroup]:
    """Plan a contiguous timeline with no initial or middle both-still target.

    Policy:

    * an initial both-still run is pre-context for the first active action;
    * a middle both-still run is settling time for the preceding action;
    * a terminal both-still run is retained as one stage only when it is at
      least ``minimum_terminal_still_frames`` long, otherwise it is settling
      time for the preceding action;
    * consecutive terminal still segments are collapsed into one stage.

    The action/sub-task semantics of each output segment come from
    ``owner_index``.  The source frame ranges are never dropped.
    """

    if minimum_terminal_still_frames < 1:
        raise ValueError("minimum_terminal_still_frames must be positive")
    if not segments:
        return []

    previous_end: int | None = None
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if end < start:
            raise ValueError(f"segment {index} has an invalid frame range")
        if previous_end is not None and start != previous_end + 1:
            raise ValueError("input segments must be contiguous")
        previous_end = end

    groups: list[dict[str, Any]] = []
    pending_initial: list[int] = []
    index = 0
    while index < len(segments):
        if not is_both_still(segments[index]):
            source_indices = [*pending_initial, index]
            groups.append(
                {
                    "owner_index": index,
                    "source_indices": source_indices,
                    "absorbed_still_indices": list(pending_initial),
                    "absorbed_still_roles": ["pre_context" for _ in pending_initial],
                }
            )
            pending_initial.clear()
            index += 1
            continue

        run_start = index
        while index < len(segments) and is_both_still(segments[index]):
            index += 1
        run_indices = list(range(run_start, index))
        at_start = run_start == 0 and not groups
        at_end = index == len(segments)

        if at_start and not at_end:
            pending_initial.extend(run_indices)
            continue

        run_frames = sum(_frame_count(segments[item]) for item in run_indices)
        keep_terminal = at_end and (
            run_frames >= minimum_terminal_still_frames or not groups
        )
        if keep_terminal:
            groups.append(
                {
                    "owner_index": run_indices[0],
                    "source_indices": run_indices,
                    "absorbed_still_indices": run_indices[1:],
                    "absorbed_still_roles": [
                        "terminal_collapse" for _ in run_indices[1:]
                    ],
                }
            )
            continue

        if not groups:
            # This is reachable only for an all-still timeline shorter than
            # the terminal threshold.  An episode cannot have zero segments,
            # so retain the sole terminal stage.
            groups.append(
                {
                    "owner_index": run_indices[0],
                    "source_indices": run_indices,
                    "absorbed_still_indices": run_indices[1:],
                    "absorbed_still_roles": [
                        "terminal_collapse" for _ in run_indices[1:]
                    ],
                }
            )
            continue

        groups[-1]["source_indices"].extend(run_indices)
        groups[-1]["absorbed_still_indices"].extend(run_indices)
        groups[-1]["absorbed_still_roles"].extend("post_settle" for _ in run_indices)

    if pending_initial:
        # Defensive fallback for an all-still timeline; the main loop handles
        # this as a terminal run, but keeping the guard makes the invariant
        # explicit if the loop is changed later.
        groups.append(
            {
                "owner_index": pending_initial[0],
                "source_indices": list(pending_initial),
                "absorbed_still_indices": list(pending_initial[1:]),
                "absorbed_still_roles": [
                    "terminal_collapse" for _ in pending_initial[1:]
                ],
            }
        )

    output: list[SegmentGroup] = []
    for group in groups:
        source_indices = tuple(int(item) for item in group["source_indices"])
        output.append(
            SegmentGroup(
                owner_index=int(group["owner_index"]),
                source_indices=source_indices,
                start_frame=int(segments[source_indices[0]]["start_frame"]),
                end_frame=int(segments[source_indices[-1]]["end_frame"]),
                absorbed_still_indices=tuple(
                    int(item) for item in group["absorbed_still_indices"]
                ),
                absorbed_still_roles=tuple(
                    str(item) for item in group["absorbed_still_roles"]
                ),
            )
        )

    if output[0].start_frame != int(segments[0]["start_frame"]):
        raise AssertionError("refined timeline lost its first frame")
    if output[-1].end_frame != int(segments[-1]["end_frame"]):
        raise AssertionError("refined timeline lost its final frame")
    for previous, current in zip(output, output[1:]):
        if current.start_frame != previous.end_frame + 1:
            raise AssertionError("refined timeline is not contiguous")
    for group in output[:-1]:
        if is_both_still(segments[group.owner_index]):
            raise AssertionError("nonterminal both-still segment survived")
    return output
