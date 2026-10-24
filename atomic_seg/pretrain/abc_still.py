"""Apply the established both-still policy to either ABC observation layout."""

from __future__ import annotations

from atomic_seg.still_policy import is_both_still, plan_still_segment_groups


def absorb_both_still(segments, *, minimum_terminal_still_frames=8):
    groups = plan_still_segment_groups(
        segments, minimum_terminal_still_frames=minimum_terminal_still_frames,
    )
    output, absorbed = [], []
    for index, group in enumerate(groups):
        owner = segments[group.owner_index]
        item = dict(owner)
        item.update(segment_index=index, start_frame=group.start_frame, end_frame=group.end_frame)
        primitives = []
        for source_index in group.source_indices:
            source = segments[source_index]
            primitives.extend(source.get("primitive_segments", [{k: source[k] for k in (
                "segment_index", "start_frame", "end_frame", "left_action", "right_action",
            )}]))
        item["primitive_segments"] = primitives
        spans = []
        for source_index, role in zip(group.absorbed_still_indices, group.absorbed_still_roles):
            source = segments[source_index]
            spans.append({"source_segment_index": source_index, "start_frame": source["start_frame"],
                          "end_frame": source["end_frame"], "role": role, "owner_source_index": group.owner_index})
        item["absorbed_still_spans"] = spans
        item["terminal_hold"] = is_both_still(item) and index == len(groups) - 1
        item["primary_action_verb"] = f"left:{item['left_action']}, right:{item['right_action']}"
        absorbed.extend(spans)
        output.append(item)
    if any(is_both_still(s) for s in output[:-1]):
        raise ValueError("ABC retained a nonterminal both-still target")
    if sum(s["end_frame"] - s["start_frame"] + 1 for s in output) != sum(
        s["end_frame"] - s["start_frame"] + 1 for s in segments
    ):
        raise ValueError("both-still cleanup changed source row coverage")
    audit = {
        "policy": "shared plan_still_segment_groups", "minimum_terminal_still_frames": minimum_terminal_still_frames,
        "initial": "absorb_into_next_action", "middle": "absorb_into_previous_action",
        "terminal": "retain_one_hold_if_long_enough_else_absorb; all-still episode remains one hold",
        "input_segments": len(segments), "output_segments": len(output),
        "input_both_still_segments": sum(is_both_still(s) for s in segments),
        "retained_terminal_holds": sum(s["terminal_hold"] for s in output),
        "absorbed_still_spans": absorbed,
        "absorbed_still_rows": sum(s["end_frame"] - s["start_frame"] + 1 for s in absorbed),
        "nonterminal_both_still_after": 0, "all_source_rows_retained": True,
    }
    return output, audit
