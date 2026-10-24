"""Recovery-specific manifests and dataset adapters.

This module is intentionally separate from the success-only manifest in
``atomic_seg.io``.  Recovery boundaries carry semantic and causal information
that cannot be recovered from end-effector kinematics alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atomic_seg.io import dataset_info, load_episode, load_episode_tasks, load_json

RECOVERY_MANIFEST_SCHEMA = "recovery_cot_manifest"
RECOVERY_OUTCOME_STATUSES = {
    "success",
    "failed_no_effect",
    "failed_different",
    "partial",
    "unknown",
}
DEFAULT_RECOVERY_CAMERAS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _flatten_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_names(item))
        return output
    return []


def recovery_state_layout(root: Path | str) -> tuple[str, ...]:
    """Return the declared state names without imposing a 16-D layout."""

    info = dataset_info(root)
    feature = info.get("features", {}).get("observation.state", {})
    shape = list(feature.get("shape") or [])
    if len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(f"{root}: observation.state must have one positive dimension")
    dimension = int(shape[0])
    names = _flatten_names(feature.get("names"))
    if len(names) != dimension:
        names = [f"state_{index}" for index in range(dimension)]
    return tuple(names)


def recovery_video_path(
    root: Path | str,
    episode_index: int,
    camera_key: str,
) -> Path:
    episode = int(episode_index)
    relative = (
        Path(f"chunk-{episode // 1000:03d}")
        / f"observation.images.{camera_key}"
        / f"episode_{episode:06d}.mp4"
    )
    for directory in ("videos", "videos_old"):
        candidate = Path(root) / directory / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(Path(root) / "videos" / relative)


@dataclass(frozen=True)
class PreviousOutcome:
    """Audited result of the atomic subtask preceding a recovery row."""

    status: str
    expected_subtask: str
    observed_outcome: str = ""
    mismatch: str = ""
    evidence: tuple[str, ...] = ()
    annotation_source: str = "manifest"
    history_frame: int | None = None
    outcome_frame: int | None = None
    expected_primary_action_verb: str = ""

    def __post_init__(self) -> None:
        if self.status not in RECOVERY_OUTCOME_STATUSES:
            raise ValueError(f"unsupported previous outcome status {self.status!r}")
        if not self.expected_subtask.strip():
            raise ValueError("recovery outcomes require expected_subtask")
        if (
            self.status
            in {
                "failed_no_effect",
                "failed_different",
                "partial",
            }
            and not self.observed_outcome.strip()
        ):
            raise ValueError(f"{self.status} requires an observed_outcome")
        if (
            self.status
            in {
                "failed_no_effect",
                "failed_different",
                "partial",
            }
            and not self.mismatch.strip()
        ):
            raise ValueError(f"{self.status} requires a mismatch description")
        for name, frame in (
            ("history_frame", self.history_frame),
            ("outcome_frame", self.outcome_frame),
        ):
            if frame is not None and frame < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RecoverySegment:
    """One authoritative recovery decision segment in dataset frame space."""

    segment_index: int
    start_frame: int
    end_frame: int
    label: str
    sub_task: str
    primary_action_verb: str
    recovery_role: str
    previous_outcome: PreviousOutcome
    boundary_source: str

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("recovery segment frame range is invalid")
        if not self.label.strip() or not self.sub_task.strip():
            raise ValueError("recovery segment label/sub_task must be non-empty")
        if not self.primary_action_verb.strip():
            raise ValueError("primary_action_verb must be non-empty")


@dataclass(frozen=True)
class RecoveryEpisode:
    """One audited failure-to-recovery episode specification."""

    data_root: Path
    episode_index: int
    episode_id: str
    task: str
    recovery_level: int
    recovery_level_name: str
    failure_type: str
    source_failure_verified: bool
    recovery_success: bool
    training_eligible: bool
    provenance: Mapping[str, Any]
    state_layout: tuple[str, ...]
    camera_keys: tuple[str, ...]
    segments: tuple[RecoverySegment, ...]

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self.episode_id or not self.task:
            raise ValueError("recovery episode id/task must be non-empty")
        if self.recovery_level not in (1, 2, 3, 4):
            raise ValueError("recovery_level must be one of 1, 2, 3, 4")
        if not self.source_failure_verified:
            raise ValueError("recovery CoT requires a verified source failure")
        if not self.recovery_success:
            raise ValueError("recovery CoT currently requires successful recovery")
        if not self.training_eligible:
            raise ValueError("recovery pair is not eligible for training")
        if not self.segments:
            raise ValueError("recovery episode contains no semantic segments")
        previous_end: int | None = None
        for expected_index, segment in enumerate(self.segments):
            if segment.segment_index != expected_index:
                raise ValueError("recovery segment indices must be contiguous")
            if previous_end is not None and segment.start_frame != previous_end + 1:
                raise ValueError(
                    "recovery semantic segments must be contiguous in dataset "
                    "frame space"
                )
            previous_end = segment.end_frame


def load_recovery_source(episode: RecoveryEpisode) -> dict[str, Any]:
    """Load the columns needed by recovery action and CoT generation."""

    return load_episode(
        episode.data_root,
        episode.episode_index,
        columns=(
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "task_index",
            "phase",
            "source_episode_id",
            "source_frame_index",
            "action_is_supervised",
            "observation_only",
            "task_success",
            "recovery_active",
        ),
    )


def _find_paired_frame(
    source: Mapping[str, Any],
    *,
    source_episode_id: str,
    source_frame_index: int,
) -> int:
    ids = source.get("source_episode_id")
    frames = source.get("source_frame_index")
    if ids is None or frames is None:
        raise ValueError(
            "native recovery adaptation requires source_episode_id and "
            "source_frame_index columns"
        )
    matches = [
        index
        for index, (episode_id, frame) in enumerate(zip(ids, frames))
        if str(episode_id) == source_episode_id
        and int(frame) == int(source_frame_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one paired row for "
            f"{source_episode_id}:{source_frame_index}, got {matches}"
        )
    return matches[0]


def _sentence_from_label(label: str) -> str:
    text = " ".join(label.replace("-", "_").split("_")).strip()
    if not text:
        raise ValueError("semantic segment label must be non-empty")
    return text[0].upper() + text[1:] + "."


def _role_from_label(label: str, index: int) -> str:
    lower = label.lower()
    if "confirm" in lower or "verify" in lower or "observe" in lower:
        return "verify_recovery"
    if "retry" in lower or "reattempt" in lower:
        return "retry_failed_subtask"
    if any(
        token in lower
        for token in (
            "withdraw",
            "recenter",
            "relocalize",
            "clear",
            "restore",
            "regrasp",
        )
    ):
        return "repair_precondition"
    return "start_recovery" if index == 0 else "resume_original_task"


def _verb_from_label(label: str) -> str:
    lower = label.lower()
    if any(token in lower for token in ("confirm", "observe", "verify", "wait")):
        return "recover:verify"
    if any(token in lower for token in ("retry", "reattempt", "press")):
        return "recover:retry"
    if any(token in lower for token in ("restore", "regrasp", "recover")):
        return "recover:restore"
    return "recover:reposition"


def _parse_outcome(value: Mapping[str, Any]) -> PreviousOutcome:
    return PreviousOutcome(
        status=str(value.get("status", "")).strip(),
        expected_subtask=str(value.get("expected_subtask", "")).strip(),
        observed_outcome=str(value.get("observed_outcome", "")).strip(),
        mismatch=str(value.get("mismatch", "")).strip(),
        evidence=tuple(str(item).strip() for item in value.get("evidence", [])),
        annotation_source=str(value.get("annotation_source", "manifest")).strip(),
        history_frame=(
            int(value["history_frame"])
            if value.get("history_frame") is not None
            else None
        ),
        outcome_frame=(
            int(value["outcome_frame"])
            if value.get("outcome_frame") is not None
            else None
        ),
        expected_primary_action_verb=str(
            value.get("expected_primary_action_verb", "")
        ).strip(),
    )


def _segment_override(
    overrides: Mapping[str, Any],
    index: int,
    label: str,
) -> Mapping[str, Any]:
    value = overrides.get(str(index), overrides.get(label, {}))
    if not isinstance(value, Mapping):
        raise ValueError(f"segment override {index}/{label!r} must be an object")
    return value


def _adapt_native_pair(
    item: Mapping[str, Any],
    *,
    base: Path,
) -> RecoveryEpisode:
    data_root = _resolve_path(item["data_root"], base)
    episode_index = int(item.get("episode_index", 0))
    recovery_start_frame = (
        int(item["recovery_start_frame"])
        if item.get("recovery_start_frame") is not None
        else None
    )
    native_path = _resolve_path(item["recovery_episode_json"], base)
    scenario_path = _resolve_path(item["scenario_json"], base)
    native = load_json(native_path)
    scenario = load_json(scenario_path)
    if not isinstance(native, Mapping) or not isinstance(scenario, Mapping):
        raise ValueError("native recovery and scenario JSON must contain objects")

    require_audited = item.get("require_audited", True) is not False
    if require_audited:
        audit_requirements = {
            "eligible_for_training": native.get("eligible_for_training") is True,
            "pair_audit_status": native.get("pair_audit_status") == "passed",
            "source_failure_verified": native.get("source_failure_verified") is True,
            "source_failure_mode": native.get("source_failure_mode")
            == "execution_failure",
            "recovery_success": native.get("recovery_success") is True,
            "new_decision_boundary": native.get("new_decision_boundary") is True,
            "direct_continuation": native.get("direct_continuation") is False,
        }
        failed = [name for name, passed in audit_requirements.items() if not passed]
        if failed:
            raise ValueError(
                f"{native_path}: recovery pair failed audit gates {failed}"
            )

    native_segments = native.get("semantic_segments")
    if not isinstance(native_segments, list) or not native_segments:
        raise ValueError(
            f"{native_path}: no semantic_segments; add explicit manifest "
            "segments after human/collector boundary annotation"
        )
    transition_policy = str(item.get("transition_outcome_policy", "require_explicit"))
    if transition_policy not in {
        "require_explicit",
        "audited_successful_semantic_boundaries",
    }:
        raise ValueError(
            "transition_outcome_policy must be require_explicit or "
            "audited_successful_semantic_boundaries"
        )

    source = load_episode(
        data_root,
        episode_index,
        columns=(
            "observation.state",
            "action",
            "source_episode_id",
            "source_frame_index",
            "action_is_supervised",
            "observation_only",
        ),
    )
    frame_count = len(source["observation.state"])
    recovery_episode_id = str(native["episode_id"])
    annotation = scenario.get("annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError(f"{scenario_path}: missing annotation object")
    source_episode_id = Path(str(native["source_episode_dir"])).name
    history_frame = _find_paired_frame(
        source,
        source_episode_id=source_episode_id,
        source_frame_index=int(annotation["attempt_start_step"]),
    )
    failure_outcome_frame = _find_paired_frame(
        source,
        source_episode_id=source_episode_id,
        source_frame_index=int(annotation["failure_outcome_step"]),
    )
    failed_subtask = str(item.get("failed_subtask", "")).strip()
    if not failed_subtask:
        raise ValueError(
            "native recovery adaptation requires failed_subtask so the "
            "expected action is not guessed from kinematics"
        )
    failure_status = str(item.get("failure_outcome_status", "")).strip()
    if failure_status not in {
        "failed_no_effect",
        "failed_different",
        "partial",
        "unknown",
    }:
        raise ValueError(
            "native recovery adaptation requires an explicit failure_outcome_status"
        )
    failure_mismatch = str(item.get("failure_mismatch", "")).strip()
    if failure_status != "unknown" and not failure_mismatch:
        raise ValueError("native recovery adaptation requires failure_mismatch")
    first_outcome = PreviousOutcome(
        status=failure_status,
        expected_subtask=failed_subtask,
        expected_primary_action_verb=str(
            item.get("failed_primary_action_verb", "")
        ).strip(),
        observed_outcome=str(annotation.get("summary", "")).strip(),
        mismatch=failure_mismatch,
        evidence=tuple(str(value).strip() for value in annotation.get("evidence", [])),
        annotation_source=str(annotation.get("source", "recovery_scenario_annotation")),
        history_frame=history_frame,
        outcome_frame=failure_outcome_frame,
    )

    overrides = item.get("segment_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("segment_overrides must be an object")
    allow_label_derived = item.get("allow_label_derived_targets") is True
    explicit_outcomes = item.get("transition_outcomes", {})
    if not isinstance(explicit_outcomes, Mapping):
        raise ValueError("transition_outcomes must be an object")

    prepared: list[dict[str, Any]] = []
    for index, raw in enumerate(native_segments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"semantic_segments[{index}] must be an object")
        label = str(raw.get("label", "")).strip()
        override = _segment_override(overrides, index, label)
        missing_target_fields = [
            field
            for field in (
                "sub_task",
                "primary_action_verb",
                "recovery_role",
            )
            if not str(override.get(field, "")).strip()
        ]
        if missing_target_fields and not allow_label_derived:
            raise ValueError(
                f"semantic segment {index} ({label!r}) requires explicit "
                f"target overrides {missing_target_fields}"
            )
        start = _find_paired_frame(
            source,
            source_episode_id=recovery_episode_id,
            source_frame_index=int(raw["start_step"]),
        )
        end = _find_paired_frame(
            source,
            source_episode_id=recovery_episode_id,
            source_frame_index=int(raw["end_step"]),
        )
        prepared.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "sub_task": str(
                    override.get("sub_task") or _sentence_from_label(label)
                ).strip(),
                "primary_action_verb": str(
                    override.get("primary_action_verb") or _verb_from_label(label)
                ).strip(),
                "recovery_role": str(
                    override.get("recovery_role") or _role_from_label(label, index)
                ).strip(),
            }
        )

    if recovery_start_frame is not None:
        offset = recovery_start_frame - int(prepared[0]["start"])
        for value in prepared:
            value["start"] = int(value["start"]) + offset
            value["end"] = int(value["end"]) + offset
    if prepared[0]["start"] < 0 or prepared[-1]["end"] >= frame_count:
        raise ValueError(
            "mapped recovery semantic segments fall outside the dataset "
            f"episode length {frame_count}"
        )

    segments: list[RecoverySegment] = []
    for index, value in enumerate(prepared):
        if index == 0:
            previous_outcome = first_outcome
        else:
            explicit = explicit_outcomes.get(str(index))
            if explicit is not None:
                if not isinstance(explicit, Mapping):
                    raise ValueError(f"transition_outcomes[{index}] must be an object")
                previous_outcome = _parse_outcome(explicit)
            elif transition_policy == "audited_successful_semantic_boundaries":
                prior = prepared[index - 1]
                previous_outcome = PreviousOutcome(
                    status="success",
                    expected_subtask=str(prior["sub_task"]),
                    observed_outcome=(
                        "The audited recovery trajectory reached the next "
                        "annotated semantic decision boundary."
                    ),
                    annotation_source=("audited_successful_recovery_semantic_boundary"),
                    history_frame=int(prior["start"]),
                    outcome_frame=int(value["start"]),
                    expected_primary_action_verb=str(prior["primary_action_verb"]),
                )
            else:
                raise ValueError(
                    f"segment {index} requires an explicit previous outcome"
                )
        segments.append(
            RecoverySegment(
                segment_index=index,
                start_frame=int(value["start"]),
                end_frame=int(value["end"]),
                label=str(value["label"]),
                sub_task=str(value["sub_task"]),
                primary_action_verb=str(value["primary_action_verb"]),
                recovery_role=str(value["recovery_role"]),
                previous_outcome=previous_outcome,
                boundary_source="native_audited_semantic_segments",
            )
        )

    task = str(
        item.get("task") or native.get("instruction") or scenario.get("instruction")
    ).strip()
    camera_keys = tuple(
        str(value) for value in item.get("camera_keys", DEFAULT_RECOVERY_CAMERAS)
    )
    return RecoveryEpisode(
        data_root=data_root,
        episode_index=episode_index,
        episode_id=recovery_episode_id,
        task=task,
        recovery_level=int(native["recovery_level"]),
        recovery_level_name=str(
            native.get("recovery_level_name", f"L{native['recovery_level']}")
        ),
        failure_type=str(
            native.get("failure_type", annotation.get("failure_type", ""))
        ),
        source_failure_verified=native.get("source_failure_verified") is True,
        recovery_success=native.get("recovery_success") is True,
        training_eligible=native.get("eligible_for_training") is True,
        provenance={
            "adapter": "native_recovery_pair",
            "manifest_entry": dict(item),
            "recovery_episode_json": str(native_path),
            "scenario_json": str(scenario_path),
            "pair_audit_status": native.get("pair_audit_status"),
            "source_episode_id": source_episode_id,
            "source_failure_outcome_frame": failure_outcome_frame,
            "transition_outcome_policy": transition_policy,
            "allow_label_derived_targets": allow_label_derived,
            "recovery_start_frame_override": recovery_start_frame,
        },
        state_layout=recovery_state_layout(data_root),
        camera_keys=camera_keys,
        segments=tuple(segments),
    )


def _parse_direct_episode(
    item: Mapping[str, Any],
    *,
    base: Path,
) -> RecoveryEpisode:
    data_root = _resolve_path(item["data_root"], base)
    episode_index = int(item.get("episode_index", 0))
    raw_segments = item.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("direct recovery episode requires a segments array")
    segments: list[RecoverySegment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ValueError(f"segments[{index}] must be an object")
        outcome = raw.get("previous_outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError(f"segments[{index}] requires previous_outcome annotation")
        label = str(raw.get("label") or raw.get("sub_task", "")).strip()
        segments.append(
            RecoverySegment(
                segment_index=index,
                start_frame=int(raw["start_frame"]),
                end_frame=int(raw["end_frame"]),
                label=label,
                sub_task=str(raw.get("sub_task") or _sentence_from_label(label)),
                primary_action_verb=str(
                    raw.get("primary_action_verb") or _verb_from_label(label)
                ),
                recovery_role=str(
                    raw.get("recovery_role") or _role_from_label(label, index)
                ),
                previous_outcome=_parse_outcome(outcome),
                boundary_source=str(
                    raw.get("boundary_source", "manifest_semantic_segments")
                ),
            )
        )
    task = str(item.get("task", "")).strip()
    if not task:
        task = load_episode_tasks(data_root).get(episode_index, "")
    return RecoveryEpisode(
        data_root=data_root,
        episode_index=episode_index,
        episode_id=str(item.get("episode_id", f"recovery_episode_{episode_index:06d}")),
        task=task,
        recovery_level=int(item["recovery_level"]),
        recovery_level_name=str(
            item.get("recovery_level_name", f"L{item['recovery_level']}")
        ),
        failure_type=str(item["failure_type"]),
        source_failure_verified=item.get("source_failure_verified") is True,
        recovery_success=item.get("recovery_success") is True,
        training_eligible=item.get("training_eligible") is True,
        provenance={
            "adapter": "direct_manifest",
            "manifest_entry": dict(item),
        },
        state_layout=recovery_state_layout(data_root),
        camera_keys=tuple(
            str(value) for value in item.get("camera_keys", DEFAULT_RECOVERY_CAMERAS)
        ),
        segments=tuple(segments),
    )


def load_recovery_manifest(path: Path | str) -> list[RecoveryEpisode]:
    """Load an isolated recovery manifest and fail closed on weak labels."""

    manifest_path = Path(path).expanduser().resolve()
    value = load_json(manifest_path)
    if not isinstance(value, Mapping):
        raise ValueError("recovery manifest must contain an object")
    if value.get("schema") != RECOVERY_MANIFEST_SCHEMA:
        raise ValueError(
            f"recovery manifest schema must be {RECOVERY_MANIFEST_SCHEMA!r}"
        )
    raw_episodes = value.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("recovery manifest must contain a non-empty episodes array")
    episodes: list[RecoveryEpisode] = []
    seen: set[tuple[Path, int]] = set()
    for index, item in enumerate(raw_episodes):
        if not isinstance(item, Mapping):
            raise ValueError(f"episodes[{index}] must be an object")
        adapter = str(item.get("adapter", "direct"))
        if adapter == "native_recovery_pair":
            episode = _adapt_native_pair(item, base=manifest_path.parent)
        elif adapter == "direct":
            episode = _parse_direct_episode(item, base=manifest_path.parent)
        else:
            raise ValueError(f"unsupported recovery adapter {adapter!r}")
        key = (episode.data_root, episode.episode_index)
        if key in seen:
            raise ValueError(
                "duplicate recovery dataset episode "
                f"{episode.data_root}:{episode.episode_index}"
            )
        seen.add(key)
        episodes.append(episode)
    return episodes


def validate_recovery_segment_rows(
    source: Mapping[str, Any],
    segment: RecoverySegment,
) -> None:
    """Reject targets that include context-only or unsupervised rows."""

    states = np.asarray(source["observation.state"])
    actions = np.asarray(source["action"])
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError(
            "recovery observation.state/action must have matching [T,D] shape"
        )
    if segment.end_frame >= len(states):
        raise ValueError(
            f"segment {segment.segment_index} exceeds episode length {len(states)}"
        )
    selection = slice(segment.start_frame, segment.end_frame + 1)
    supervised = source.get("action_is_supervised")
    if supervised is not None and not np.asarray(supervised)[selection].all():
        raise ValueError(
            f"segment {segment.segment_index} contains unsupervised actions"
        )
    observation_only = source.get("observation_only")
    if observation_only is not None and np.asarray(observation_only)[selection].any():
        raise ValueError(
            f"segment {segment.segment_index} contains observation-only rows"
        )
