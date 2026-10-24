"""Generate and publish normal RoboTwin CoT with grounded dual critic checks."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
import shutil

from atomic_seg.common.artifacts import (
    atomic_write_json, compact_json, iter_jsonl, repair_jsonl_tail, sha256_file, utc_now,
)
from atomic_seg.common.locking import locked_output
from .client import Journal, ModelClient
from .evidence import (
    ANNOTATION_SCHEMA, QUALITY_POLICY, digest, freeze_row, normalize_plan_reference,
    request_payload, validate_annotation, validate_critic, validate_observer, validate_writer,
)
from .prompts import PROMPT_VERSION
from .styles import STYLE_SPECS
from .validation import _semantic_issues, _word_count

SCHEMA_VERSION = "robotwin_grounded_cot_generation_v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-Flash-Next"


class AnnotationRejected(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"row {record['row_index']}: {record['status']}")


def style_id_for_row(row_index):
    if row_index < 0:
        raise ValueError("row_index must be non-negative")
    return row_index % len(STYLE_SPECS)


def generate_annotation(row_index, row, *, endpoints, model, timeout, max_tokens,
                        max_rounds, seed_salt=0, style_id_override=None, client=None):
    style_id = style_id_for_row(row_index) if style_id_override is None else style_id_override
    if type(style_id) is not int or not 0 <= style_id < len(STYLE_SPECS):
        raise ValueError("Invalid style ID")
    if client is None:
        raise ValueError("An audited model client is required")
    frozen = freeze_row(row)
    record = {
        "row_index": row_index, "source_row_sha256": frozen["source_row_sha256"],
        "status": "RUNNING", "observer_attempts": [], "rounds": [],
    }
    observer, observer_receipt = None, None
    feedback = []
    for attempt in range(1, 4):
        entry = {"attempt": attempt}
        record["observer_attempts"].append(entry)
        try:
            value, receipt = client.call(
                "observe", frozen, request_payload("observe", frozen, feedback=feedback),
                row_index=row_index, attempt=attempt,
            )
            entry.update(response=value, receipt=receipt)
            validate_observer(value, frozen)
            observer, observer_receipt = value, receipt
            break
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            feedback = [{"structural_error": str(exc)}]
    if observer is None:
        record["status"] = "OBSERVER_EXHAUSTED"
        raise AnnotationRejected(record)
    record.update(observer=observer, observer_receipt=observer_receipt)
    feedback, last_draft = [], None
    style = STYLE_SPECS[style_id]["name"]
    for attempt in range(1, max_rounds + 1):
        entry = {"round": attempt, "status": "REVISE"}
        record["rounds"].append(entry)
        try:
            value, receipt = client.call(
                "write", frozen,
                request_payload("write", frozen, observer=observer, style=style,
                                feedback=feedback, previous_draft=last_draft),
                row_index=row_index, attempt=attempt,
            )
            writer, changes = normalize_plan_reference(value, "write")
            entry.update(writer=writer, writer_receipt=receipt, metadata_normalizations=changes)
            validate_writer(writer, frozen, observer)
            if writer["status"] != "CANDIDATE":
                record["status"] = "REVIEW_" + writer["status"]
                raise AnnotationRejected(record)
            last_draft = writer["reasoning"]
            issues = _semantic_issues(last_draft, frozen["target"], style_id)
            if issues:
                feedback = [{"action_error": issue} for issue in issues]
                entry["issues"] = issues
                continue

            def critic(stage):
                result, receipt = client.call(
                    stage, frozen, request_payload(stage, frozen, reasoning=last_draft),
                    row_index=row_index, attempt=attempt,
                )
                result, changes = normalize_plan_reference(result, stage)
                review = {"review": result, "receipt": receipt,
                          "metadata_normalizations": changes}
                try:
                    validate_critic(result, frozen, last_draft, stage)
                except ValueError as exc:
                    review["validation_error"] = str(exc)
                return review

            reviews = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(critic, stage): stage
                           for stage in ("grounding_critic", "action_critic")}
                for future in concurrent.futures.as_completed(futures):
                    stage = futures[future]
                    try:
                        reviews[stage] = future.result()
                    except Exception as exc:
                        reviews[stage] = {"error": f"{type(exc).__name__}: {exc}"}
            entry["critics"] = reviews
            feedback = []
            for stage, review in reviews.items():
                if review.get("error") or review.get("validation_error"):
                    feedback.append({"critic": stage, "structural_error":
                                     review.get("error", review.get("validation_error"))})
                elif review["review"]["verdict"] != "PASS":
                    feedback.append({"critic": stage, **review["review"]})
            if any(review.get("review", {}).get("verdict") in {
                "TARGET_CONFLICT", "INSUFFICIENT_EVIDENCE",
            } for review in reviews.values()):
                record["status"] = "REVIEW_REQUIRED"
                raise AnnotationRejected(record)
            if feedback:
                continue
            entry["status"] = "PASS"
            annotation = {
                "schema_version": ANNOTATION_SCHEMA, "quality_policy": QUALITY_POLICY,
                "prompt_version": PROMPT_VERSION, "model": model,
                "row_index": row_index, "episode_index": int(row["episode_index"]),
                "segment_index": int(row["segment_index"]), "style_id": style_id,
                "style_name": style, "reasoning": last_draft,
                "word_count": _word_count(last_draft),
                "reasoning_sha256": hashlib.sha256(last_draft.encode()).hexdigest(),
                "accepted_round": attempt, "observer": observer,
                "observer_receipt": observer_receipt,
                "observer_attempts": record["observer_attempts"], "rounds": record["rounds"],
                "frozen_hashes": {key: frozen[key] for key in (
                    "source_row_sha256", "student_visible_sha256", "final_json_sha256",
                )},
            }
            validate_annotation(annotation, row, frozen=frozen)
            return annotation
        except AnnotationRejected:
            raise
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            feedback.append({"structural_error": str(exc),
                             "instruction": "Preserve substantive corrections and recheck the images."})
    record["status"] = "WRITER_EXHAUSTED"
    raise AnnotationRejected(record)


def _read_endpoints(path):
    values = list(dict.fromkeys(
        line.strip() for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ))
    if not values:
        raise ValueError("Empty endpoint file")
    return values


def _result_index(path):
    entries = {}
    if path.is_file():
        with path.open("rb") as stream:
            while raw := stream.readline():
                value = json.loads(raw)
                index = value["row_index"]
                if value["status"] == "ACCEPTED":
                    if index in entries:
                        raise ValueError("Duplicate accepted result in journal")
                    entries[index] = (stream.tell() - len(raw), len(raw))
    return entries


@locked_output("annotation_path")
def generate_annotations(*, input_vlm, annotation_path, endpoint_file, model=DEFAULT_MODEL,
                         workers=64, batch_size=256, timeout=600.0, max_tokens=3072,
                         max_rounds=4, job_attempts=3, limit_rows=None, start_row=0,
                         end_row=None, resume=False):
    if min(workers, batch_size, job_attempts, max_rounds, max_tokens) < 1 or timeout <= 0:
        raise ValueError("Concurrency, token budgets and retry counts must be positive")
    if start_row < 0 or (end_row is not None and end_row <= start_row) or (
        limit_rows is not None and limit_rows < 0
    ):
        raise ValueError("Invalid source row range")
    stop_row = end_row
    if limit_rows is not None:
        stop_row = min(end_row, start_row + limit_rows) if end_row is not None else start_row + limit_rows
    input_vlm, annotation_path = Path(input_vlm), Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = annotation_path.with_name(annotation_path.name + ".run.json")
    result_path = annotation_path.with_name(annotation_path.name + ".results.jsonl")
    request_path = annotation_path.with_name(annotation_path.name + ".requests.jsonl")
    review_path = annotation_path.with_name(annotation_path.name + ".review.jsonl")
    package = Path(__file__).parent
    identity = {
        "quality_policy": QUALITY_POLICY, "prompt_version": PROMPT_VERSION,
        "input_vlm": str(input_vlm.resolve()), "input_sha256": sha256_file(input_vlm),
        "model": model, "max_tokens": max_tokens, "max_rounds": max_rounds,
        "source_start_row": start_row,
        "code": {str(p.relative_to(package.parent)): sha256_file(p) for p in sorted([
            *package.glob("*.py"), *(package.parent / "common").glob("*.py"),
            package.parent / "prompt_contract.py", package.parent / "action_language.py",
        ])},
    }
    if config_path.exists():
        if not resume:
            raise FileExistsError("Existing generation; use --resume")
        if json.loads(config_path.read_text()) != identity:
            raise ValueError("Inputs, model or quality contract changed; use a new output")
    else:
        if any(path.exists() for path in (annotation_path, result_path, request_path, review_path)):
            raise FileExistsError("Generation outputs lack a matching run identity")
        atomic_write_json(config_path, identity)
    if resume and annotation_path.exists():
        repair_jsonl_tail(annotation_path)
    completed_count = 0
    if annotation_path.exists():
        for annotation in iter_jsonl(annotation_path):
            if annotation["row_index"] != start_row + completed_count:
                raise ValueError("Completed annotation indices are not contiguous")
            completed_count += 1
    next_row = start_row + completed_count
    if stop_row is not None and next_row > stop_row:
        raise ValueError("Completed annotations exceed requested range")
    completed_iter = iter_jsonl(annotation_path) if completed_count else iter(())
    results_journal, request_journal = Journal(result_path), Journal(request_path)
    review_journal = Journal(review_path)
    saved_results = _result_index(result_path)
    client = ModelClient(
        _read_endpoints(endpoint_file), model=model, timeout=timeout,
        max_tokens=max_tokens, journal=request_journal, concurrency=workers,
    )
    failures = {}
    with input_vlm.open() as source, annotation_path.open("a") as destination:
        skipped = 0
        while skipped < next_row:
            raw = source.readline()
            if not raw:
                raise ValueError("Requested source row is beyond input")
            if raw.strip():
                if skipped >= start_row:
                    validate_annotation(next(completed_iter), json.loads(raw),
                                        journal_root=annotation_path.parent)
                skipped += 1
        while stop_row is None or next_row < stop_row:
            batch = []
            while len(batch) < batch_size and (
                stop_row is None or next_row + len(batch) < stop_row
            ):
                raw = source.readline()
                if not raw:
                    break
                if raw.strip():
                    batch.append((next_row + len(batch), json.loads(raw)))
            if not batch:
                break
            pending, accepted = {}, {}
            for index, row in batch:
                if index in saved_results:
                    offset, length = saved_results[index]
                    with result_path.open("rb") as cache:
                        cache.seek(offset)
                        saved = json.loads(cache.read(length))
                    if saved["source_row_sha256"] != digest(row):
                        raise ValueError("Cached result source changed")
                    validate_annotation(saved["annotation"], row)
                    accepted[index] = saved["annotation"]
                else:
                    pending[index] = row
            for job_attempt in range(job_attempts):
                if not pending:
                    break
                current, pending = pending, {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(current))) as executor:
                    futures = {
                        executor.submit(
                            generate_annotation, index, row, endpoints=client.endpoints,
                            model=model, timeout=timeout, max_tokens=max_tokens,
                            max_rounds=max_rounds, seed_salt=job_attempt, client=client,
                        ): (index, row) for index, row in current.items()
                    }
                    for future in concurrent.futures.as_completed(futures):
                        index, row = futures[future]
                        try:
                            annotation = future.result()
                            validate_annotation(annotation, row)
                            accepted[index] = annotation
                            receipt = results_journal.append({
                                "row_index": index, "status": "ACCEPTED",
                                "source_row_sha256": annotation["frozen_hashes"]["source_row_sha256"],
                                "annotation": annotation,
                            })
                            saved_results[index] = (receipt["byte_offset"], receipt["bytes"])
                            failures.pop(index, None)
                        except Exception as exc:
                            record = exc.record if isinstance(exc, AnnotationRejected) else {
                                "row_index": index, "status": "PROCESSING_ERROR", "error": str(exc),
                            }
                            failures[index] = record
                            results_journal.append({**record, "source": row})
                            if not record["status"].startswith("REVIEW"):
                                pending[index] = row
                for index in pending:
                    print(compact_json({"event": "cot_retry", "row_index": index,
                                        "attempt": job_attempt + 1}), flush=True)
            for index, row in batch:
                if index not in accepted:
                    review_journal.append({**failures[index], "source": row})
            # Later successes remain durable when a rejected row blocks the
            # ordered annotation stream. Resume reuses those accepted results.
            for index, _row in batch:
                if index in failures or index != start_row + completed_count:
                    break
                destination.write(compact_json(accepted[index]) + "\n")
                completed_count += 1
            destination.flush()
            os.fsync(destination.fileno())
            next_row += len(batch)
            print(compact_json({"event": "cot_progress", "processed_rows": next_row,
                                "accepted_rows": len(saved_results), "review_rows": len(failures)}), flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} unresolved rows; accepted results are retained in {result_path}, "
            f"review records in {review_path}; release is blocked"
        )
    return {"annotations": completed_count, "start_row": start_row, "next_row": next_row,
            "end_row": stop_row, "endpoint_count": len(client.endpoints),
            "quality_policy": QUALITY_POLICY, "unresolved_rows": 0}


def finalize_dataset(*, input_root, output_root, annotation_path, overwrite=False,
                     require_modulo_style_allocation=True):
    input_root, output_root = Path(input_root), Path(output_root)
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and release roots must be separate")
    input_vlm = input_root / "data/vlm/train.jsonl"
    output_vlm = output_root / "data/vlm/train.jsonl"
    report_path = output_root / "COT_STYLE_REPORT.json"
    copies = ("data/vla/train.json", "metadata/source_manifest.jsonl")
    for name in copies:
        if not (input_root / name).is_file():
            raise FileNotFoundError(input_root / name)
    for path in (output_vlm, report_path, *(output_root / name for name in copies)):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    annotations = iter(iter_jsonl(annotation_path))
    output_vlm.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_vlm.with_name(f".{output_vlm.name}.tmp-{os.getpid()}")
    styles, rounds = Counter(), Counter()
    words, hashes, count = [], set(), 0
    try:
        with temporary.open("w") as destination:
            for row in iter_jsonl(input_vlm):
                annotation = next(annotations, None)
                if annotation is None or annotation["row_index"] != count:
                    raise ValueError("Annotations are incomplete or out of order")
                if (annotation["episode_index"], annotation["segment_index"]) != (
                    row["episode_index"], row["segment_index"],
                ):
                    raise ValueError("Annotation/source identity mismatch")
                if require_modulo_style_allocation and annotation["style_id"] != style_id_for_row(count):
                    raise ValueError("Per-row style allocation mismatch")
                frozen = validate_annotation(annotation, row, journal_root=Path(annotation_path).parent)
                result = copy.deepcopy(row)
                result.pop("cot_annotation_required", None)
                result["messages"][-1]["content"] = (
                    "<think>\n" + annotation["reasoning"] + " " + frozen["closing_sentence"]
                    + "\n</think>\n\n" + frozen["final_json_text"]
                )
                destination.write(compact_json(result) + "\n")
                styles[annotation["style_name"]] += 1
                rounds[annotation["accepted_round"]] += 1
                words.append(annotation["word_count"])
                hashes.add(annotation["reasoning_sha256"])
                count += 1
        if next(annotations, None) is not None or not count:
            raise ValueError("Annotation count differs from source")
        expected = {spec["name"]: count // len(STYLE_SPECS) + int(i < count % len(STYLE_SPECS))
                    for i, spec in enumerate(STYLE_SPECS)}
        if require_modulo_style_allocation and dict(styles) != {k: v for k, v in expected.items() if v}:
            raise ValueError("Style allocation mismatch")
        if not require_modulo_style_allocation and (
            set(styles) != {k for k, v in expected.items() if v}
            or max(styles.values()) - min(styles.values()) > max(1, math.ceil(count * 0.001))
        ):
            raise ValueError("Preserved styles are not near-balanced")
        os.replace(temporary, output_vlm)
    finally:
        temporary.unlink(missing_ok=True)
    for name in copies:
        target = output_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_root / name, target)
    report = {
        "schema_version": SCHEMA_VERSION, "quality_policy": QUALITY_POLICY,
        "prompt_version": PROMPT_VERSION, "created_at_utc": utc_now(),
        "input_root": str(input_root), "output_root": str(output_root),
        "styles": list(STYLE_SPECS),
        "counts": {"rows": count, "styles": dict(styles),
                   "accepted_rounds": {str(k): v for k, v in rounds.items()},
                   "unique_reasoning_hashes": len(hashes), "fallback_rows": 0},
        "quality": {"word_count_min": min(words), "word_count_mean": statistics.mean(words),
                    "word_count_max": max(words), "target_fields_unchanged": True,
                    "final_json_text_unchanged": True, "system_user_images_unchanged": True,
                    "observer_and_two_critics_verified": True},
        "hashes": {"data/vlm/train.jsonl": sha256_file(output_vlm),
                   **{name: sha256_file(output_root / name) for name in copies},
                   "annotations": sha256_file(annotation_path)},
    }
    atomic_write_json(report_path, report)
    return report
