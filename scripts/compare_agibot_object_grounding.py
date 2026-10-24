#!/usr/bin/env python3
"""Replay selected AgiBot episodes from a frozen run with object grounding enabled."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This comparison consumes already decoded images; no video decoding occurs.
# The source annotation module imports the video package during module loading.
if importlib.util.find_spec("av") is None:
    sys.modules["av"] = types.ModuleType("av")

from atomic_seg.pretrain.agibot.annotate import annotate
from atomic_seg.pretrain.agibot.api import check, load_token
from atomic_seg.pretrain.agibot.config import API_BASE, Settings
from atomic_seg.pretrain.agibot.contract import teacher_review_system, teacher_system
from atomic_seg.pretrain.agibot.grounding import grounding_system
from atomic_seg.pretrain.realomni.storage import atomic_json, file_hash, read_json


DEFAULT_EPISODES = (
    "task_431/episode_000047",  # coffee-machine lid versus whole machine
    "task_440/episode_000447",  # removable iron versus fixed support pole
    "task_521/episode_000216",  # rigid trash can versus bag liner
)


def load_inputs(run, episode_id):
    accepted = {}
    for line in (run / "accepted.jsonl").read_text().splitlines():
        entry = json.loads(line)
        accepted[entry["episode_id"]] = entry
    if episode_id not in accepted:
        raise ValueError(f"episode is not a committed whole-episode baseline export: {episode_id}")
    entry = accepted[episode_id]
    source = None
    for line in (run / "inventory/episodes.jsonl").open():
        candidate = json.loads(line)
        if candidate["episode_id"] == episode_id:
            source = candidate
            break
    if source is None:
        raise ValueError(f"missing inventory row: {episode_id}")
    key = source["key"]
    export = Path(entry["path"])
    fixed_path = run / "episodes" / key[:2] / key / "segmented/segmentation.json"
    media = run / "episodes" / key[:2] / key / "media"
    baseline = read_json(export / "vla.json")["segments"]
    fixed = read_json(fixed_path)
    if len(baseline) != len(fixed["segments"]) or any(
        any(old[field] != fresh[field] for field in
            ("segment_index", "start_frame", "end_frame", "left_action", "right_action"))
        for old, fresh in zip(baseline, fixed["segments"])
    ):
        raise ValueError(f"baseline segmentation differs from frozen inputs: {episode_id}")
    with np.load(export / "reference.npz", allow_pickle=False) as arrays:
        states = arrays["measured_states"].tolist()
    episode = {"task": source["task"], "measured_states": states,
               "source_annotations": [
                   {"start_frame": phase["start_frame"], "end_frame": phase["end_frame"],
                    "text": phase["action_text"], "skill": phase.get("skill")}
                   for phase in source["metadata"].get("action_config", [])]}
    hashes = {"baseline_export_vla": file_hash(export / "vla.json"),
              "fixed_segmentation": file_hash(fixed_path),
              "source_media_commit": file_hash(media / "COMMIT.json")}
    return episode, fixed, media, baseline, hashes


def run_one(run, output, episode_id, settings, endpoint, token, boundary_adjudications=None):
    episode, fixed, media, baseline, hashes = load_inputs(run, episode_id)
    target = output / episode_id.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    descriptor = {"episode_id": episode_id, "source_run": str(run),
                  "source_hashes": hashes, "object_grounding": True,
                  "segments": len(fixed["segments"]), "model": settings.model,
                  "batch_size": settings.batch_size}
    if boundary_adjudications is not None:
        descriptor["boundary_adjudications"] = boundary_adjudications
    manifest = target / "INPUTS.json"
    if manifest.exists() and read_json(manifest) != descriptor:
        raise ValueError(f"experiment inputs changed: {episode_id}")
    if not manifest.exists():
        atomic_json(manifest, descriptor)
    trial_contract = {"settings": settings.as_dict(), "grounding_system": grounding_system(),
                      "teacher_system": teacher_system(settings),
                      "review_system": teacher_review_system(True)}
    contract_path = target / "TRIAL_CONTRACT.json"
    if contract_path.exists() and read_json(contract_path) != trial_contract:
        raise ValueError(f"trial prompt or settings changed: {episode_id}")
    if not contract_path.exists():
        atomic_json(contract_path, trial_contract)
    print(f"START {episode_id}: {len(fixed['segments'])} fixed segments", flush=True)
    labels = annotate(episode, fixed, media, target / "labels", settings,
                      [endpoint], lambda done, total: print(
                          f"PROGRESS {episode_id}: {done}/{total}", flush=True), api_key=token,
                      boundary_adjudications=boundary_adjudications)
    atomic_json(target / "segments.json", labels)
    changed = sum(old["sub_task"] != new["sub_task"] for old, new in zip(baseline, labels))
    outcome = {**descriptor, "status": "complete", "changed_sub_tasks": changed,
               "trial_segments_sha256": file_hash(target / "segments.json")}
    atomic_json(target / "RESULT.json", outcome)
    print(f"DONE {episode_id}: {len(labels)} segments; {changed} changed labels", flush=True)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--api-token-file", type=Path, default=ROOT / ".api_token")
    parser.add_argument("--boundary-adjudications", type=Path,
                        help="JSON map of episode IDs to frame-level image adjudications")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    run, output = args.source_run.resolve(), args.output.resolve()
    if run == output or run in output.parents or output in run.parents:
        parser.error("source run and trial output must be separate")
    episodes = args.episode or list(DEFAULT_EPISODES)
    if len(episodes) != len(set(episodes)):
        parser.error("duplicate episode")
    source_settings = read_json(run / "run_config.json")["settings"]
    if source_settings.get("object_grounding"):
        parser.error("source run must be the original ungrounded baseline")
    settings = Settings(**{**source_settings, "object_grounding": True})
    adjudications = read_json(args.boundary_adjudications) if args.boundary_adjudications else {}
    if set(adjudications) - set(episodes):
        parser.error("boundary adjudications include an episode outside this run")
    token = load_token(args.api_token_file)
    print(json.dumps(check(args.api_base, token, settings)), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_one, run, output, episode_id, settings,
                                   args.api_base, token, adjudications.get(episode_id)): episode_id
                   for episode_id in episodes}
        for future in as_completed(futures):
            episode_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures[episode_id] = f"{type(error).__name__}: {error}"
                print(f"FAILED {episode_id}: {failures[episode_id]}", flush=True)
    atomic_json(output / "STATUS.json", {"source_run": str(run),
                                         "complete": sorted(r["episode_id"] for r in results),
                                         "failed": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
