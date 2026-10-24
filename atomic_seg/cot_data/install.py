"""Install an audited release with checksum verification and rollback copies."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from atomic_seg.common.artifacts import atomic_write_json, sha256_file, utc_now
from atomic_seg.common.locking import writer_lock
from .evidence import QUALITY_POLICY

REQUIRED = (
    "data/vlm/train.jsonl",
    "data/vla/train.json",
    "data/action_segmentation/train.jsonl",
    "metadata/source_manifest.jsonl",
    "metadata/ACTION_LABEL_CORRECTIONS.jsonl",
    "metadata/MANIFEST_SUMMARY.json",
    "FINAL_REPORT.json",
    "FINAL_AUDIT.json",
    "COT_STYLE_REPORT.json",
    "COT_QUALITY_AUDIT.json",
    "README.md",
    "work/cot5_annotations.jsonl",
    "work/REQUEST_JOURNALS.json",
)
OPTIONAL = (
    "ACTION_MANIFEST_REPORT.json",
    "SEMANTIC_CLEANUP_REPORT.json",
    "SHORT_SEGMENT_MERGE_REPORT.json",
    "metadata/SHORT_SEGMENT_MAPPING.jsonl",
)


def install_release(
    release: Path, dataset: Path, *, replace: bool = False, dry_run: bool = False
):
    release, dataset = release.resolve(), dataset.resolve()
    if release == dataset or release in dataset.parents or dataset in release.parents:
        raise ValueError("release and dataset must be separate directories")
    files = [*REQUIRED, *(name for name in OPTIONAL if (release / name).is_file())]
    for name in files:
        if not (release / name).is_file():
            raise FileNotFoundError(release / name)
    if not (release / "control/COMPLETE").is_file():
        raise ValueError("release is not marked complete")
    audit = json.loads((release / "FINAL_AUDIT.json").read_text())
    quality = json.loads((release / "COT_QUALITY_AUDIT.json").read_text())
    if audit.get("passed") is not True or quality.get("passed") is not True:
        raise ValueError("Both release audits must pass")
    if audit.get("quality_policy") != QUALITY_POLICY or quality.get("quality_policy") != QUALITY_POLICY or audit.get("quality", {}).get("observer_and_two_critics_verified") is not True:
        raise ValueError("Release lacks the current observer and dual critic contract")
    journals = json.loads((release / "work/REQUEST_JOURNALS.json").read_text())["files"]
    for name in journals:
        path = Path("work") / name
        if Path(name).is_absolute() or ".." in path.parts:
            raise ValueError("Request journal path must stay inside release work")
        files.append(str(path))
    for name in (
        "data/vlm/train.jsonl",
        "data/vla/train.json",
        "data/action_segmentation/train.jsonl",
        "metadata/source_manifest.jsonl",
        "metadata/ACTION_LABEL_CORRECTIONS.jsonl",
        "COT_QUALITY_AUDIT.json",
        "COT_STYLE_REPORT.json",
        "work/cot5_annotations.jsonl",
        "work/REQUEST_JOURNALS.json",
        *(str(Path("work") / name) for name in journals),
    ):
        if audit["hashes"].get(name) != sha256_file(release / name):
            raise ValueError(f"Audited file changed: {name}")
    if quality["hashes"]["vlm"] != audit["hashes"]["data/vlm/train.jsonl"]:
        raise ValueError("Quality audit is not for the released VLM")
    expected_hashes = {name: sha256_file(release / name) for name in files}
    conflicts = [
        name
        for name in files
        if (dataset / name).exists() or (dataset / name).is_symlink()
    ]
    if conflicts and not replace:
        raise FileExistsError(
            "Existing output; --replace retains rollback copies before replacement"
        )
    plan = {
        "installed_release": str(release),
        "dataset_root": str(dataset),
        "files": files,
        "replace": conflicts,
        "dry_run": dry_run,
    }
    if dry_run:
        return plan
    dataset.mkdir(parents=True, exist_ok=True)
    with writer_lock(dataset.parent / (dataset.name + ".install.lock")):
        conflicts = [
            name
            for name in files
            if (dataset / name).exists() or (dataset / name).is_symlink()
        ]
        if conflicts and not replace:
            raise FileExistsError("Destination changed before acquiring install lock")
        staging = Path(tempfile.mkdtemp(prefix=".cot-install-", dir=dataset.parent))
        backup = (
            Path(
                tempfile.mkdtemp(prefix=dataset.name + ".rollback-", dir=dataset.parent)
            )
            if conflicts
            else None
        )
        try:
            for name in files:
                destination = dataset / name
                if destination.is_symlink() or any(
                    p.is_symlink() for p in destination.parents if p != dataset.parent
                ):
                    raise ValueError(
                        f"Refusing to install through symlink: {destination}"
                    )
                staged = staging / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(release / name, staged)
                if sha256_file(staged) != expected_hashes[name]:
                    raise ValueError(f"Staging copy differs: {name}")
                if destination.exists() and backup is not None:
                    (backup / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, backup / name)
            # Per-file atomic replacement; readers must pause during installation.
            for name in files:
                (dataset / name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / name, dataset / name)
            plan.update(
                {
                    "installed_at": utc_now(),
                    "rollback_root": str(backup) if backup else None,
                    "hashes": {name: sha256_file(dataset / name) for name in files},
                }
            )
            atomic_write_json(dataset / "INSTALL_REPORT.json", plan)
        finally:
            shutil.rmtree(staging)
    return plan
