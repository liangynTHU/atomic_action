"""Video-frame extraction used by CoT generation and visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


def _av_module():
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "video decoding requires PyAV; install requirements.txt"
        ) from exc
    return av


def extract_video_frames(
    video: Path | str,
    frame_indices: Iterable[int],
) -> dict[int, Image.Image]:
    """Decode selected zero-based frames from one video."""

    wanted = sorted({int(frame) for frame in frame_indices})
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError("frame indices must be non-negative")
    av = _av_module()
    found: dict[int, Image.Image] = {}
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                found[index] = frame.to_image().convert("RGB")
            if index > wanted[-1] or len(found) == len(wanted):
                break
    missing = sorted(set(wanted) - found.keys())
    if missing:
        raise ValueError(f"{video}: missing decoded frames {missing}")
    return found


def materialize_camera_frames(
    videos: Mapping[str, Path],
    requested: Mapping[str, Iterable[int]],
    output_dir: Path | str,
    *,
    quality: int = 90,
) -> dict[tuple[str, int], Path]:
    """Extract and save all requested camera/frame pairs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output: dict[tuple[str, int], Path] = {}
    for camera, indices in requested.items():
        frames = extract_video_frames(videos[camera], indices)
        for frame_index, image in frames.items():
            path = destination / f"{camera}_f{frame_index:06d}.jpg"
            image.save(path, format="JPEG", quality=quality)
            output[(camera, frame_index)] = path
    return output
