"""Validate every declared camera frame; persist only requested observations."""

from pathlib import Path
import math
import av
from PIL import Image, ImageDraw, ImageFont, ImageOps


class InvalidVideo(ValueError):
    pass


def read_video(video, n, fps, tolerance, *, selected=(), destination=None, preview=None, heartbeat=None):
    """LeRobot v3 video offsets are global within a shared MP4 file.

    Exact frame identity comes from PTS, not seek return order. All N frames in
    the episode must decode once. A terminal target image is a real observation.
    """
    if "path" not in video:
        raise InvalidVideo("missing_video_metadata")
    path = Path(video["path"])
    if not path.is_file():
        raise InvalidVideo("missing_video_file")
    start, stop = float(video["from_s"]), float(video["to_s"])
    if start < 0 or abs((stop-start) - n/fps) > tolerance:
        raise InvalidVideo("video_episode_duration_mismatch")
    selected = set(int(x) for x in selected)
    if any(x < 0 or x >= n for x in selected):
        raise ValueError("requested observation outside episode")
    if destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
    writer = None
    stream_out = None
    paths = {}
    expected = 0
    max_skew = 0.
    try:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise InvalidVideo("unexpected_video_stream_count")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            if stream.average_rate and abs(float(stream.average_rate)-fps) > .01:
                raise InvalidVideo("video_fps_mismatch")
            if start:
                container.seek(max(0, math.floor(start/float(stream.time_base))), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None or frame.is_corrupt:
                    raise InvalidVideo("missing_pts_or_corrupt_video_frame")
                timestamp = float(frame.pts * frame.time_base)
                relative = (timestamp-start)*fps
                index = round(relative)
                if timestamp < start-tolerance:
                    continue
                if index >= n:
                    break
                if index != expected or abs(timestamp-(start+index/fps)) > tolerance:
                    raise InvalidVideo(f"video_gap_duplicate_or_misalignment: expected={expected}, actual={index}")
                max_skew = max(max_skew, abs(timestamp-(start+index/fps)))
                if index in selected or preview:
                    im = frame.to_image()
                    if index in selected:
                        if destination is None:
                            raise ValueError("selected frames need a destination")
                        im_small = im.copy()
                        im_small.thumbnail((640, 480))
                        target = Path(destination) / f"{index:07d}.jpg"
                        im_small.save(target, quality=88)
                        paths[index] = str(target)
                    if preview:
                        im.thumbnail((640, 480))
                        if writer is None:
                            writer = av.open(str(preview), "w")
                            stream_out = writer.add_stream("libx264", rate=fps)
                            stream_out.width = im.width//2*2
                            stream_out.height = im.height//2*2
                            stream_out.pix_fmt = "yuv420p"
                            stream_out.options = {"crf": "25", "preset": "veryfast"}
                        out = av.VideoFrame.from_image(im.resize((stream_out.width, stream_out.height)))
                        out.pts = index
                        for packet in stream_out.encode(out):
                            writer.mux(packet)
                expected += 1
                if heartbeat and expected % 300 == 0:
                    heartbeat(expected)
            if expected != n or set(paths) != selected:
                raise InvalidVideo(f"truncated_video_episode: expected={n}, decoded={expected}")
            if writer:
                for packet in stream_out.encode():
                    writer.mux(packet)
    finally:
        if writer:
            writer.close()
    return {"status": "passed", "frames": expected, "max_pts_skew_s": max_skew,
            "episode_from_s": start, "episode_to_s": stop, "source": str(path), "paths": paths}


def sample_frames(segments):
    import numpy as np
    return sorted({int(i) for segment in segments
                   for i in np.linspace(segment["start_frame"], segment["end_frame"]+1, 5, dtype=int)})


def montage(paths, labels, target):
    result = Image.new("RGB", (1920, 520), "#162130")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=22)
    names = {"head": "HEAD CAMERA", "hand_left": "PHYSICAL LEFT WRIST CAMERA",
             "hand_right": "PHYSICAL RIGHT WRIST CAMERA"}
    for i, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as im:
            tile = ImageOps.contain(im.convert("RGB"), (640, 480))
            result.paste(tile, (i*640+(640-tile.width)//2, 40+(480-tile.height)//2))
        draw.text((i*640+8, 8), names.get(label, label), fill="white", font=font)
    result.save(target, quality=88)


def temporal_camera_sheet(media_dir, camera, frames, target):
    """Keep a single camera's own gripper together across chronological frames."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height, header = 640, 480, 36
    columns = min(3, len(frames))
    result = Image.new("RGB", (width * columns, (height + header) * math.ceil(len(frames) / columns)), "#162130")
    draw, font = ImageDraw.Draw(result), ImageFont.load_default(size=17)
    name = {"head": "HEAD", "hand_left": "PHYSICAL LEFT WRIST",
            "hand_right": "PHYSICAL RIGHT WRIST"}[camera]
    for i, frame in enumerate(frames):
        x, y = (i % columns) * width, (i // columns) * (height + header)
        with Image.open(Path(media_dir) / camera / f"{frame:07d}.jpg") as image:
            tile = ImageOps.contain(image.convert("RGB"), (width, height))
            result.paste(tile, (x + (width - tile.width) // 2,
                                y + header + (height - tile.height) // 2))
        draw.text((x + 6, y + 8), f"{name} | f{frame}", fill="white", font=font)
    result.save(target, quality=95)
    return target
