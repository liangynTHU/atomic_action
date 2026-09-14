import tempfile
import unittest
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image

from atomic_seg import segment_episode
from atomic_seg.io import CAMERA_KEYS, SuccessEpisode
from atomic_seg.visualization import (
    package_visualizations,
    write_episode_html,
    write_segmentation_svg,
    write_visualization_index,
)


class ReferenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("src", "href"):
            if key in values:
                self.references.append(values[key])


def synthetic_trajectory(frame_count=40):
    trajectory = np.zeros((frame_count, 16), dtype=np.float64)
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = 1.0
    trajectory[:, 11] = 1.0
    trajectory[:, 15] = 1.0
    trajectory[5:25, 0] = np.linspace(0.0, 0.2, 20)
    trajectory[25:, 0] = 0.2
    return trajectory


class VisualizationTests(unittest.TestCase):
    def _build_bundle(self, root: Path):
        segmentation = segment_episode(
            synthetic_trajectory(),
            np.arange(40) * 0.02,
        )
        entries = []
        for episode_index in range(3):
            task = f"Synthetic successful task {episode_index}"
            episode = SuccessEpisode(
                episode_index=episode_index,
                task=task,
                success=True,
                success_provenance="synthetic test fixture",
            )
            svg_path = (
                root
                / "segmentation"
                / f"episode_{episode_index:06d}.svg"
            )
            write_segmentation_svg(
                svg_path,
                episode=episode,
                result=segmentation,
            )

            image_paths = []
            image_roles = []
            for camera_index, camera in enumerate(CAMERA_KEYS):
                relative = (
                    Path("assets")
                    / f"episode_{episode_index:06d}"
                    / f"{camera}.jpg"
                )
                absolute = root / relative
                absolute.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (32, 24),
                    color=(40 * episode_index, 60 * camera_index, 100),
                ).save(absolute)
                image_paths.append(relative.as_posix())
                image_roles.append(f"current_{camera}")

            source_videos = {}
            for camera in CAMERA_KEYS:
                video = (
                    root.parent
                    / "synthetic_sources"
                    / f"episode_{episode_index:06d}_{camera}.mp4"
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"synthetic-video-placeholder")
                source_videos[camera] = video

            action_episode = {
                "episode_index": episode_index,
                "task": task,
                "frame_count": 40,
                "segments": [
                    {
                        "segment_index": 0,
                        "start_frame": 0,
                        "end_frame": 39,
                        "primary_action_verb": (
                            "left:move, right:still"
                        ),
                        "sub_task": (
                            "Move the left arm while keeping the right arm "
                            "still."
                        ),
                        "coordination_relation": "left_only",
                        "num_chunks": 5,
                        "guide_action": [0.0] * 16,
                    }
                ],
            }
            cot_rows = [
                {
                    "segment_index": 0,
                    "images": image_paths,
                    "image_roles": image_roles,
                    "conversations": [
                        {"from": "human", "value": "synthetic"},
                        {
                            "from": "gpt",
                            "value": (
                                "<think>Synthetic pre-action reasoning."
                                "</think>\n{}"
                            ),
                        },
                    ],
                }
            ]
            html_path = (
                root
                / "episodes"
                / f"episode_{episode_index:06d}.html"
            )
            write_episode_html(
                html_path,
                output_root=root,
                action_episode=action_episode,
                cot_rows=cot_rows,
                videos=source_videos,
            )
            entries.append(
                {
                    "episode_index": episode_index,
                    "task": task,
                    "segmentation_svg": svg_path.relative_to(
                        root
                    ).as_posix(),
                    "episode_html": html_path.relative_to(root).as_posix(),
                }
            )
        write_visualization_index(root / "index.html", entries=entries)
        return entries

    def test_synthetic_bundle_has_three_of_each_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visualizations"
            entries = self._build_bundle(root)
            self.assertEqual(len(entries), 3)
            self.assertEqual(
                len(list((root / "segmentation").glob("*.svg"))),
                3,
            )
            self.assertEqual(
                len(list((root / "episodes").glob("episode_*.html"))),
                3,
            )

    def test_all_generated_html_assets_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visualizations"
            self._build_bundle(root)
            for page in root.rglob("*.html"):
                parser = ReferenceParser()
                parser.feed(page.read_text())
                for reference in parser.references:
                    if reference.startswith(("#", "http:", "https:")):
                        continue
                    self.assertTrue(
                        (page.parent / reference).exists(),
                        f"{page}: missing {reference}",
                    )

    def test_complete_episode_pages_include_three_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visualizations"
            self._build_bundle(root)
            for page in sorted(
                (root / "episodes").glob("episode_*.html")
            ):
                text = page.read_text()
                self.assertEqual(text.count("<video "), 3)
                self.assertIn("Action-only supervision", text)
                self.assertIn("Embedded pre-action CoT", text)

    def test_segmentation_svgs_show_both_arm_lanes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visualizations"
            self._build_bundle(root)
            for path in (root / "segmentation").glob("*.svg"):
                text = path.read_text()
                self.assertIn(">left</text>", text)
                self.assertIn(">right</text>", text)
                self.assertIn("Atomic segment table", text)

    def test_portable_zip_contains_all_six_visualizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "visualizations"
            self._build_bundle(root)
            zip_path = Path(directory) / "visualizations.zip"
            report = package_visualizations(root, zip_path)
            self.assertEqual(report["segmentation_svg_count"], 3)
            self.assertEqual(report["complete_episode_html_count"], 3)
            self.assertEqual(report["complete_camera_video_count"], 9)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
