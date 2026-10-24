#!/usr/bin/env python3
"""Check offline review playback, exact training content and responsive layout."""

import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "complete_episodes.json").read_text())
    errors, results, media = [], [], []
    with sync_playwright() as playwright:
        options = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage",
                                               "--allow-file-access-from-files"], "timeout": 30000}
        if args.browser:
            options["executable_path"] = str(args.browser)
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page.set_default_timeout(30000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        assert page.locator("[data-episode]").count() == 12
        for episode in data["episodes"]:
            page.locator(f'[data-episode="{episode["id"]}"]').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
            assert page.locator(".action-card").count() == len(episode["segments"])
            exact = page.evaluate("""() => Array.from(document.querySelectorAll('.action-card')).map(card=>({
                messages:Array.from(card.querySelectorAll('details.message pre')).map(p=>p.textContent),
                training:JSON.parse(card.querySelector('.full-json pre').textContent),
                vla:JSON.parse(card.querySelector('.vla-segment pre').textContent)
            }))""")
            for index, value in enumerate(exact):
                assert value["messages"] == [message["content"] for message in episode["vlm"][index]["messages"]]
                assert value["training"] == episode["vlm"][index]
                assert value["vla"] == episode["segments"][index]
            page.evaluate("document.querySelectorAll('img.zoom').forEach(image=>image.loading='eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('img.zoom')).every(i=>i.complete&&i.naturalWidth>0)")
            for index, segment in enumerate(episode["segments"]):
                page.locator(f'.timeline [data-seek="{index}"]').click()
                page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
                expected = episode["display_timestamps"][segment["start_frame"]]
                times = page.locator("video").evaluate_all("videos=>videos.map(v=>v.currentTime)")
                assert all(abs(time - expected) < .003 for time in times), (episode["id"], index, times, expected)
                assert page.locator(f'.timeline [data-active="{index}"]').get_attribute("class") == "active"
            page.locator('.timeline [data-seek="0"]').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            page.locator("#next-frame").click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            assert all(abs(time - episode["display_timestamps"][1]) < .003
                       for time in page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)"))
            page.locator("#prev-frame").click()
            page.locator("#speed").select_option("1.5")
            assert page.locator("video").evaluate_all("vs=>vs.every(v=>v.playbackRate===1.5)")
            page.locator("#speed").select_option("1")
            page.locator("#play").click()
            page.wait_for_function("document.querySelector('video').currentTime>.4")
            page.locator("#play").click()
            times = page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)")
            assert max(times) - min(times) < .2, times
            for index, video in enumerate(episode["videos"]):
                locator = page.locator("video").nth(index)
                pixels = np.asarray(Image.open(io.BytesIO(locator.screenshot())).convert("RGB"))
                assert float(pixels.std()) > 10, (episode["id"], "blank video", index)
                with av.open(str(root / video["path"])) as container:
                    stream = container.streams.video[0]
                    frames = sum(1 for _ in container.decode(video=0))
                    assert frames == episode.get("observation_count", episode["frame_count"])
                media.append({"id": episode["id"], "path": video["path"], "decoded_frames": frames,
                              "browser_pixel_std": float(pixels.std())})
            page.locator("#action-0 img.zoom").first.click()
            assert page.locator("#image-dialog").evaluate("dialog=>dialog.open")
            page.wait_for_function("document.querySelector('#large-image').complete&&document.querySelector('#large-image').naturalWidth>0")
            page.locator("#close-image").click()
            for href in page.locator("a[download]").evaluate_all("links=>links.map(a=>a.getAttribute('href'))"):
                assert (root / href).is_file(), href
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=str(root / f"preview_{episode['id']}.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("window.scrollTo(0,0)")
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), episode["id"]
            page.screenshot(path=str(root / f"preview_mobile_{episode['id']}.png"))
            page.set_viewport_size({"width": 1600, "height": 1100})
            results.append({"id": episode["id"], "segments": len(episode["segments"]),
                            "exact_messages": True, "exact_training_json": True, "exact_action_segments": True,
                            "all_images_loaded": True, "every_segment_seek_verified": True,
                            "video_playback_and_pixels": True, "frame_step_and_speed": True,
                            "image_lightbox": True, "mobile_no_overflow": True, "downloads_exist": True})
            print("PASS", episode["id"], flush=True)
        for dataset in ("ABC-130k", "AgiBot", "RoboTwin"):
            page.locator(f'[data-filter="{dataset}"]').click()
            assert page.locator("[data-episode]").count() == 4
        page.locator('[data-filter="all"]').click()
        assert page.locator("[data-episode]").count() == 12
        with page.expect_download() as download:
            page.locator('a[download][href$="/vlm.jsonl"]').click()
        destination = root / "download_roundtrip.tmp.jsonl"
        download.value.save_as(destination)
        assert destination.read_bytes() == (root / "episodes/robotwin_01/vlm.jsonl").read_bytes()
        destination.unlink()
        browser.close()
    assert not errors, errors
    report = {"passed": True, "offline_file_open": True, "javascript_errors": errors,
              "dataset_filter_counts": True, "vlm_download_byte_identical": True, "episodes": results}
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "MEDIA_VALIDATION.json").write_text(json.dumps({"passed": True, "videos": media}, indent=2) + "\n")
    print("BROWSER PASS", len(results), "episodes", len(media), "videos", flush=True)


if __name__ == "__main__":
    main()
