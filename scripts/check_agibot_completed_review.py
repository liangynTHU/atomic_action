#!/usr/bin/env python3
"""Validate the portable review against the original committed AgiBot exports."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "examples.json").read_text())
    errors, results = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"], timeout=20000)
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        assert page.locator(".ep").count() == len(data["examples"])
        for index, example in enumerate(data["examples"]):
            page.locator(f'[data-episode="{index}"]').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
            durations = page.locator("video").evaluate_all("vs=>vs.map(v=>v.duration)")
            assert len(durations) == 3
            assert all(abs(duration - example["duration_s"]) < .04 for duration in durations)
            assert page.locator(".seg").count() == 2 * len(example["segments"])
            assert page.locator("[data-row]").count() == len(example["segments"])
            source = Path(example["source_export"])
            vlm = [json.loads(line) for line in (source / "vlm.jsonl").read_text().splitlines()]
            for i, segment in enumerate(example["segments"]):
                page.locator(f'[data-row="{i}"]').click()
                page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
                times = page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)")
                assert all(abs(t - segment["start_s"]) < .04 for t in times)
                assert page.locator("#subtask").inner_text() == segment["sub_task"]
                messages = page.locator("#training-messages pre").all_text_contents()
                assert messages == [m["content"] for m in vlm[i]["messages"]]
                assert page.locator(".seg.active").count() == 2
                page.locator("#student-observations > summary").click()
                page.locator("#student-observations img").evaluate_all("imgs=>imgs.forEach(i=>i.loading='eager')")
                page.wait_for_function("Array.from(document.querySelectorAll('#segment-detail img')).every(i=>i.complete&&i.naturalWidth>0)")
            for href in page.locator("a[download]").evaluate_all("a=>a.map(x=>x.getAttribute('href'))"):
                assert (root / href).is_file(), href
            page.locator('[data-segment="0"]').first.click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            page.locator("#next-frame").click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            assert all(abs(t - 1/data["fps"]) < .002 for t in page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)"))
            page.locator("#previous-frame").click()
            page.locator("#speed").select_option("1.5")
            assert page.locator("video").evaluate_all("vs=>vs.every(v=>v.playbackRate===1.5)")
            page.locator("#speed").select_option("1")
            boundary = example["segments"][0]["end_s"]
            page.locator("#time").evaluate("(el,t)=>{el.value=t;el.dispatchEvent(new Event('input'))}", max(0, boundary-.2))
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            page.locator("#play").click()
            page.wait_for_function("t=>document.querySelector('video').currentTime>t", arg=boundary+.25)
            page.locator("#play").click()
            assert page.locator('.seg.active').first.get_attribute("data-segment") == "1"
            times = page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)")
            assert max(times)-min(times)<.18, times
            page.locator('[data-segment="2"]').first.click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            page.locator(".shots [data-image]").first.click()
            assert page.locator("#lightbox").evaluate("d=>d.open")
            page.wait_for_function("document.querySelector('#large-image').complete&&document.querySelector('#large-image').naturalWidth>0")
            page.locator("#close-image").click()
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=str(root / f"preview_{index+1}.png"))
            if index == 0:
                page.locator("#segment-detail").screenshot(path=str(root / "preview_segment.png"))
                page.evaluate("window.scrollTo(0,0)")
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("window.scrollTo(0,0)")
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth"), example["episode_id"]
            if index == 0:
                page.screenshot(path=str(root / "preview_mobile.png"))
            page.set_viewport_size({"width": 1600, "height": 1100})
            result = {"episode_id": example["episode_id"], "segments": len(example["segments"]),
                      "all_segment_seeks": True, "exact_original_training_messages": True,
                      "all_observation_images_loaded": True, "three_videos_playable": True,
                      "durations_verified": True, "auto_follow_segment": True,
                      "frame_step": True, "speed_control": True, "image_lightbox": True,
                      "downloads_exist": True, "mobile_no_overflow": True}
            results.append(result)
            print("PASS", example["episode_id"], flush=True)
        with page.expect_download() as download_info:
            page.locator('a[download][href$="vla.json"]').click()
        download_info.value.save_as(root / "download_validation.tmp.json")
        assert json.loads((root / "download_validation.tmp.json").read_text()) == json.loads((Path(data["examples"][-1]["source_export"]) / "vla.json").read_text())
        (root / "download_validation.tmp.json").unlink()
        browser.close()
    assert not errors, errors
    report = {"passed": True, "offline_file_open": True, "errors": errors, "examples": results}
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
