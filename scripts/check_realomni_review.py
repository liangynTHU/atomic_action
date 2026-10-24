#!/usr/bin/env python3
"""Check the generated RealOmni review in a local Chromium browser."""

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
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"], timeout=20000)
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri(), wait_until="load", timeout=60000)
        page.evaluate("document.fonts.ready")
        assert page.locator("[data-episode]").count() == data["preview_episodes"]
        for example in data["examples"]:
            page.locator(f'[data-episode="{example["id"]}"]').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
            assert page.locator("video").count() == len(example["cameras"])
            assert page.locator("tbody tr").count() == len(example["streams"])
            if example["status"] == "rejected":
                assert "未进入分割" in page.locator("#main").inner_text()
                assert example["segmentation"] is None
            if example.get("media_errors"):
                assert "无法完整解码" in page.locator("#main").inner_text()
                assert page.locator("#play").is_disabled()
                results.append({"id": example["id"], "video_decode_failure_visible": True, "partial_video_not_presented": True})
                print("browser PASS", example["id"], "explicit media failure", flush=True)
                continue
            page.evaluate("document.querySelectorAll('img').forEach(im=>im.loading='eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('img')).every(im=>im.complete&&im.naturalWidth>0)")
            for i, camera in enumerate(example["cameras"]):
                duration = page.locator(f"#camera-{i}").evaluate("v=>v.duration")
                expected = (camera["last_timestamp_ns"] - camera["first_timestamp_ns"]) / 1e9 + 1 / 30
                assert abs(duration - expected) < .08, (camera["name"], duration, expected)
            # Closely spaced markers can overlap at episode scale; the evidence
            # cards offer individually clickable targets for the same timestamps.
            gap = page.locator(".evidence [data-seek]").first
            seek = float(gap.get_attribute("data-seek"))
            gap.click(force=True)
            assert abs(float(page.locator("#time").input_value()) - seek) < .002
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            for i, camera in enumerate(example["cameras"]):
                actual = page.locator(f"#camera-{i}").evaluate("v=>v.currentTime")
                expected = max(0, min(camera["end_s"] - camera["offset_s"], seek - camera["offset_s"]))
                assert abs(actual - expected) < .02, (actual, expected)
            page.locator("#restart").click()
            page.locator("#play").click()
            page.wait_for_function("Number(document.querySelector('#time').value)>.6")
            page.locator("#play").click()
            assert all(t > .2 for t in page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)"))
            page.locator("details").first.locator("summary").click()
            raw = json.loads(page.locator("details").first.locator("pre").inner_text())
            assert raw == example["reasons"]
            for link in page.locator("a[download]").all():
                assert (root / link.get_attribute("href")).is_file()
            results.append({"id": example["id"], "videos_playable": True, "all_images_loaded": True,
                            "timestamp_seeking": True, "exact_rejection_reasons": True})
            print("browser PASS", example["id"], flush=True)
        page.locator('[data-episode="example_02"]').click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
        page.locator("#play").click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.currentTime>1)")
        page.locator("#play").click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2&&!v.seeking)")
        page.evaluate("document.fonts.ready")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(root / "preview_desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        page.screenshot(path=str(root / "preview_mobile.png"))
        with page.expect_download() as download:
            page.get_by_role("link", name="原始判定 JSON", exact=True).click()
        destination = root / "download_check.json"
        download.value.save_as(str(destination))
        original = json.loads((root / "example_02/decision.json").read_text())
        assert json.loads(destination.read_text()) == original
        destination.unlink()
        browser.close()
    assert not errors, errors
    report = {"passed": True, "offline_file_open": True, "javascript_errors": errors, "examples": results,
              "download_roundtrip": True, "mobile_no_overflow": True}
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ALL BROWSER CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
