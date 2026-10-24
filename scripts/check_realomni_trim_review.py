#!/usr/bin/env python3
"""Verify segment navigation, exact targets, full media and offline downloads."""

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
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"], timeout=20000)
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri(), wait_until="load", timeout=60000)
        page.evaluate("document.fonts.ready")
        for e in data["examples"]:
            page.locator(f'[data-episode="{e["id"]}"]').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
            assert page.locator(".card").count() == len(e["segments"])
            assert page.locator(".timeline button").count() == len(e["segments"])
            assert float(page.locator("#scrubber").get_attribute("min")) == e["retained_start_s"]
            assert float(page.locator("#scrubber").get_attribute("max")) == e["retained_end_s"]
            page.evaluate("document.querySelectorAll('img').forEach(i=>i.loading='eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('img')).every(i=>i.complete&&i.naturalWidth>0)", timeout=60000)
            targets = page.locator(".card details:first-of-type pre").evaluate_all("es=>es.map(e=>JSON.parse(e.textContent))")
            for i, s in enumerate(e["segments"]):
                assert targets[i]["guide_action"] == s["guide_action"]
                assert targets[i]["target_state"] == s["target_state"]
                assert len(targets[i]["guide_action"]) == 16
                button = page.locator(f'.actions [data-segment="{i}"]')
                button.click()
                assert abs(float(page.locator("#scrubber").input_value()) - s["start_time_seconds"]) < .0011
                assert "active" in (page.locator(f'.timeline [data-active="{i}"]').get_attribute("class") or "")
            for i in (0, len(e["segments"]) // 2, len(e["segments"]) - 1):
                page.locator(f'.actions [data-segment="{i}"]').click()
                page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2&&!v.seeking)")
                for j, camera in enumerate(e["cameras"]):
                    actual = page.locator(f"#camera-{j}").evaluate("v=>v.currentTime")
                    assert abs(actual - (e["segments"][i]["start_time_seconds"] - camera["offset_s"])) < .03
            page.locator("#full-source").check()
            assert float(page.locator("#scrubber").get_attribute("min")) == 0
            page.locator("#scrubber").evaluate("e=>{e.value=0;e.dispatchEvent(new Event('input'))}")
            assert "已删除" in page.locator("#position-note").inner_text()
            page.locator("#full-source").uncheck()
            page.locator("#restart").click()
            page.locator("#play").click()
            page.wait_for_function("(t)=>Number(document.querySelector('#scrubber').value)>t+.6", arg=e["retained_start_s"])
            page.locator("#play").click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2&&!v.seeking)")
            checks.append({"id": e["id"], "segments": len(e["segments"]), "all_segment_seeks": True,
                           "guide_and_target_exact": True, "all_images_loaded": True, "crop_controls": True,
                           "both_videos_playable": True})
            print("browser PASS", e["id"], len(e["segments"]), "segments", flush=True)
        page.locator('[data-episode="example_01"]').click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
        page.locator("#play").click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.currentTime>1)")
        page.locator("#play").click()
        page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2&&!v.seeking)")
        page.evaluate("document.fonts.ready")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(root / "preview_desktop.png"))
        page.locator("#action-0").screenshot(path=str(root / "preview_segment.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(root / "preview_mobile.png"))
        with page.expect_download() as download:
            page.get_by_role("link", name="全部状态与参考动作", exact=True).click()
        destination = root / "download_check.json"
        download.value.save_as(str(destination))
        assert destination.read_bytes() == (root / "example_01/reference_actions.json").read_bytes()
        destination.unlink()
        browser.close()
    assert not errors, errors
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps({"passed": True, "examples": checks,
            "mobile_no_overflow": True, "downloads_preserve_nanoseconds": True, "javascript_errors": errors}, indent=2) + "\n")
    print("ALL BROWSER CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
