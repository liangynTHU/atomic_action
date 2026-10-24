#!/usr/bin/env python3
"""Browser validation for the offline AgiBot review."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "examples.json").read_text())
    errors, results = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((root / "index.html").as_uri(), wait_until="load")
        assert page.locator(".ep").count() == len(data["examples"])
        for i, example in enumerate(data["examples"]):
            page.locator(f'[data-episode="{i}"]').click()
            if not example["segments"]:
                assert "未进入分割" in page.locator("main").inner_text()
                assert page.locator("video").count() == 0
                results.append({"episode": example["episode_id"], "rejection_visible": True})
                continue
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=2)")
            assert page.locator(".seg").count() == len(example["segments"])
            page.locator('[data-row="2"]').click()
            expected = example["segments"][2]["start_s"]
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>!v.seeking)")
            assert all(abs(t-expected)<.04 for t in page.locator("video").evaluate_all("vs=>vs.map(v=>v.currentTime)"))
            page.wait_for_function("Array.from(document.querySelectorAll('img')).every(i=>i.complete&&i.naturalWidth>0)")
            page.locator("#play").click()
            page.wait_for_function("Number(document.querySelector('#time').value)>"+str(expected+.3))
            page.locator("#play").click()
            page.locator('[data-row="0"]').click()
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=str(root / "preview.png"), full_page=False)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
            page.screenshot(path=str(root / "preview_mobile.png"))
            page.set_viewport_size({"width": 1600, "height": 1100})
            results.append({"episode": example["episode_id"], "videos_playable": True,
                            "segment_seek": True, "images_loaded": True, "mobile_no_overflow": True})
        browser.close()
    assert not errors, errors
    report = {"passed": True, "errors": errors, "examples": results, "offline_file_open": True}
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
