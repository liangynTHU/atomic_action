#!/usr/bin/env python3
"""Exercise a generated offline non-CoT index with a local Playwright browser."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = json.loads((root / "complete_episodes.json").read_text())
    errors, results = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"], timeout=20000)
        page = browser.new_page(viewport={"width": 1600, "height": 1050}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((root / "index.html").as_uri(), wait_until="load", timeout=60000)
        page.evaluate("document.fonts.ready")
        for episode in data["episodes"]:
            page.locator(f'[data-episode="{episode["id"]}"]').click()
            page.wait_for_function('document.querySelector("#episode-video").readyState>=2', timeout=60000)
            assert page.locator(".action-card").count() == len(episode["segments"])
            if data.get("student_format"):
                assert page.locator("#student-format").count() == 1
                for row in episode["vlm"]:
                    assert set(row) == {"messages", "images"}
                    assert "\n" not in row["messages"][0]["content"]
                    assert "<think>" not in row["messages"][0]["content"]
                assert len(episode["vlm_metadata"]) == len(episode["vlm"])
            if data.get("length_comparison"):
                assert page.locator("#length-comparison").count() == 1
                if episode.get("length_audit"):
                    policy = episode["length_audit"]["policy"]
                    if policy.get("minimum_frames") is not None:
                        assert all(s["valid_action_rows"] >= policy["minimum_frames"] for s in episode["segments"])
                    else:
                        assert "不设16帧硬下限" in page.locator("#length-comparison").inner_text()
                    assert page.locator(".short-flag").count() == sum(bool(s.get("short_segment_reason")) for s in episode["segments"])
                if episode.get("both_still_cleanup"):
                    assert all(s["left_action"] != "still" or s["right_action"] != "still" for s in episode["segments"][:-1])
                    assert page.locator("#both-still-policy").count() == 1
            page.locator("#open-messages").click()
            rendered = page.evaluate("""() => Array.from(document.querySelectorAll('.action-card')).map(c=>({
                messages:Array.from(c.querySelectorAll('details.message pre')).map(p=>p.textContent),
                vla:JSON.parse(c.querySelector('details.vla-segment pre').textContent)
            }))""")
            for i, value in enumerate(rendered):
                assert value["messages"] == [m["content"] for m in episode["vlm"][i]["messages"]]
                assert value["vla"] == episode["segments"][i]
                assert "<think>" not in value["messages"][-1]
            page.evaluate("document.querySelectorAll('img').forEach(i=>i.loading='eager')")
            page.wait_for_function('Array.from(document.querySelectorAll("img:not(#large-image)")).every(i=>i.complete&&i.naturalWidth>0)', timeout=60000)
            for i, segment in enumerate(episode["segments"]):
                page.locator(f'.chain [data-seek="{i}"]').click()
                page.wait_for_function('(n)=>document.querySelector("#frame-label").textContent.startsWith(`row ${n} /`)', arg=segment["start_frame"])
                assert "active" in (page.locator(f'.timeline [data-active="{i}"]').get_attribute("class") or "")
                time = page.locator("#episode-video").evaluate("v=>v.currentTime")
                assert abs(time - episode["timestamps"][segment["start_frame"]]) < .002
            page.locator("#episode-video").evaluate("v=>{v.currentTime=0;return v.play()}")
            page.wait_for_function('document.querySelector("#episode-video").currentTime>.15')
            page.locator("#episode-video").evaluate("v=>v.pause()")
            for link in page.locator("a[download]").all():
                assert (root / link.get_attribute("href")).is_file()
            results.append({"episode_id": episode["id"], "segments": len(episode["segments"]),
                            "exact_messages": True, "exact_vla": True, "all_images_loaded": True,
                            "all_segment_seeks": True, "video_playback": True})
            print("browser PASS", episode["id"], flush=True)
        for dataset in ["OXE", "ABC-130k"]:
            page.locator(f'[data-filter="{dataset}"]').click()
            assert page.locator("[data-episode]").count() == 4
        page.locator('[data-filter="all"]').click()
        page.locator('[data-episode="oxe_bridge_02"]').click()
        page.wait_for_function('document.querySelector("#episode-video").readyState>=2')
        page.evaluate("document.fonts.ready")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(root / "preview_desktop.png"))
        page.locator("#action-0").screenshot(path=str(root / "preview_action.png"))
        if data.get("student_format"):
            page.locator("#student-format").screenshot(path=str(root / "preview_student_format.png"))
        page.locator('[data-episode="abc_whiteboard"]').click()
        page.wait_for_function('document.querySelector("#episode-video").readyState>=2')
        page.evaluate("document.fonts.ready")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(root / "preview_abc.png"))
        if data.get("length_comparison"):
            page.locator('[data-episode="abc_towel"]').click()
            page.wait_for_function('document.querySelector("#episode-video").readyState>=2')
            page.evaluate("document.fonts.ready")
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(path=str(root / "preview_abc_towel.png"))
            page.locator("#length-comparison").screenshot(path=str(root / "preview_length_comparison.png"))
            page.locator('[data-episode="abc_whiteboard"]').click()
            page.wait_for_function('document.querySelector("#episode-video").readyState>=2')
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "mobile overflow"
        page.screenshot(path=str(root / "preview_mobile.png"))
        with page.expect_download() as downloading:
            page.get_by_role("link", name="完整 VLM JSONL", exact=True).click()
        path = root / "browser_download_check.jsonl"
        downloading.value.save_as(str(path))
        downloaded = [json.loads(line) for line in path.read_text().splitlines()]
        download_episode = next(e for e in data["episodes"] if e["id"] == "abc_whiteboard")
        expected = download_episode["vlm"]
        assert downloaded == expected
        if data.get("student_format"):
            assert path.read_text() == download_episode["vlm_jsonl_text"]
        path.unlink()
        browser.close()
    assert not errors, errors
    report = {"passed": True, "offline_file_open": True, "javascript_errors": errors, "episodes": results,
              "filters": True, "mobile_no_overflow": True, "vlm_download_roundtrip": True,
              "length_comparison_and_short_flags": bool(data.get("length_comparison"))}
    report["robotwin_aligned_training_format"] = bool(data.get("student_format"))
    (root / "BROWSER_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ALL BROWSER CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
