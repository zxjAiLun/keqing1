from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


def _extract_paipu_id(url: str) -> Optional[str]:
    """
    Try to extract paipu UUID from common Majsoul/Mahjongsoul URLs.
    Examples:
      - ...?paipu=XXXX
      - /view_game/XXXX
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    paipu = qs.get("paipu", [None])[0]
    if paipu:
        return paipu

    # Fallback: path segment
    m = re.search(r"/view_game/([^/?#]+)", parsed.path)
    if m:
        return m.group(1)

    # Fallback: raw string query
    m = re.search(r"paipu=([^&#?/]+)", url)
    if m:
        return m.group(1)

    return None


def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("._")


def _load_links(links_file: Path) -> List[str]:
    links: List[str] = []
    for line in links_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        links.append(line)
    return links


def _wait_for_record_uuid(page: Any, timeout_ms: int) -> None:
    # Wait until the game viewer is ready for downloadlogs.js trigger.
    # downloadlogs.js checks `GameMgr.Inst.record_uuid`.
    try:
        page.wait_for_function(
            """
            () => {
                try {
                    return !!(window.GameMgr && window.GameMgr.Inst && window.GameMgr.Inst.record_uuid);
                } catch (e) {
                    return false;
                }
            }
            """,
            timeout=timeout_ms,
        )
    except Exception as e:
        # Best-effort diagnostics to help decide whether this site uses different globals.
        try:
            record_uuid = page.evaluate(
                """
                () => {
                    try {
                        return window.GameMgr && window.GameMgr.Inst ? window.GameMgr.Inst.record_uuid : null;
                    } catch (err) {
                        return null;
                    }
                }
                """
            )
        except Exception:
            record_uuid = None
        raise RuntimeError(f"Page.wait_for_function: Timeout {timeout_ms}ms exceeded; record_uuid={record_uuid!r}") from e


def _load_downloadlogs_js(*, url: str, local_path: Optional[str]) -> str:
    """
    Load downloadlogs.js source code.
    We inject it via `page.add_script_tag(content=...)` so the browser doesn't
    need to fetch the script remotely (avoids Playwright network/CSP issues).
    """
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")

    with urlopen(url, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _patch_downloadlogs_for_capture(js_text: str) -> str:
    """
    Patch injected downloadlogs.js so that when it would "download" data,
    we instead capture `{ filename, text }` into `window.__downloadlogs_last`.

    This avoids relying on Playwright's `page.expect_download`, which may not
    fire for data-URL downloads.
    """
    # Replace the whole `function download(filename, text) { ... }` block.
    # Keep it lenient to whitespace differences.
    pattern = r"function\s+download\s*\(\s*filename\s*,\s*text\s*\)\s*\{[\s\S]*?\n\s*return;\s*\n\s*\}"
    m = re.search(pattern, js_text)
    if not m:
        raise RuntimeError("failed to patch downloadlogs.js: download() function not found")

    replacement = (
        "function download(filename, text)\n"
        "{\n"
        "    try {\n"
        "        window.__downloadlogs_last = { filename: filename, text: text };\n"
        "    } catch (e) {}\n"
        "    return;\n"
        "}\n"
    )
    js_text = js_text[: m.start()] + replacement + js_text[m.end() :]

    # Expose downloadlog() so we can call it directly without simulating key presses.
    # Best-effort: inject right before the closing `})();` of the IIFE.
    if "__downloadlogs_downloadlog" not in js_text:
        marker_old = "})();\n// vim: ts=4  et"
        marker_new = "window.__downloadlogs_downloadlog = downloadlog;\n})();\n// vim: ts=4  et"
        if marker_old in js_text:
            js_text = js_text.replace(marker_old, marker_new, 1)
        else:
            # Fallback: inject near the last `})();`
            parts = js_text.rsplit("})();", 1)
            if len(parts) == 2:
                prefix, suffix = parts
                js_text = prefix + "window.__downloadlogs_downloadlog = downloadlog;\n})();" + suffix
            # else: keep original

    return js_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--links-file",
        default="dataset/links/majsoul_links.txt",
        help="Text file with one majsoul paipu link per line.",
    )
    parser.add_argument("--output-dir", default="dataset/tenhou6/ds2")
    parser.add_argument("--max-links", type=int, default=0, help="0 = all links")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--timeout-sec", type=int, default=90, help="record_uuid 等待超时（秒）")
    parser.add_argument("--download-timeout-sec", type=int, default=20, help="触发下载的等待超时（秒）")

    # Session options
    parser.add_argument("--storage-state", default=None, help="Playwright storageState.json path")
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help="Use a persistent Chromium profile directory (alternative to --storage-state).",
    )

    # Inject this downloader JS (Tampermonkey-like). Must be reachable from the browser.
    parser.add_argument(
        "--downloadlogs-url",
        default="https://gist.githubusercontent.com/Equim-chan/875a232a2c1d31181df8b3a8704c3112/raw/a0533ae7a0ab0158ca9ad9771663e94b82b61572/downloadlogs.js",
    )
    parser.add_argument(
        "--downloadlogs-local",
        default=None,
        help="Optional local path to downloadlogs.js (used if remote fetch is blocked).",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headless-false", action="store_true", default=False, help="Force headful mode")

    # Trigger key: downloadlogs.js listens for keycode 83 or 's' / 'S'.
    parser.add_argument("--trigger-key", default="s")
    parser.add_argument("--trigger-retry", type=int, default=2)

    args = parser.parse_args()
    links_file = Path(args.links_file)
    if not links_file.exists():
        raise RuntimeError(f"missing links file: {links_file}")

    if args.storage_state is None and args.user_data_dir is None:
        raise RuntimeError("need either --storage-state or --user-data-dir")

    if args.storage_state is not None and not Path(args.storage_state).exists():
        raise RuntimeError(f"missing storage-state file: {args.storage_state}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    links = _load_links(links_file)
    if args.max_links and args.max_links > 0:
        links = links[: args.max_links]

    headless = bool(args.headless)
    if args.headless_false:
        headless = False

    results: List[Dict[str, Any]] = []

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "playwright is required. Install with `pip install playwright` "
            "and run `playwright install chromium`."
        ) from e

    # Fetch once and re-use for all pages.
    downloadlogs_js = _load_downloadlogs_js(
        url=args.downloadlogs_url,
        local_path=args.downloadlogs_local,
    )
    downloadlogs_js = _patch_downloadlogs_for_capture(downloadlogs_js)

    with sync_playwright() as p:
        context: Any
        browser = None
        if args.user_data_dir is not None:
            # Persistent context supports manual login once, then re-uses cookies.
            context = p.chromium.launch_persistent_context(
                args.user_data_dir,
                headless=headless,
            )
        else:
            browser = p.chromium.launch(headless=headless)
            # accept_downloads 是 context 级别配置。
            context = browser.new_context(storage_state=args.storage_state, accept_downloads=True)

        for i, url in enumerate(links):
            paipu_id = _extract_paipu_id(url)
            if not paipu_id:
                results.append({"i": i, "url": url, "ok": False, "error": "cannot extract paipu id"})
                continue
            paipu_id_safe = _safe_filename(paipu_id)
            out_file = output_dir / f"{paipu_id_safe}.json"

            if args.skip_existing and out_file.exists() and not args.overwrite:
                results.append({"i": i, "url": url, "paipu": paipu_id_safe, "ok": True, "skipped": True})
                continue

            page = context.new_page()
            page.set_default_timeout(args.timeout_sec * 1000)
            page.set_default_navigation_timeout(args.timeout_sec * 1000)
            download_timeout_ms = args.download_timeout_sec * 1000

            try:
                print(f"[{i+1}/{len(links)}] open: {url}", flush=True)
                page.goto(url, wait_until="domcontentloaded")

                # Inject downloadlogs.js source so it registers the keydown handler.
                print(f"[{i+1}/{len(links)}] inject downloadlogs.js...", flush=True)
                page.add_script_tag(content=downloadlogs_js)

                print(f"[{i+1}/{len(links)}] wait for record_uuid...", flush=True)
                _wait_for_record_uuid(page, timeout_ms=args.timeout_sec * 1000)
                time.sleep(0.8)  # small buffer for UI state to settle

                # downloadlogs.js 只有在 UI_Replay/UI_Loading 的 _enable 为 true 时才会触发下载。
                # 这里做一个短暂等待，提升命中率。
                try:
                    ui_wait_ms = min(download_timeout_ms, 5000)
                    page.wait_for_function(
                        """
                        () => {
                            const u = window.uiscript;
                            const en = (scene) => {
                                if (!scene) return false;
                                return !!((scene.Inst && scene.Inst._enable) || (scene._Inst && scene._Inst._enable));
                            };
                            return en(u && u.UI_Replay) || en(u && u.UI_Loading);
                        }
                        """,
                        timeout=ui_wait_ms,
                    )
                except Exception:
                    pass

                last_err: Optional[str] = None
                saved = False
                for attempt in range(args.trigger_retry + 1):
                    try:
                        # Clear previous capture.
                        page.evaluate("() => { window.__downloadlogs_last = null; }")

                        # 直接调用注入脚本里的 downloadlog()，避免依赖 keydown/checkscene。
                        page.evaluate(
                            """
                            () => {
                                if (window.__downloadlogs_downloadlog) {
                                    window.__downloadlogs_downloadlog();
                                }
                            }
                            """
                        )

                        # Wait until the patched download() captured the JSON string.
                        page.wait_for_function(
                            """
                            () => window.__downloadlogs_last && window.__downloadlogs_last.text && window.__downloadlogs_last.text.length > 0
                            """,
                            timeout=download_timeout_ms,
                        )
                        last = page.evaluate("() => window.__downloadlogs_last")
                        text = str(last.get("text", ""))
                        out_file.write_text(text, encoding="utf-8")
                        saved = True
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(0.6)

                if not saved:
                    raise RuntimeError(f"download not triggered: {last_err}")

                results.append({"i": i, "url": url, "paipu": paipu_id_safe, "ok": True, "out": str(out_file)})
            except Exception as e:
                results.append(
                    {
                        "i": i,
                        "url": url,
                        "paipu": paipu_id_safe,
                        "ok": False,
                        "error": str(e),
                    }
                )
            finally:
                try:
                    page.close()
                except Exception:
                    pass

    ok_count = sum(1 for r in results if r.get("ok"))
    out_report = output_dir / "browser_export_report.json"
    out_report.write_text(json.dumps({"total": len(results), "ok": ok_count, "results": results}, ensure_ascii=False, indent=2))
    print(f"done: ok={ok_count}/{len(results)} report={out_report}")


if __name__ == "__main__":
    main()

