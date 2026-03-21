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
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    paipu = qs.get("paipu", [None])[0]
    if paipu:
        return paipu

    m = re.search(r"/view_game/([^/?#]+)", parsed.path)
    if m:
        return m.group(1)

    m = re.search(r"paipu=([^&#?/]+)", url)
    if m:
        return m.group(1)

    return None


def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("._")


def _extract_game_key_and_player(paipu_id: str) -> tuple[str, Optional[str]]:
    """
    Majsoul paipu id often looks like:
      230526-<uuid>_a67401109
    where the last segment is a player-specific suffix (the trailing `a` + digits).

    For duplicate exports that differ only by the trailing `_a<number>`, we treat the
    "game key" (everything before the last `_<letter><digits>`) as identical.
    """
    # Split by last "_<letter><digits>".
    m = re.match(r"^(.*)_([a-zA-Z]\d+)$", paipu_id)
    if not m:
        return paipu_id, None
    base = m.group(1)
    suffix = m.group(2)  # like 'a67401109'
    if suffix.startswith("a"):
        return base, suffix[1:]
    return base, None


def _load_links(links_file: Path) -> List[str]:
    links: List[str] = []
    for line in links_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        links.append(line)
    return links


def _load_downloadlogs_js(*, url: str, local_path: Optional[str]) -> str:
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")
    with urlopen(url, timeout=30) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _patch_downloadlogs_to_expose_parse(js_text: str) -> str:
    """
    Insert `window.__downloadlogs_parse = parse;` before the IIFE ends.
    """
    marker_old = "})();\n// vim: ts=4  et"
    marker_new = "window.__downloadlogs_parse = parse;\n})();\n// vim: ts=4  et"
    if marker_old not in js_text:
        raise RuntimeError("downloadlogs.js marker not found; can't expose parse()")
    return js_text.replace(marker_old, marker_new, 1)


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
    parser.add_argument("--fetch-timeout-sec", type=int, default=120, help="fetchGameRecord 超时（秒）")
    parser.add_argument("--parse-timeout-sec", type=int, default=120, help="parse(record) 超时（秒）")
    parser.add_argument(
        "--dedup-by-game-key",
        action="store_true",
        default=True,
        help="按 paipu 去掉结尾 `_a<number>` 后的 game key 去重，避免同一局导出多份。",
    )
    parser.add_argument(
        "--no-dedup-by-game-key",
        action="store_false",
        dest="dedup_by_game_key",
        help="关闭 game-key 去重（仅按文件名精确跳过）。",
    )

    parser.add_argument("--storage-state", default=None, help="Playwright storageState.json path")
    parser.add_argument("--user-data-dir", default=None, help="Persistent Chromium profile directory")

    parser.add_argument(
        "--downloadlogs-url",
        default="https://gist.githubusercontent.com/Equim-chan/875a232a2c1d31181df8b3a8704c3112/raw/a0533ae7a0ab0158ca9ad9771663e94b82b61572/downloadlogs.js",
    )
    parser.add_argument("--downloadlogs-local", default=None, help="Optional local path for downloadlogs.js")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--headless-false", action="store_true", default=False)

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

    downloadlogs_js = _load_downloadlogs_js(url=args.downloadlogs_url, local_path=args.downloadlogs_local)
    downloadlogs_js_with_parse = _patch_downloadlogs_to_expose_parse(downloadlogs_js)

    results: List[Dict[str, Any]] = []

    # Build set of already-exported "game keys" to skip duplicate paipu suffixes.
    existing_game_keys: set[str] = set()
    if args.skip_existing and args.dedup_by_game_key and output_dir.exists():
        for p in output_dir.glob("*.json"):
            if p.name == "browser_export_report.json":
                continue
            stem = p.stem
            game_key, _player = _extract_game_key_and_player(stem)
            existing_game_keys.add(game_key)

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("playwright is required. Install playwright.") from e

    with sync_playwright() as p:
        context: Any
        if args.user_data_dir is not None:
            context = p.chromium.launch_persistent_context(
                args.user_data_dir,
                headless=headless,
            )
        else:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(storage_state=args.storage_state, accept_downloads=True)

        for i, url in enumerate(links):
            paipu_id = _extract_paipu_id(url)
            if not paipu_id:
                results.append({"i": i, "url": url, "ok": False, "error": "cannot extract paipu id"})
                continue

            paipu_id_safe = _safe_filename(paipu_id)
            game_key, player_num = _extract_game_key_and_player(paipu_id)
            out_file = output_dir / f"{paipu_id_safe}.json"

            if args.skip_existing and not args.overwrite:
                if args.dedup_by_game_key and game_key in existing_game_keys:
                    results.append(
                        {
                            "i": i,
                            "url": url,
                            "paipu": paipu_id_safe,
                            "game_key": game_key,
                            "player_num": player_num,
                            "ok": True,
                            "skipped": True,
                            "reason": "dedup-by-game-key",
                        }
                    )
                    continue
                if out_file.exists():
                    results.append(
                        {
                            "i": i,
                            "url": url,
                            "paipu": paipu_id_safe,
                            "game_key": game_key,
                            "player_num": player_num,
                            "ok": True,
                            "skipped": True,
                            "reason": "existing-file",
                        }
                    )
                    continue

            page = context.new_page()
            page.set_default_timeout(args.timeout_sec * 1000)
            page.set_default_navigation_timeout(args.timeout_sec * 1000)

            try:
                print(f"[{i+1}/{len(links)}] open: {url}", flush=True)
                page.goto(url, wait_until="domcontentloaded")

                # Wait until GameMgr record_uuid is ready.
                page.wait_for_function(
                    """
                    () => window.GameMgr && window.GameMgr.Inst && window.GameMgr.Inst.record_uuid && String(window.GameMgr.Inst.record_uuid).length > 0
                    """,
                    timeout=args.timeout_sec * 1000,
                )

                # Fetch game record and store it into window (callback may not preserve parse; we only store record).
                page.evaluate("() => { window.__fetch_record_obj = null; window.__fetch_record_ready = false; }")
                page.evaluate(
                    """
                    () => {
                        const rid = window.GameMgr.Inst.record_uuid;
                        const ver = window.GameMgr.Inst.getClientVersion();
                        window.app.NetAgent.sendReq2Lobby(
                            'Lobby',
                            'fetchGameRecord',
                            { game_uuid: rid, client_version_string: ver },
                            function(i, record) {
                                window.__fetch_record_obj = record;
                                window.__fetch_record_ready = true;
                            }
                        );
                    }
                    """
                )

                page.wait_for_function(
                    """
                    () => window.__fetch_record_ready === true && window.__fetch_record_obj != null
                    """,
                    timeout=args.fetch_timeout_sec * 1000,
                )

                # Re-inject downloadlogs.js with exposed parse() because callback realm may clear earlier injected globals.
                page.add_script_tag(content=downloadlogs_js_with_parse)

                page.wait_for_function(
                    """() => typeof window.__downloadlogs_parse === 'function'""",
                    timeout=30000,
                )

                # Convert record -> tenhou6 JSON (return as string).
                page.wait_for_function(
                    """() => window.__fetch_record_obj != null""",
                    timeout=10000,
                )
                json_text = page.evaluate(
                    """
                    () => {
                        const results = window.__downloadlogs_parse(window.__fetch_record_obj);
                        return JSON.stringify(results);
                    }
                    """
                )

                # Save. Ensure it's a string.
                if not isinstance(json_text, str):
                    json_text = str(json_text)
                out_file.write_text(json_text, encoding="utf-8")
                if args.dedup_by_game_key:
                    existing_game_keys.add(game_key)
                results.append(
                    {
                        "i": i,
                        "url": url,
                        "paipu": paipu_id_safe,
                        "game_key": game_key,
                        "player_num": player_num,
                        "ok": True,
                        "out": str(out_file),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "i": i,
                        "url": url,
                        "paipu": paipu_id_safe if paipu_id else None,
                        "game_key": game_key if paipu_id else None,
                        "player_num": player_num if paipu_id else None,
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

