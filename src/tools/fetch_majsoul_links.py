from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set
from urllib.parse import urljoin


def _extract_links_with_playwright(
    page_url: str,
    keyword: str,
    max_scrolls: int,
    scroll_pause_ms: int,
) -> List[str]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "playwright is required. Install with `pip install playwright` "
            "and run `playwright install chromium`."
        ) from e

    out: Set[str] = set()
    kw = keyword.lower()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(page_url, wait_until="networkidle", timeout=60_000)

        for _ in range(max_scrolls):
            # Collect candidate links whose surrounding row/card contains keyword.
            links = page.evaluate(
                """(kw) => {
                    const toAbs = (href) => {
                        try { return new URL(href, window.location.href).toString(); }
                        catch { return href; }
                    };
                    const candidates = Array.from(document.querySelectorAll('a[href]'));
                    const picked = [];
                    for (const a of candidates) {
                        const href = a.getAttribute('href') || '';
                        if (!href) continue;
                        if (!(href.includes('paipu=') || href.includes('/view_game/'))) continue;
                        const row = a.closest('tr, li, .row, .card, .list-group-item, div');
                        const text = (row ? row.textContent : a.textContent || '').toLowerCase();
                        if (!text.includes(kw)) continue;
                        picked.push(toAbs(href));
                    }
                    return picked;
                }""",
                kw,
            )
            out.update(links)

            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(scroll_pause_ms)

        browser.close()

    return sorted(out)


def _normalize_link(link: str, page_url: str) -> str:
    return urljoin(page_url, link)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Amae-koromo player page URL")
    parser.add_argument("--keyword", default="keqing1", help="Case-insensitive name keyword")
    parser.add_argument("--output", default="majsoul_links.txt", help="Output links file")
    parser.add_argument("--max-scrolls", type=int, default=20)
    parser.add_argument("--scroll-pause-ms", type=int, default=1200)
    args = parser.parse_args()

    links = _extract_links_with_playwright(
        page_url=args.url,
        keyword=args.keyword,
        max_scrolls=args.max_scrolls,
        scroll_pause_ms=args.scroll_pause_ms,
    )
    links = [_normalize_link(x, args.url) for x in links]

    out_path = Path(args.output)
    out_path.write_text("\n".join(links) + ("\n" if links else ""), encoding="utf-8")
    print(f"wrote {len(links)} links to {out_path}")


if __name__ == "__main__":
    main()

