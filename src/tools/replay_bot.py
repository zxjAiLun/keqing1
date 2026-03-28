"""跑谱 Review 系统：将 tenhou6 JSON 牌谱发给 V5Bot 进行逐步决策分析，输出 HTML 报告。"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
from pathlib import Path

from convert.tenhou6_utils import tenhou6_to_mjson
from mahjong_env.replay import read_mjai_jsonl
from v5model.bot import V5Bot
from v5model.action_space import action_to_idx

PLAYER_NAMES = ["East", "South", "West", "North"]

# 牌字符串 -> SVG 文件名映射
_TILE_SVG = {
    **{f"{n}m": f"Man{n}" for n in range(1, 10)},
    **{f"{n}p": f"Pin{n}" for n in range(1, 10)},
    **{f"{n}s": f"Sou{n}" for n in range(1, 10)},
    "5mr": "Man5-Dora", "5pr": "Pin5-Dora", "5sr": "Sou5-Dora",
    "E": "Ton", "S": "Nan", "W": "Shaa", "N": "Pei",
    "P": "Haku", "F": "Hatsu", "C": "Chun",
}

# 排序权重：m(0-8) p(9-17) s(18-26) 字(27-33)
_TILE_ORDER = {
    **{f"{n}m": n-1 for n in range(1,10)},
    "5mr": 4,
    **{f"{n}p": 9+n-1 for n in range(1,10)},
    "5pr": 13,
    **{f"{n}s": 18+n-1 for n in range(1,10)},
    "5sr": 22,
    "E":27,"S":28,"W":29,"N":30,"P":31,"F":32,"C":33,
}

_SVG_DIR = Path(__file__).parent.parent.parent / "tiles" / "riichi-mahjong-tiles" / "Regular"


def _svg_b64(tile: str) -> str:
    name = _TILE_SVG.get(tile, "Blank")
    path = _SVG_DIR / f"{name}.svg"
    if not path.exists():
        path = _SVG_DIR / "Blank.svg"
    data = path.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def _sort_hand(hand: list[str]) -> list[str]:
    return sorted(hand, key=lambda t: _TILE_ORDER.get(t, 99))


def _tile_img(tile: str, width: int = 40, style: str = "") -> str:
    src = _svg_b64(tile)
    return f'<img src="{src}" width="{width}" title="{tile}" style="margin:1px;{style}">'


def action_label(a: dict) -> str:
    t = a.get("type", "?")
    if t == "dahai":
        tsumogiri = " (摸切)" if a.get("tsumogiri") else ""
        return f"打 {a.get('pai','?')}{tsumogiri}"
    if t == "reach":      return "立直"
    if t == "chi":        return f"吃 {a.get('pai','?')} ({','.join(a.get('consumed',[]))})"
    if t == "pon":        return f"碰 {a.get('pai','?')}"
    if t == "daiminkan":  return f"大明杠 {a.get('pai','?')}"
    if t == "ankan":      return f"暗杠 {a.get('pai','?')}"
    if t == "kakan":      return f"加杠 {a.get('pai','?')}"
    if t == "hora":       return "胡牌"
    if t == "ryukyoku":   return "流局"
    if t == "none":       return "过"
    return t


def _action_pai(a: dict) -> str | None:
    """从动作中提取主要牌（用于在手牌上标注）。"""
    t = a.get("type", "")
    if t == "dahai":     return a.get("pai")
    if t == "reach":     return None  # 立直本身不标注单张
    if t == "chi":       return a.get("pai")
    if t == "pon":       return a.get("pai")
    if t == "daiminkan": return a.get("pai")
    return None


def _render_hand_tiles(hand: list[str], highlight_pai: str | None, highlight_color: str = "#e74c3c") -> str:
    """渲染手牌图片行，highlight_pai 对应的牌加红框标注。"""
    sorted_hand = _sort_hand(hand)
    imgs = []
    for tile in sorted_hand:
        if tile == highlight_pai:
            style = f"border:2px solid {highlight_color};border-radius:3px;box-sizing:border-box;"
        else:
            style = ""
        imgs.append(_tile_img(tile, width=38, style=style))
    return "".join(imgs)


def _render_candidates_logit(candidates: list[dict], chosen: dict, gt_action: dict | None) -> str:
    """模式1：传统 logit 条形图表格。"""
    if not candidates:
        return ""
    max_logit = candidates[0]["logit"]
    min_logit = candidates[-1]["logit"]
    logit_range = max(max_logit - min_logit, 1e-3)
    rows = []
    for c in candidates:
        a = c["action"]
        logit = c["logit"]
        label = action_label(a)
        is_chosen = (a == chosen)
        is_gt = (gt_action is not None and a == gt_action)
        pct = max(0, (logit - min_logit) / logit_range * 100)
        bar_color = "#e74c3c" if is_chosen else "#3498db"
        marks = ""
        if is_chosen: marks += " ✓Bot"
        if is_gt:     marks += " ★玩家"
        row_style = "background:#fff3cd;" if is_chosen else ("background:#d4edda;" if is_gt else "")
        rows.append(f"""
          <tr style="{row_style}">
            <td style="padding:2px 8px;font-weight:{'bold' if is_chosen else 'normal'}">{label}{marks}</td>
            <td style="padding:2px 8px;text-align:right;font-family:monospace">{logit:+.3f}</td>
            <td style="padding:2px 8px;width:180px">
              <div style="background:#eee;border-radius:3px;height:14px;width:180px">
                <div style="background:{bar_color};height:14px;width:{pct:.1f}%;border-radius:3px"></div>
              </div>
            </td>
          </tr>""")
    return f"""
    <table style="border-collapse:collapse">
      <tr style="background:#f0f0f0">
        <th style="padding:2px 8px;text-align:left">动作</th>
        <th style="padding:2px 8px">Logit</th>
        <th style="padding:2px 8px">相对强度</th>
      </tr>
      {''.join(rows)}
    </table>"""


def _render_candidates_tile(candidates: list[dict], chosen: dict, gt_action: dict | None, hand: list[str]) -> str:
    """模式2：在手牌图上方标注 bot 选择（dahai 类动作）。非 dahai 退回文字显示。"""
    bot_pai = _action_pai(chosen)
    gt_pai = _action_pai(gt_action) if gt_action else None
    sorted_hand = _sort_hand(hand)
    imgs = []
    for tile in sorted_hand:
        is_bot = (tile == bot_pai)
        is_gt  = (tile == gt_pai)
        if is_bot and is_gt:
            label_html = '<div style="text-align:center;font-size:10px;color:#8e44ad;font-weight:bold">✓★</div>'
        elif is_bot:
            label_html = '<div style="text-align:center;font-size:10px;color:#e74c3c;font-weight:bold">✓Bot</div>'
        elif is_gt:
            label_html = '<div style="text-align:center;font-size:10px;color:#27ae60;font-weight:bold">★玩家</div>'
        else:
            label_html = '<div style="height:14px"></div>'
        border = ""
        if is_bot:   border = "border:2px solid #e74c3c;"
        elif is_gt:  border = "border:2px solid #27ae60;"
        imgs.append(f'<div style="display:inline-block;text-align:center;margin:1px">{label_html}{_tile_img(tile, width=38, style=border+"border-radius:3px;")}</div>')
    # 非 dahai 动作补充文字
    extra = ""
    if chosen.get("type") not in ("dahai", "none"):
        extra = f'<div style="margin-top:4px;font-size:12px">Bot: <b>{action_label(chosen)}</b></div>'
    if gt_action and gt_action.get("type") not in ("dahai", "none"):
        extra += f'<div style="font-size:12px;color:#27ae60">玩家: <b>{action_label(gt_action)}</b></div>'
    return f'<div style="margin:4px 0">{"".join(imgs)}</div>{extra}'


def render_html(bot: V5Bot, tiles_dir: Path | None = None) -> str:
    global _SVG_DIR
    if tiles_dir is not None:
        _SVG_DIR = tiles_dir
    pid = bot.player_id
    log = bot.decision_log
    step_divs = []
    for entry in log:
        hand        = entry["hand"]
        dora        = entry["dora_markers"]
        scores      = entry["scores"]
        reached     = entry["reached"]
        bakaze      = entry["bakaze"]
        kyoku       = entry["kyoku"]
        honba       = entry["honba"]
        chosen      = entry["chosen"]
        candidates  = entry["candidates"]
        gt_action   = entry.get("gt_action")

        score_str = "  ".join(f"P{i}({'R' if reached[i] else ''}):{scores[i]}" for i in range(4))
        dora_imgs = "".join(_tile_img(t, 32) for t in dora)
        hand_imgs_logit = _render_hand_tiles(hand, _action_pai(chosen))
        logit_table = _render_candidates_logit(candidates, chosen, gt_action)
        tile_mode_content = _render_candidates_tile(candidates, chosen, gt_action, hand)

        gt_label = action_label(gt_action) if gt_action else "—"
        gt_color = "#27ae60" if (gt_action and gt_action == chosen) else "#c0392b"

        step_id = f"step{entry['step']}"
        step_divs.append(f"""
        <div class="step" style="margin:10px 0;border:1px solid #ddd;border-radius:6px;padding:10px;font-family:sans-serif;font-size:13px">
          <div style="font-weight:bold;margin-bottom:4px">Step {entry['step']} | {bakaze}{kyoku}局 {honba}本场</div>
          <div style="color:#555;margin-bottom:4px">点数: {score_str}</div>
          <div style="margin-bottom:4px">Dora指示: {dora_imgs}</div>
          <div style="margin-bottom:6px">
            <span style="font-size:12px;color:#555">手牌：</span>{hand_imgs_logit}
          </div>
          <div style="margin-bottom:4px;font-size:12px">
            Bot选择: <b>{action_label(chosen)}</b> &nbsp;
            玩家实际: <b style="color:{gt_color}">{gt_label}</b>
          </div>
          <div class="mode-logit" id="{step_id}-logit">{logit_table}</div>
          <div class="mode-tile" id="{step_id}-tile" style="display:none">{tile_mode_content}</div>
        </div>""")

    toggle_js = """
    <script>
    var mode = 'logit';
    function toggleMode() {
        mode = (mode === 'logit') ? 'tile' : 'logit';
        document.querySelectorAll('.mode-logit').forEach(function(el){ el.style.display = mode==='logit' ? '' : 'none'; });
        document.querySelectorAll('.mode-tile').forEach(function(el){ el.style.display = mode==='tile' ? '' : 'none'; });
        document.getElementById('toggle-btn').textContent = mode==='logit' ? '切换到牌面模式' : '切换到Logit模式';
    }
    </script>
    """
    return f"""
    <html><body>
    {toggle_js}
    <h2>Bot {pid} ({PLAYER_NAMES[pid]}) 决策日志 — 共 {len(log)} 步</h2>
    <button id="toggle-btn" onclick="toggleMode()" style="margin-bottom:12px;padding:6px 16px;font-size:13px">切换到牌面模式</button>
    {''.join(step_divs)}
    </body></html>
    """


def _is_player_action(event: dict, player_id: int) -> bool:
    """判断此事件是否是 player_id 的实际动作（非元事件）。"""
    ACTION_TYPES = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku", "none"}
    return event.get("type") in ACTION_TYPES and event.get("actor") == player_id


def main():
    parser = argparse.ArgumentParser(description="跑谱 Review 系统")
    parser.add_argument("--input",       required=True, help="tenhou6 JSON 文件路径")
    parser.add_argument("--player-id",   type=int, default=0, help="review 哪个座位 (0-3)")
    parser.add_argument("--checkpoint",  default="artifacts/models/modelv5/latest.pth")
    parser.add_argument("--output",      default=None, help="HTML 输出路径")
    parser.add_argument("--tiles-dir",   default=None, help="SVG 牌图目录（默认 tiles/riichi-mahjong-tiles/Regular）")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint 未找到：{checkpoint}")
        raise SystemExit(1)

    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else None

    t6 = json.loads(Path(args.input).read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(suffix=".mjson", delete=False) as f:
        tmp_mjson = Path(f.name)

    try:
        if not tenhou6_to_mjson(t6, tmp_mjson):
            print("ERROR: convlog 转换失败")
            raise SystemExit(1)
        events = read_mjai_jsonl(tmp_mjson)
    finally:
        if tmp_mjson.exists():
            tmp_mjson.unlink()

    bot = V5Bot(player_id=args.player_id, model_path=checkpoint)
    for event in events:
        # 玩家的实际动作事件：回填到上一条 decision_log 的 gt_action
        if _is_player_action(event, args.player_id) and bot.decision_log:
            last = bot.decision_log[-1]
            if last.get("gt_action") is None:
                last["gt_action"] = event
        bot.react(event)

    out_path = Path(args.output) if args.output else Path(f"artifacts/replays/review_p{args.player_id}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(bot, tiles_dir=tiles_dir), encoding="utf-8")
    print(f"Review 完成：{len(bot.decision_log)} 步决策 -> {out_path}")
    import webbrowser
    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
