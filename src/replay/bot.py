"""跑谱 Review 系统：将 tenhou6 JSON / mjai JSONL 牌谱发给 KeqingBot 进行逐步决策分析，输出 HTML 报告。"""

from __future__ import annotations

import argparse
import base64
import json
import sys as _sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.types import action_dict_to_spec, action_specs_match
from static_tables.demo import annotate_replay_candidate_demo

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for _dir in (str(_SRC_DIR), str(_SCRIPTS_DIR), str(_PROJECT_ROOT)):
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)

# 支持 keqingv1 / keqingv2，v5model 缺失时允许按可选依赖处理
try:
    from v5model.bot import V5Bot as _V5Bot
    from v5model.action_space import action_to_idx as _action_to_idx_v5
except ModuleNotFoundError:
    _V5Bot = None
    _action_to_idx_v5 = None

from keqingv1.bot import KeqingBot as _KeqingBot
from keqingv1.action_space import action_to_idx as _action_to_idx_k1

# bot 类型 → run_replay_from_source 内部创建 Bot 时用
_BOT_CLASSES = {
    "keqingv1": _KeqingBot,
    "keqingv2": _KeqingBot,
    "keqingv3": _KeqingBot,
}
if _V5Bot is not None:
    _BOT_CLASSES["v5"] = _V5Bot

PLAYER_NAMES = ["East", "South", "West", "North"]

# 默认 checkpoint 路径（按 bot 类型，相对于 PROJECT_ROOT）
_DEFAULT_CHECKPOINTS = {
    "v5": _PROJECT_ROOT / "artifacts/models/modelv5/latest.pth",
    "keqingv1": _PROJECT_ROOT / "artifacts/models/keqingv1/latest.pth",
    "keqingv2": _PROJECT_ROOT / "artifacts/models/keqingv2/best.pth",
    "keqingv3": _PROJECT_ROOT / "artifacts/models/keqingv3/best.pth",
}
_RESPONSE_ACTION_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan", "hora"}


def _is_mjai_dict(data: dict) -> bool:
    """判断 dict 是否是 mjai JSONL 格式（根对象有 type 字段且为已知 mjai 类型）。"""
    return data.get("type") in (
        "start_game",
        "start_kyoku",
        "tsumo",
        "dahai",
        "reach",
        "reach_accepted",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "hora",
        "ryukyoku",
        "dora",
        "end_kyoku",
        "end_game",
    )


def _load_events_from_source(
    source: Union[str, Path, dict, list],
    input_type: str = "auto",
) -> list[dict]:
    """从各种输入源加载 mjai 事件列表。

    Parameters
    ----------
    source : str | Path | dict | list
        输入源：
        - str/dict：tenhou6 JSON 对象 或 mjai JSONL 字符串（line-separated JSON）
        - Path：.json 文件（tenhou6 或 mjai JSONL）或 .jsonl 文件（mjai JSONL）
        - list：mjai 事件列表
    input_type : str
        解析模式："auto"（自动检测）、"tenhou6"（强制 tenhou6 格式）、
        "mjai"（强制 mjai JSONL 格式）、"url"（已是 mjai 事件列表，跳过转换）。
    """
    tmp_mjson: Optional[Path] = None

    try:
        # ---------- 已有事件列表（url 模式） ----------
        if input_type == "url":
            if isinstance(source, list):
                return source
            raise ValueError("url 模式下 source 必须是 mjai 事件列表")

        # ---------- 确定 data ----------
        if isinstance(source, (str, Path)):
            path = Path(source) if not isinstance(source, Path) else source
            if path.suffix == ".jsonl":
                return list(read_mjai_jsonl(path))
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # 多行 JSONL 但后缀是 .json，当文件处理
                return list(read_mjai_jsonl(path))
        elif isinstance(source, list):
            return source
        else:
            data = source

        # ---------- 根据 input_type 决定解析方式 ----------
        if input_type == "mjai":
            # 强制当作 mjai JSONL 解析
            if isinstance(data, list):
                return data
            text = json.dumps(data, separators=(",", ":"))
            tmp_mjson = Path(tempfile.mktemp(suffix=".mjson"))
            tmp_mjson.write_text(text, encoding="utf-8")
            return list(read_mjai_jsonl(tmp_mjson))

        if input_type == "tenhou6":
            from convert.tenhou6_utils import tenhou6_to_mjson as _tenhou6_to_mjson

            tmp_mjson = Path(tempfile.mktemp(suffix=".mjson"))
            if not _tenhou6_to_mjson(data, tmp_mjson):
                raise RuntimeError("tenhou6_to_mjson 转换失败")
            return list(read_mjai_jsonl(tmp_mjson))

        # ---------- auto 模式：自动判断 ----------
        if isinstance(data, dict) and _is_mjai_dict(data):
            json_str = json.dumps(data, separators=(",", ":"))
            tmp_mjson = Path(tempfile.mktemp(suffix=".mjson"))
            tmp_mjson.write_text(json_str, encoding="utf-8")
            return list(read_mjai_jsonl(tmp_mjson))

        # 尝试 tenhou6 JSON → convlog 转换
        from convert.tenhou6_utils import tenhou6_to_mjson as _tenhou6_to_mjson

        tmp_mjson = Path(tempfile.mktemp(suffix=".mjson"))
        if not _tenhou6_to_mjson(data, tmp_mjson):
            raise RuntimeError("无法识别输入格式，且 tenhou6_to_mjson 转换失败")
        return list(read_mjai_jsonl(tmp_mjson))

    finally:
        if tmp_mjson and tmp_mjson.exists():
            tmp_mjson.unlink()


def run_replay_from_source(
    source: Union[str, Path, dict, list],
    player_id: int,
    checkpoint: Union[str, Path] | None = None,
    input_type: str = "auto",
    bot_type: str = "keqingv1",
) -> tuple:
    """运行跑谱并返回 (Bot实例, HTML报告字符串)。

    Parameters
    ----------
    source : str | Path | dict | list
        输入源：
        - str: 粘贴的 tenhou6 JSON 文本 或 mjai JSONL 文本 或文件路径
        - Path: .json / .jsonl 文件路径
        - dict: tenhou6 JSON dict
        - list: mjai 事件列表
    player_id : int
        视角座位号 0-3
    checkpoint : str | Path | None
        模型 checkpoint 路径，None 时使用 bot_type 对应的默认路径。
    input_type : str
        输入内容类型："auto"（自动检测）、"tenhou6"（tenhou6 JSON）、"mjai"（mjai JSONL）。
        "url" 模式下 source 已是 mjai 事件列表。
    bot_type : str
        Bot 类型："v5"（V5Bot）或 "keqingv1"（KeqingBot）。
        checkpoint 为 None 时使用对应的默认路径。

    Returns
    -------
    tuple
        (跑谱完毕的 Bot, HTML 报告字符串)
    """
    if checkpoint is None:
        checkpoint = _DEFAULT_CHECKPOINTS[bot_type]
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute() and not checkpoint.exists():
            # 相对路径：尝试相对于 PROJECT_ROOT 解析
            candidate = _PROJECT_ROOT / checkpoint
            if candidate.exists():
                checkpoint = candidate
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint 未找到：{checkpoint}")

    events = _load_events_from_source(source, input_type=input_type)
    bot_cls = _BOT_CLASSES[bot_type]
    bot = bot_cls(player_id=player_id, model_path=checkpoint)
    bot.player_names: list[str] = []  # 从 start_game 提取

    _MELD_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}
    # 其他家操作类型（需要记录观察步）
    _OBS_TYPES = {
        "dahai",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "reach",
        "hora",
        "ryukyoku",
    }

    def _find_pending_decision() -> dict | None:
        for entry in reversed(bot.decision_log):
            if entry.get("is_obs"):
                continue
            if entry.get("gt_action") is None:
                return entry
        return None

    for event_index, event in enumerate(events):
        if event.get("type") == "start_game" and isinstance(event.get("names"), list):
            bot.player_names = [str(n) for n in event["names"]]
        pending = _find_pending_decision()
        if pending is not None:
            if _is_player_action(event, player_id):
                pending["gt_action"] = event
            else:
                chosen = pending.get("chosen") or {}
                candidates = pending.get("candidates", [])
                has_non_none_candidate = any(
                    c.get("action", {}).get("type") != "none" for c in candidates
                )
                # 响应窗口已越过且当前步选择的是“过”，需要把真实动作补成显式 none，
                # 不能留成 null，否则前端会把它当成“没有实际选择”。
                if chosen.get("type") == "none" and has_non_none_candidate:
                    pending["gt_action"] = {"type": "none", "actor": player_id}
        bot.react(event)
        # 记录其他家的操作观察步（棋盘快照，无 bot 推理数据）
        ev_actor = event.get("actor")
        ev_type = event.get("type", "")
        should_record_obs = (
            ev_type in _OBS_TYPES
            and (
                (ev_actor is not None and ev_actor != player_id)
                or (ev_type == "ryukyoku")
            )
        )
        if should_record_obs:
            snap = bot.game_state.snapshot(player_id)
            obs_entry = {
                "step": len(bot.decision_log),
                "bakaze": snap.get("bakaze", ""),
                "kyoku": snap.get("kyoku", 0),
                "honba": snap.get("honba", 0),
                "oya": snap.get("oya", 0),
                "scores": snap.get("scores", []),
                "hand": snap.get("hand", []),
                "discards": snap.get("discards", []),
                "melds": snap.get("melds", []),
                "dora_markers": snap.get("dora_markers", []),
                "reached": snap.get("reached", []),
                "actor_to_move": snap.get("actor_to_move"),
                "last_discard": snap.get("last_discard"),
                "tsumo_pai": None,
                "candidates": [],
                "chosen": event,  # 他家实际执行的动作
                "value": None,
                "gt_action": event,
                "is_obs": True,  # 标记为观察步（非自家决策）
                "obs_kind": (
                    "discard"
                    if ev_type == "dahai"
                    else "meld"
                    if ev_type in _MELD_TYPES
                    else "reach"
                    if ev_type == "reach"
                    else "terminal"
                ),
                "board_phase": "after_action",
                "source_event_index": event_index,
            }
            bot.decision_log.append(obs_entry)

    html = render_html(bot)
    return bot, html


# 牌字符串 -> SVG 文件名映射
_TILE_SVG = {
    **{f"{n}m": f"Man{n}" for n in range(1, 10)},
    **{f"{n}p": f"Pin{n}" for n in range(1, 10)},
    **{f"{n}s": f"Sou{n}" for n in range(1, 10)},
    "5mr": "Man5-Dora",
    "5pr": "Pin5-Dora",
    "5sr": "Sou5-Dora",
    "E": "Ton",
    "S": "Nan",
    "W": "Shaa",
    "N": "Pei",
    "P": "Haku",
    "F": "Hatsu",
    "C": "Chun",
}

# 排序权重：m(0-8) p(9-17) s(18-26) 字(27-33)
_TILE_ORDER = {
    **{f"{n}m": n - 1 for n in range(1, 10)},
    "5mr": 4,
    **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
    "5pr": 13,
    **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    "5sr": 22,
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,
    "F": 32,
    "C": 33,
}

_SVG_DIR = (
    Path(__file__).parent.parent.parent / "tiles" / "riichi-mahjong-tiles" / "Regular"
)


def _svg_b64(tile: str) -> str:
    name = _TILE_SVG.get(tile, "Blank")
    path = _SVG_DIR / f"{name}.svg"
    if not path.exists():
        path = _SVG_DIR / "Blank.svg"
    if path.exists():
        data = path.read_bytes()
    else:
        data = b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"/>'
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def _sort_hand(hand: list[str], tsumo_pai: str | None = None) -> list[str]:
    """排序手牌，tsumo_pai 放在最右边。"""
    others = [t for t in hand if t != tsumo_pai or tsumo_pai is None]
    if tsumo_pai and tsumo_pai in hand:
        others = sorted(
            [t for t in hand if t != tsumo_pai], key=lambda t: _TILE_ORDER.get(t, 99)
        )
        return others + [tsumo_pai]
    return sorted(hand, key=lambda t: _TILE_ORDER.get(t, 99))


def _action_cmp_key(action: dict | None):
    if not action:
        return None
    normalized = dict(action)
    if normalized.get("type") == "pass":
        normalized["type"] = "none"
    try:
        return action_dict_to_spec(normalized)
    except Exception:
        return None


def _candidate_score_for_replay(candidate: dict) -> float | None:
    """统一读取 replay 候选动作分数。

    新版本优先使用 final_score；旧版本兼容 beam_score/logit。
    """
    if candidate.get("final_score") is not None:
        return float(candidate["final_score"])
    if candidate.get("beam_score") is not None:
        return float(candidate["beam_score"])
    if candidate.get("logit") is not None:
        return float(candidate["logit"])
    return None


def _same_action(a: dict | None, b: dict | None) -> bool:
    spec_a = _action_cmp_key(a)
    spec_b = _action_cmp_key(b)
    if spec_a is None or spec_b is None:
        return False
    return action_specs_match(spec_a, spec_b)


def normalize_replay_decisions(decisions: dict, meta: dict | None = None) -> dict:
    if not isinstance(decisions, dict):
        return decisions

    log = decisions.get("log", [])
    player_id = decisions.get("player_id")
    pending_idx: int | None = None
    for idx, entry in enumerate(log):
        if not entry.get("is_obs") and entry.get("gt_action") is None:
            pending_idx = idx
        if pending_idx is None or idx == pending_idx:
            continue
        pending = log[pending_idx]
        if pending.get("gt_action") is not None:
            pending_idx = None
            continue
        chosen = pending.get("chosen") or {}
        candidates = pending.get("candidates", [])
        has_none_candidate = any(
            c.get("action", {}).get("type") == "none" for c in candidates
        )
        has_non_none_candidate = any(
            c.get("action", {}).get("type") != "none" for c in candidates
        )
        if chosen.get("type") == "none" and has_non_none_candidate:
            pending["gt_action"] = {"type": "none", "actor": player_id}
            pending_idx = None
        elif chosen.get("type") in _RESPONSE_ACTION_TYPES and has_none_candidate:
            pending["gt_action"] = {
                **chosen,
                "actor": chosen.get("actor", player_id),
            }
            pending_idx = None

    own_log = [e for e in log if not e.get("is_obs")]
    total_ops = len(own_log)
    match_count = sum(
        1 for e in own_log if _same_action(e.get("chosen"), e.get("gt_action"))
    )
    return {
        **decisions,
        "log": log,
        "total_ops": total_ops,
        "match_count": match_count,
        "bot_type": (meta or {}).get("bot_type", decisions.get("bot_type")),
        "player_names": decisions.get("player_names") or (meta or {}).get("player_names"),
    }


def _tile_img(tile: str, width: int = 40, style: str = "") -> str:
    src = _svg_b64(tile)
    base_border = "border:1px solid #bbb;border-radius:3px;box-sizing:border-box;"
    return f'<img src="{src}" width="{width}" title="{tile}" style="margin:1px;{base_border}{style}">'


def action_label(a: dict) -> str:
    t = a.get("type", "?")
    if t == "dahai":
        tsumogiri = " (摸切)" if a.get("tsumogiri") else ""
        return f"打 {a.get('pai', '?')}{tsumogiri}"
    if t == "reach":
        return "立直"
    if t == "chi":
        return f"吃 {a.get('pai', '?')} ({','.join(a.get('consumed', []))})"
    if t == "pon":
        return f"碰 {a.get('pai', '?')}"
    if t == "daiminkan":
        return f"大明杠 {a.get('pai', '?')}"
    if t == "ankan":
        return f"暗杠 {a.get('pai', '?')}"
    if t == "kakan":
        return f"加杠 {a.get('pai', '?')}"
    if t == "hora":
        return "胡牌"
    if t == "ryukyoku":
        return "流局"
    if t == "none":
        return "过"
    return t


def _action_pai(a: dict) -> str | None:
    """从动作中提取主要牌（用于在手牌上标注）。"""
    t = a.get("type", "")
    if t == "dahai":
        return a.get("pai")
    if t == "reach":
        return None  # 立直本身不标注单张
    if t == "chi":
        return a.get("pai")
    if t == "pon":
        return a.get("pai")
    if t == "daiminkan":
        return a.get("pai")
    return None


def _render_hand_tiles(
    hand: list[str],
    highlight_pai: str | None,
    tsumo_pai: str | None = None,
    highlight_color: str = "#e74c3c",
) -> str:
    """渲染手牌图片行，highlight_pai 对应的牌加红框标注，tsumo_pai 放最右。"""
    sorted_hand = _sort_hand(hand, tsumo_pai)
    imgs = []
    for tile in sorted_hand:
        if tile == highlight_pai:
            style = f"border:2px solid {highlight_color};border-radius:3px;box-sizing:border-box;"
        else:
            style = ""
        imgs.append(_tile_img(tile, width=38, style=style))
    return "".join(imgs)


def _render_candidates_logit(
    candidates: list[dict], chosen: dict, gt_action: dict | None
) -> str:
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
        is_chosen = _same_action(a, chosen)
        is_gt = gt_action is not None and _same_action(a, gt_action)
        pct = max(0, (logit - min_logit) / logit_range * 100)
        bar_color = "#e74c3c" if is_chosen else "#3498db"
        marks = ""
        if is_chosen:
            marks += " ✓Bot"
        if is_gt:
            marks += " ★玩家"
        row_style = (
            "background:#fff3cd;"
            if is_chosen
            else ("background:#d4edda;" if is_gt else "")
        )
        rows.append(f"""
          <tr style="{row_style}">
            <td style="padding:2px 8px;font-weight:{"bold" if is_chosen else "normal"}">{label}{marks}</td>
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
      {"".join(rows)}
    </table>"""


def _render_candidates_tile(
    candidates: list[dict],
    chosen: dict,
    gt_action: dict | None,
    hand: list[str],
    tsumo_pai: str | None = None,
) -> str:
    """模式2：在手牌图上方标注 bot 选择（dahai 类动作）。非 dahai 退回文字显示。"""
    bot_pai = _action_pai(chosen)
    gt_pai = _action_pai(gt_action) if gt_action else None
    sorted_hand = _sort_hand(hand, tsumo_pai)
    imgs = []
    for tile in sorted_hand:
        is_bot = tile == bot_pai
        is_gt = tile == gt_pai
        if is_bot and is_gt:
            label_html = '<div style="text-align:center;font-size:10px;color:#8e44ad;font-weight:bold">✓★</div>'
        elif is_bot:
            label_html = '<div style="text-align:center;font-size:10px;color:#e74c3c;font-weight:bold">✓Bot</div>'
        elif is_gt:
            label_html = '<div style="text-align:center;font-size:10px;color:#27ae60;font-weight:bold">★玩家</div>'
        else:
            label_html = '<div style="height:14px"></div>'
        border = ""
        if is_bot:
            border = "border:2px solid #e74c3c;"
        elif is_gt:
            border = "border:2px solid #27ae60;"
        imgs.append(
            f'<div style="display:inline-block;text-align:center;margin:1px">{label_html}{_tile_img(tile, width=38, style=border + "border-radius:3px;")}</div>'
        )
    # 非 dahai 动作补充文字
    extra = ""
    if chosen.get("type") not in ("dahai", "none"):
        extra = f'<div style="margin-top:4px;font-size:12px">Bot: <b>{action_label(chosen)}</b></div>'
    if gt_action and gt_action.get("type") not in ("dahai", "none"):
        extra += f'<div style="font-size:12px;color:#27ae60">玩家: <b>{action_label(gt_action)}</b></div>'
    return f'<div style="margin:4px 0">{"".join(imgs)}</div>{extra}'


def _render_candidates_bar(
    candidates: list[dict],
    chosen: dict,
    gt_action: dict | None,
    hand: list[str],
    tsumo_pai: str | None = None,
) -> str:
    """柱状模式：每张候选牌上方显示 logit 强度柱（仅 dahai，其他退回 tile 模式）。"""
    if chosen.get("type") not in ("dahai", "none"):
        return _render_candidates_tile(candidates, chosen, gt_action, hand, tsumo_pai)

    if not candidates:
        return ""

    max_logit = candidates[0]["logit"]
    min_logit = candidates[-1]["logit"]
    logit_range = max(max_logit - min_logit, 1e-3)

    tile_logit: dict[str, float] = {}
    for c in candidates:
        a = c["action"]
        if a.get("type") == "dahai":
            tile_logit[a.get("pai", "")] = c["logit"]

    bot_pai = _action_pai(chosen)
    gt_pai = _action_pai(gt_action) if gt_action else None
    sorted_hand = _sort_hand(hand, tsumo_pai)

    imgs = []
    for tile in sorted_hand:
        logit = tile_logit.get(tile, min_logit)
        pct = max(1, (logit - min_logit) / logit_range * 100)
        is_bot = tile == bot_pai
        is_gt = tile == gt_pai

        if is_bot and is_gt:
            bar_color = "#8e44ad"
            marks = "✓★"
        elif is_bot:
            bar_color = "#e74c3c"
            marks = "✓"
        elif is_gt:
            bar_color = "#27ae60"
            marks = "★"
        else:
            bar_color = "#3498db"
            marks = ""

        MAX_BAR_H = 60  # 柱最高 60px
        bar_height = max(4, int(pct / 100 * MAX_BAR_H))
        border = ""
        if is_bot:
            border = "border:2px solid #e74c3c;"
        elif is_gt:
            border = "border:2px solid #27ae60;"
        # tile 渲染为 38px 宽，SVG 原始 300x400，等比高度 ≈ 38*400/300 ≈ 51px
        TILE_H = 51
        # 每列：tile 固定在底部，bar 叠在 tile 上方，bar 底部紧贴 tile 顶部往上长
        imgs.append(
            f'<div style="display:inline-block;width:38px;text-align:center;margin:0 1px;vertical-align:bottom;position:relative">'
            f'<div style="position:absolute;bottom:{TILE_H}px;left:0;width:38px;height:{bar_height}px;background:{bar_color};border-radius:{bar_height}px {bar_height}px 0 0;display:flex;align-items:center;justify-content:center;overflow:hidden">'
            f'<span style="font-size:9px;color:#fff;font-weight:bold;white-space:nowrap">{marks}</span>'
            f"</div>"
            f"{_tile_img(tile, width=38, style=border + 'border-radius:3px;')}"
            f"</div>"
        )

    extra = ""
    if chosen.get("type") not in ("dahai", "none"):
        extra = f'<div style="margin-top:4px;font-size:12px">Bot: <b>{action_label(chosen)}</b></div>'
    if gt_action and gt_action.get("type") not in ("dahai", "none"):
        extra += f'<div style="font-size:12px;color:#27ae60">玩家: <b>{action_label(gt_action)}</b></div>'
    return f'<div style="margin:4px 0;display:flex;flex-wrap:wrap">{"".join(imgs)}</div>{extra}'


def render_replay_json(bot: KeqingBot) -> dict:
    """将 bot.decision_log 转换为前端 Vue 可用的 JSON 数据结构。"""
    log = bot.decision_log

    # 按小局分组
    kyoku_order = []
    seen_keys = set()
    for entry in log:
        key = (entry["bakaze"], entry["kyoku"], entry["honba"])
        if key not in seen_keys:
            seen_keys.add(key)
            kyoku_order.append({"bakaze": key[0], "kyoku": key[1], "honba": key[2]})

    # 重新按顺序编排 step（obs 步插入后编号可能乱）
    for i, entry in enumerate(log):
        entry["step"] = i
        if entry.get("candidates"):
            entry["candidates"] = [
                annotate_replay_candidate_demo(candidate, entry)
                for candidate in entry["candidates"]
            ]

    # 补 kyoku_key 方便前端过滤
    for entry in log:
        entry["kyoku_key"] = {
            "bakaze": entry["bakaze"],
            "kyoku": entry["kyoku"],
            "honba": entry["honba"],
        }

    # total_ops/matches 只统计自家决策步（排除 obs 步）
    own_log = [e for e in log if not e.get("is_obs")]
    total_ops = len(own_log)
    matches = sum(1 for e in own_log if _same_action(e.get("chosen"), e.get("gt_action")))

    # Rating: exp(-alpha * delta_score)，alpha=0.5
    # delta 基于模型最终用于决策的综合分数（final_score）；旧版本回放回退到 beam/logit。
    import math

    _ALPHA = 0.5
    rating_scores = []
    for e in log:
        gt = e.get("gt_action")
        if not gt:
            continue
        candidates = e.get("candidates", [])
        if not candidates:
            continue
        chosen = e.get("chosen", {})

        bot_score = None
        gt_score = None
        for c in candidates:
            s = _candidate_score_for_replay(c)
            if s is None:
                continue
            if _same_action(c["action"], chosen) and bot_score is None:
                bot_score = s
            if _same_action(c["action"], gt) and gt_score is None:
                gt_score = s
        if bot_score is None or gt_score is None:
            continue
        delta = bot_score - gt_score  # >= 0 when bot != gt
        rating_scores.append(math.exp(-_ALPHA * max(delta, 0.0)))

    rating = (
        round(100.0 * sum(rating_scores) / len(rating_scores), 1)
        if rating_scores
        else None
    )

    return normalize_replay_decisions({
        "log": log,
        "kyoku_order": kyoku_order,
        "total_ops": total_ops,
        "match_count": matches,
        "rating": rating,
        "player_id": bot.player_id,
        "player_names": getattr(bot, "player_names", []),
    })


def render_html(bot: KeqingBot, tiles_dir: Path | None = None) -> str:
    global _SVG_DIR
    if tiles_dir is not None:
        _SVG_DIR = tiles_dir
    pid = bot.player_id
    log = bot.decision_log

    # 按小局分组，key=(bakaze, kyoku, honba)
    kyoku_order: list[tuple] = []
    kyoku_steps: dict[tuple, list[int]] = {}
    for entry in log:
        key = (entry["bakaze"], entry["kyoku"], entry["honba"])
        if key not in kyoku_steps:
            kyoku_order.append(key)
            kyoku_steps[key] = []
        kyoku_steps[key].append(entry["step"])

    # 统计 matches（排除 obs 步）
    own_log = [e for e in log if not e.get("is_obs")]
    total_ops = len(own_log)
    matches = sum(1 for e in own_log if _same_action(e.get("chosen"), e.get("gt_action")))

    step_divs = []
    for i, entry in enumerate(log):
        hand = entry["hand"]
        dora = entry["dora_markers"]
        scores = entry["scores"]
        reached = entry["reached"]
        bakaze = entry["bakaze"]
        kyoku = entry["kyoku"]
        honba = entry["honba"]
        chosen = entry["chosen"]
        candidates = entry["candidates"]
        gt_action = entry.get("gt_action")
        tsumo_pai = entry.get("tsumo_pai")
        kyoku_key = (bakaze, kyoku, honba)
        kyoku_idx = kyoku_order.index(kyoku_key)

        score_str = "  ".join(
            f"P{i2}({'R' if reached[i2] else ''}):{scores[i2]}" for i2 in range(4)
        )
        dora_imgs = "".join(_tile_img(t, 32) for t in dora)
        bar_mode_content = _render_candidates_bar(
            candidates, chosen, gt_action, hand, tsumo_pai
        )

        gt_label = action_label(gt_action) if gt_action else "—"
        gt_color = "#27ae60" if _same_action(gt_action, chosen) else "#c0392b"

        # 立直步骤：从下一条 log 取出宣言牌信息（跳过 obs 步）
        reach_dahai_str = ""
        if chosen.get("type") == "reach":
            for j in range(i + 1, len(log)):
                next_entry = log[j]
                if next_entry.get("is_obs"):
                    continue
                next_chosen = next_entry["chosen"]
                next_gt = next_entry.get("gt_action")
                if next_chosen.get("type") == "dahai":
                    bot_pai = next_chosen.get("pai", "?")
                    gt_pai = (
                        next_gt.get("pai", "?")
                        if next_gt and next_gt.get("type") == "dahai"
                        else "—"
                    )
                    reach_dahai_str = f' <span style="font-size:12px;color:#888">（宣言牌 Bot: <b>{bot_pai}</b> 玩家: <b>{gt_pai}</b>）</span>'
                break

        step_id = f"step{entry['step']}"
        step_divs.append(f"""
        <div class="step" data-kyoku="{kyoku_idx}" id="{step_id}" style="margin:10px 0;border:1px solid #ddd;border-radius:6px;padding:10px;font-family:sans-serif;font-size:13px">
          <div style="font-weight:bold;margin-bottom:4px">Step {entry["step"]} | {bakaze}{kyoku}局 {honba}本场</div>
          <div style="color:#555;margin-bottom:4px">点数: {score_str}</div>
          <div style="margin-bottom:4px">Dora指示: {dora_imgs}</div>
          <div style="margin-bottom:4px;font-size:12px">
            Bot选择: <b>{action_label(chosen)}</b> &nbsp;
            玩家实际: <b style="color:{gt_color}">{gt_label}</b>{reach_dahai_str}
          </div>
          <div>{bar_mode_content}</div>
        </div>""")

    num_kyoku = len(kyoku_order)
    # 构建小局标题列表供 JS 使用
    kyoku_labels_js = "["
    for bk, ky, hb in kyoku_order:
        kyoku_labels_js += f'"第{bk}{ky}局 {hb}本场",'
    kyoku_labels_js += "]"

    match_pct = f"{matches / total_ops * 100:.1f}" if total_ops > 0 else "0.0"

    script = f"""
    <script>
    var curKyoku = 0;
    var numKyoku = {num_kyoku};
    var kyokuLabels = {kyoku_labels_js};
    var totalOps = {total_ops};
    var matchCount = {matches};
    var matchPct = '{match_pct}';

    function showKyoku(idx) {{
        curKyoku = idx;
        document.querySelectorAll('.step').forEach(function(el){{
            el.style.display = (parseInt(el.dataset.kyoku) === idx) ? '' : 'none';
        }});
        document.getElementById('kyoku-label').textContent = kyokuLabels[idx] + ' (' + (idx+1) + '/' + numKyoku + ')';
        document.getElementById('btn-prev-kyoku').disabled = (idx === 0);
        document.getElementById('btn-next-kyoku').disabled = (idx === numKyoku - 1);
        var first = document.querySelector('.step[data-kyoku="' + idx + '"]');
        if (first) first.scrollIntoView({{behavior:'smooth', block:'start'}});
    }}

    function prevKyoku() {{ if (curKyoku > 0) showKyoku(curKyoku - 1); }}
    function nextKyoku() {{ if (curKyoku < numKyoku - 1) showKyoku(curKyoku + 1); }}

    function getVisibleSteps() {{
        return Array.from(document.querySelectorAll('.step[data-kyoku="' + curKyoku + '"]'));
    }}

    function prevStep() {{
        var steps = getVisibleSteps();
        for (var i = steps.length - 1; i >= 0; i--) {{
            var rect = steps[i].getBoundingClientRect();
            if (rect.bottom < 80) {{
                steps[i].scrollIntoView({{behavior:'smooth', block:'start'}});
                return;
            }}
        }}
        if (steps.length) steps[0].scrollIntoView({{behavior:'smooth', block:'start'}});
    }}

    function nextStep() {{
        var steps = getVisibleSteps();
        for (var i = 0; i < steps.length; i++) {{
            var rect = steps[i].getBoundingClientRect();
            if (rect.top > 80) {{
                steps[i].scrollIntoView({{behavior:'smooth', block:'start'}});
                return;
            }}
        }}
    }}

    function showStats() {{
        var msg = '操作数: ' + totalOps + '\\nMatches: ' + matchCount + '\\n匹配率: ' + matchPct + '% = ' + matchCount + ' / ' + totalOps;
        alert(msg);
    }}

    // 键盘快捷键
    document.addEventListener('keydown', function(e) {{
        var tag = document.activeElement.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        if (e.key === 'ArrowUp' || e.key === 'k') {{ e.preventDefault(); prevStep(); }}
        if (e.key === 'ArrowDown' || e.key === 'j') {{ e.preventDefault(); nextStep(); }}
        if (e.key === 'ArrowLeft' || e.key === 'h') {{ e.preventDefault(); prevKyoku(); }}
        if (e.key === 'ArrowRight' || e.key === 'l') {{ e.preventDefault(); nextKyoku(); }}
    }});

    window.onload = function() {{ showKyoku(0); }};
    </script>
    """

    btn_style = (
        "padding:5px 12px;font-size:13px;margin-right:6px;cursor:pointer;flex-shrink:0"
    )
    nav_style = "position:sticky;top:0;z-index:100;background:#f0f2f5;padding:8px 0;margin-bottom:10px;display:flex;align-items:center;flex-wrap:wrap;gap:6px;border-bottom:1px solid #ddd"
    return f"""
    <html><head><meta charset='utf-8'></head><body style='font-family:sans-serif'>
    {script}
    <h2 style="margin-bottom:8px">Bot {pid} ({PLAYER_NAMES[pid]}) 决策日志 — 共 {len(log)} 步</h2>
    <div style="{nav_style}">
      <button id="btn-prev-kyoku" onclick="prevKyoku()" style="{btn_style}">◀ 上一局</button>
      <span id="kyoku-label" style="font-weight:bold;font-size:14px;min-width:100px"></span>
      <button id="btn-next-kyoku" onclick="nextKyoku()" style="{btn_style}">下一局 ▶</button>
      <span style="margin-left:8px;display:flex;align-items:center;gap:6px">
        <button onclick="prevStep()" style="{btn_style}">↑上一步</button>
        <button onclick="nextStep()" style="{btn_style}">↓下一步</button>
      </span>
      <span style="margin-left:8px">
        <button onclick="showStats()" style="{btn_style}">📊统计</button>
      </span>
      <span style="margin-left:12px;font-size:11px;color:#888">快捷键: ←/→局  ↑/↓步</span>
    </div>
    {"".join(step_divs)}
    </body></html>
    """


def render_replay_html(bot: KeqingBot) -> str:
    """返回嵌入到 result-view 的 HTML 内容片段（无外层 html/body）。"""
    html = render_html(bot)
    start = html.find("<body")
    if start == -1:
        return html
    start = html.find(">", start) + 1
    end = html.rfind("</body>")
    return html[start:end].strip()


def _is_player_action(event: dict, player_id: int) -> bool:
    """判断此事件是否是 player_id 的实际动作（非元事件）。"""
    ACTION_TYPES = {
        "dahai",
        "reach",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "hora",
        "ryukyoku",
        "none",
    }
    return event.get("type") in ACTION_TYPES and event.get("actor") == player_id


def main():
    parser = argparse.ArgumentParser(description="跑谱 Review 系统")
    parser.add_argument(
        "--input",
        required=True,
        help="tenhou6 JSON / mjai JSONL 文件路径，或直接粘贴 JSON 文本",
    )
    parser.add_argument(
        "--player-id", type=int, default=0, help="review 哪个座位 (0-3)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="模型 checkpoint 路径（默认根据 --bot-type 使用内置路径）",
    )
    parser.add_argument(
        "--bot-type",
        default="keqingv1",
        choices=["v5", "keqingv1"],
        help="Bot 类型：v5（V5Bot）或 keqingv1（KeqingBot）",
    )
    parser.add_argument("--output", default=None, help="HTML 输出路径")
    parser.add_argument(
        "--tiles-dir",
        default=None,
        help="SVG 牌图目录（默认 tiles/riichi-mahjong-tiles/Regular）",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    tiles_dir = Path(args.tiles_dir) if args.tiles_dir else None

    bot, _ = run_replay_from_source(
        args.input,
        player_id=args.player_id,
        checkpoint=checkpoint,
        bot_type=args.bot_type,
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path(f"artifacts/replays/review_p{args.player_id}.html")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(bot, tiles_dir=tiles_dir), encoding="utf-8")
    print(f"Review 完成：{len(bot.decision_log)} 步决策 -> {out_path}")
    import webbrowser

    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
