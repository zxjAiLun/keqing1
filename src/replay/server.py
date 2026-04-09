# -*- coding: utf-8 -*-
"""FastAPI Web 服务入口 — Replay Review 系统（支持持久化存储）。"""
from __future__ import annotations

import json
import sys
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from replay.normalize import normalize_replay_decisions


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from mjlog2mjai_parse import parse_mjlog_to_mjai

app = FastAPI(title="Keqing Unified Server", description="立直麻将 Review + 对战服务")

# ========== 合并 Battle Router ==========
from gateway.api.battle import router as battle_router
app.include_router(battle_router)

# ========== 静态资源 ==========
app.mount("/tiles", StaticFiles(directory=BASE_DIR.parent.parent / "tiles" / "riichi-mahjong-tiles" / "Regular"), name="tiles")

# 挂载 React 构建产物（生产环境）
_REACT_DIST = BASE_DIR.parent / "replay_ui" / "dist"
if _REACT_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_REACT_DIST / "assets")), name="assets")

# ========== 存储相关 ==========

from replay.storage import get_storage


def _normalize_replay_events(events: list[dict] | None) -> list[dict]:
    if not events:
        return events or []

    from copy import deepcopy

    from mahjong_env.scoring import score_hora
    from mahjong_env.state import GameState, apply_event

    normalized = [deepcopy(ev) for ev in events]
    state = GameState()

    for idx, event in enumerate(normalized):
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "hora":
            actor = int(event.get("actor", 0))
            target = int(event.get("target", actor))
            is_tsumo = actor == target
            pai = event.get("pai")
            if not pai:
                if is_tsumo:
                    pai = state.last_tsumo_raw[actor] or state.last_tsumo[actor]
                elif state.last_discard:
                    pai = state.last_discard.get("pai_raw") or state.last_discard.get("pai")
            ura_markers = [str(m) for m in (event.get("ura_dora_markers") or event.get("ura_markers") or [])]
            if pai:
                try:
                    result = score_hora(
                        state,
                        actor=actor,
                        target=target,
                        pai=str(pai),
                        is_tsumo=is_tsumo,
                        ura_dora_markers=ura_markers,
                    )
                    deltas = list(result.deltas)
                    scores = [int(state.scores[i] + deltas[i]) for i in range(4)]
                    normalized[idx] = {
                        **event,
                        "pai": str(pai),
                        "is_tsumo": is_tsumo,
                        "han": result.han,
                        "fu": result.fu,
                        "yaku": result.yaku,
                        "yaku_details": result.yaku_details,
                        "cost": result.cost,
                        "deltas": deltas,
                        "scores": scores,
                        "honba": int(event.get("honba", state.honba)),
                        "kyotaku": int(event.get("kyotaku", state.kyotaku)),
                        "ura_dora_markers": ura_markers,
                    }
                    event = normalized[idx]
                except Exception:
                    pass

        try:
            apply_event(state, event)
        except Exception:
            continue

    return normalized


def _merge_terminal_event_details(decisions: dict, events: list[dict] | None) -> dict:
    if not isinstance(decisions, dict) or not events:
        return decisions

    event_lookup = {
        idx: event
        for idx, event in enumerate(events)
        if isinstance(event, dict)
    }

    for entry in decisions.get("log", []):
        source_event_index = entry.get("source_event_index")
        if source_event_index is None:
            continue
        event = event_lookup.get(int(source_event_index))
        if not event:
            continue
        action = entry.get("gt_action") or entry.get("chosen") or {}
        if action.get("type") not in {"hora", "ryukyoku"}:
            continue
        if entry.get("chosen"):
            entry["chosen"] = {**entry["chosen"], **event}
        if entry.get("gt_action"):
            entry["gt_action"] = {**entry["gt_action"], **event}
    return decisions

def _infer_player_bot_type(player_name: str | None, fallback: str | None = None) -> str:
    raw = (player_name or "").lower()
    if "keqingv1" in raw:
        return "keqingv1"
    if "xmodel1" in raw:
        return "xmodel1"
    if "keqingv3" in raw:
        return "keqingv3"
    if "keqingv2" in raw:
        return "keqingv2"
    return fallback or "keqingv1"


def _default_checkpoint_for_bot_type(bot_type: str) -> Path:
    mapping = {
        "keqingv1": BASE_DIR.parent.parent / "artifacts" / "models" / "keqingv1" / "best.pth",
        "keqingv2": BASE_DIR.parent.parent / "artifacts" / "models" / "keqingv2" / "best.pth",
        "keqingv3": BASE_DIR.parent.parent / "artifacts" / "models" / "keqingv3" / "best.pth",
        "xmodel1": BASE_DIR.parent.parent / "artifacts" / "models" / "xmodel1" / "best.pth",
    }
    return mapping[bot_type]


# ========== 旧接口（兼容） ==========

@app.post("/api/replay")
async def replay(
    player_id: Annotated[int, Form()] = 0,
    checkpoint: Annotated[str, Form()] = "",
    bot_type: Annotated[str, Form()] = "keqingv1",
    files: Annotated[list[UploadFile], File()] = [],
    json_text: Annotated[str, Form()] = "",
    input_type: Annotated[str, Form()] = "url",
):
    """Python 只返回 JSON，前端 Vue 渲染。（兼容旧接口）"""
    from replay.api import run_replay_single_raw
    from replay.bot import render_replay_json
    storage = get_storage()

    has_files = files and any(f.filename for f in files if f.filename)
    events: list[dict] | None = None

    try:
        if has_files:
            valid_files = [f for f in files if f.filename]
            if len(valid_files) > 1:
                return JSONResponse(status_code=400, content={"error": "暂不支持多文件跑谱，请一次上传一个文件"})
            upload = valid_files[0]
            text = (await upload.read()).decode("utf-8", errors="replace").strip()
            if not text:
                return JSONResponse(status_code=400, content={"error": "上传文件为空"})
        elif not json_text.strip():
            return JSONResponse(status_code=400, content={"error": "请上传文件或粘贴 JSON 文本"})
        else:
            text = json_text.strip()

        if input_type == "url":
            parsed = urlparse(text)
            qs = parse_qs(parsed.query)
            ids = qs.get("log", [])
            if not ids:
                return JSONResponse(status_code=400, content={"error": "天凤链接中未找到 log 参数"})
            log_id = ids[0]

            xml_url = f"https://tenhou.net/0/log/?{log_id}"
            req = Request(xml_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; mahjong-research-bot/1.0)",
                "Referer": "https://tenhou.net/",
            })
            with urlopen(req, timeout=30) as resp:
                xml_str = resp.read().decode("utf-8", errors="replace")
            if "<mjloggm" not in xml_str and "<mjlog" not in xml_str.lower():
                return JSONResponse(status_code=400, content={"error": "XML 内容异常，可能牌谱不存在或需要权限"})

            root = ET.fromstring(xml_str)
            mjson_str = parse_mjlog_to_mjai(root)
            events = [json.loads(l) for l in mjson_str.splitlines() if l.strip()]
            bot = run_replay_single_raw(events, player_id=player_id, checkpoint=checkpoint or None, input_type="url", bot_type=bot_type)

        elif input_type == "tenhou6":
            data = json.loads(text)
            bot = run_replay_single_raw(data, player_id=player_id, checkpoint=checkpoint or None, input_type="tenhou6", bot_type=bot_type)
            try:
                from replay.bot import _parse_to_events

                events = _parse_to_events(data, input_type="tenhou6")
            except Exception:
                events = None

        elif input_type == "mjai":
            lines = text.splitlines()
            if len(lines) > 1:
                events = [json.loads(l) for l in lines if l.strip()]
            else:
                # 单行：判断是 JSON 数组还是普通 JSON 对象
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        events = parsed  # JSON 数组直接使用
                    else:
                        events = [parsed]  # 单个 JSON 对象包装成列表
                except json.JSONDecodeError:
                    events = [json.loads(l) for l in lines if l.strip()]
            bot = run_replay_single_raw(events, player_id=player_id, checkpoint=checkpoint or None, input_type="mjai", bot_type=bot_type)

        else:
            return JSONResponse(status_code=400, content={"error": f"未知的 input_type：{input_type}"})

        normalized_events = _normalize_replay_events(events)
        result = normalize_replay_decisions(render_replay_json(bot))
        result = _merge_terminal_event_details(result, normalized_events)
        if events:
            replay_id = storage.save(
                events=normalized_events,
                decisions=result,
                bot_type=bot_type,
                player_names=result.get("player_names"),
            )
            result = {
                **result,
                "replay_id": replay_id,
            }
        return Response(
            content=json.dumps(result, cls=_NumpyEncoder, ensure_ascii=False),
            media_type="application/json",
        )

    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"error": f"JSON 解析失败: {e}"})
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})


# ========== 持久化存储 API（新增） ==========

@app.post("/api/replay/save", response_class=JSONResponse)
async def save_replay(
    events: list[dict] = Form(...),
    decisions: dict = Form(...),
    bot_type: str = Form("keqingv1"),
    player_names: str = Form(""),
):
    """手动保存回放到存储（内部使用或高级用户）。"""
    storage = get_storage()
    names = player_names.split(",") if player_names else None
    replay_id = storage.save(
        events=events,
        decisions=decisions,
        bot_type=bot_type,
        player_names=names,
    )
    return JSONResponse(content={"replay_id": replay_id, "status": "saved"})


@app.get("/api/replay/list", response_class=JSONResponse)
async def list_replays():
    """列出所有已保存的回放（按时间倒序）。"""
    storage = get_storage()
    metas = storage.list()
    return JSONResponse(content=metas)


@app.get("/api/replay/{replay_id}", response_class=JSONResponse)
async def get_replay(replay_id: str, player_id: int | None = None):
    """获取回放完整数据（decisions）。"""
    storage = get_storage()
    decisions = storage.load_decisions(replay_id)
    if decisions is None:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 不存在"})
    meta = storage.load_meta(replay_id) or {}
    events = _normalize_replay_events(storage.load_events(replay_id))
    if (
        isinstance(decisions, dict)
        and player_id is not None
        and decisions.get("player_id") != player_id
    ):
        from replay.api import run_replay_single_raw
        from replay.bot import render_replay_json

        if not events:
            return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 事件不存在"})
        player_names = meta.get("player_names") or decisions.get("player_names") or []
        fallback_bot_type = meta.get("bot_type") or decisions.get("bot_type") or "keqingv1"
        player_name = player_names[player_id] if 0 <= player_id < len(player_names) else None
        bot_type = _infer_player_bot_type(player_name, fallback=fallback_bot_type)
        checkpoint = _default_checkpoint_for_bot_type(bot_type)
        bot = run_replay_single_raw(
            events,
            player_id=player_id,
            checkpoint=checkpoint,
            input_type="url",
            bot_type=bot_type,
        )
        decisions = render_replay_json(bot)
        decisions["bot_type"] = bot_type
    if isinstance(decisions, dict):
        decisions = normalize_replay_decisions(decisions, meta=meta)
        decisions = _merge_terminal_event_details(decisions, events)
    return JSONResponse(content=decisions)


@app.get("/api/replay/{replay_id}/meta", response_class=JSONResponse)
async def get_replay_meta(replay_id: str):
    """获取回放元信息。"""
    storage = get_storage()
    meta = storage.load_meta(replay_id)
    if meta is None:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 不存在"})
    return JSONResponse(content=meta)


@app.get("/api/replay/{replay_id}/events", response_class=JSONResponse)
async def get_replay_events(replay_id: str, from_step: int = 0, limit: int = 50):
    """分页获取 mjai 事件流。"""
    storage = get_storage()
    events = _normalize_replay_events(storage.load_events(replay_id))
    if not events:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 事件不存在"})

    total = len(events)
    paged = events[from_step : from_step + limit]
    return JSONResponse(content={
        "events": paged,
        "total": total,
        "next_step": from_step + len(paged) if from_step + len(paged) < total else None,
    })


def _collect_replay_groups():
    """递归聚合项目中的回放导出清单。"""
    project_root = BASE_DIR.parent.parent
    groups = []

    manifest_paths = []
    for pattern in ("**/replays/manifest.json", "**/anomaly_replays/manifest.json"):
        manifest_paths.extend(project_root.glob(pattern))

    for manifest_path in sorted(
        manifest_paths,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        export_dir = manifest_path.parent
        run_dir = export_dir.parent
        collection_type = export_dir.name

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        stats_path = run_dir / "stats.json"
        stats = None
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except Exception:
                stats = None

        groups.append(
            {
                "output_dir": run_dir.name,
                "output_dir_path": str(run_dir),
                "manifest_path": str(manifest_path),
                "collection_type": collection_type,
                "updated_at": manifest_path.stat().st_mtime,
                "stats": {
                    "games": stats.get("games"),
                    "completed_games": stats.get("completed_games"),
                    "error_games": len(stats.get("error_games", [])),
                    "seconds_per_game": stats.get("seconds_per_game"),
                    "avg_turns": stats.get("avg_turns"),
                }
                if isinstance(stats, dict)
                else None,
                "items": manifest,
            }
        )
    return groups


@app.get("/api/selfplay/replay-collections", response_class=JSONResponse)
async def list_selfplay_replay_collections():
    """列出项目内通用对局回放清单，供 ReplayUI 浏览。"""
    return JSONResponse(content={"groups": _collect_replay_groups()})


@app.get("/api/selfplay/anomaly-replays", response_class=JSONResponse)
async def list_selfplay_anomaly_replays():
    """兼容旧接口，返回相同的聚合对局回放清单。"""
    return JSONResponse(content={"groups": _collect_replay_groups()})


@app.delete("/api/replay/{replay_id}", response_class=JSONResponse)
async def delete_replay(replay_id: str):
    """删除回放。"""
    storage = get_storage()
    deleted = storage.delete(replay_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 不存在"})
    return JSONResponse(content={"ok": True, "replay_id": replay_id})


# ========== 导出 & 健康检查 ==========

@app.post("/api/export-html")
async def export_html(data: str = Form("data")):
    """将 replay JSON 数据生成为独立离线 HTML 文件下载。"""
    try:
        replay_data = json.loads(data)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON 数据"})

    template = (BASE_DIR / "templates" / "replay_view.html").read_text(encoding="utf-8")
    init_script = f'<script>window.__replayData = {json.dumps(replay_data, ensure_ascii=False)};</script>'
    html = template.replace('</body>', f'{init_script}</body>')

    pid = replay_data.get("player_id", 0)
    names = ["E", "S", "W", "N"]
    bakaze = ""
    if replay_data.get("kyoku_order"):
        first = replay_data["kyoku_order"][0]
        bakaze = first.get("bakaze", "E") if isinstance(first, dict) else "E"
    filename = f"replay_{bakaze}_{names[pid]}_{len(replay_data.get('log', []))}steps.html"

    from fastapi.responses import Response
    return Response(
        content=html.encode("utf-8"),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ========== 页面路由（React SPA）==========
# 注意：必须在所有 API 路由之后定义，避免 fallback 抢先匹配 /api/*

@app.get("/", response_class=HTMLResponse)
async def index():
    """主入口：React SPA（前端路由由 react-router-dom 处理）。"""
    return (_REACT_DIST / "index.html").read_text(encoding="utf-8")


@app.get("/replay", response_class=HTMLResponse)
async def replay_page():
    """兼容 /replay 路由（由 React Router 处理）。"""
    return (_REACT_DIST / "index.html").read_text(encoding="utf-8")


@app.get("/{path:path}", response_class=HTMLResponse)
async def spa_fallback(path: str):
    """所有未匹配路由都 fallback 到 index.html，由 React Router 处理。"""
    return (_REACT_DIST / "index.html").read_text(encoding="utf-8")
