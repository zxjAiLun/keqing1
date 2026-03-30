# -*- coding: utf-8 -*-
"""FastAPI Web 服务入口 — Replay Review 系统（支持持久化存储）。"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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

    has_files = files and any(f.filename for f in files if f.filename)

    try:
        if has_files:
            return JSONResponse(status_code=400, content={"error": "多文件模式暂不支持"})

        if not json_text.strip():
            return JSONResponse(status_code=400, content={"error": "请上传文件或粘贴 JSON 文本"})

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

        elif input_type == "mjai":
            lines = text.splitlines()
            if all(_is_json_line(l) for l in lines if l.strip()):
                events = [json.loads(l) for l in lines if l.strip()]
            else:
                events = [json.loads(text)]
            bot = run_replay_single_raw(events, player_id=player_id, checkpoint=checkpoint or None, input_type="mjai", bot_type=bot_type)

        else:
            return JSONResponse(status_code=400, content={"error": f"未知的 input_type：{input_type}"})

        result = render_replay_json(bot)
        return JSONResponse(content=result)

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
async def get_replay(replay_id: str):
    """获取回放完整数据（decisions）。"""
    storage = get_storage()
    decisions = storage.load_decisions(replay_id)
    if decisions is None:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 不存在"})
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
    events = storage.load_events(replay_id)
    if not events:
        return JSONResponse(status_code=404, content={"error": f"回放 {replay_id} 事件不存在"})

    total = len(events)
    paged = events[from_step : from_step + limit]
    return JSONResponse(content={
        "events": paged,
        "total": total,
        "next_step": from_step + len(paged) if from_step + len(paged) < total else None,
    })


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


def _is_json_line(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except Exception:
        return False
