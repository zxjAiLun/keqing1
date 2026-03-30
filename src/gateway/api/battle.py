from __future__ import annotations

import asyncio
import os
import sys
import gc
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, APIRouter, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from gateway.battle import BattleConfig, BattleManager, BattleRoom, get_manager
from gateway.bot_driver import BotDriver
from mahjong_env.legal_actions import enumerate_legal_actions

app = FastAPI()
manager = get_manager()

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
tiles_path = os.path.join(project_root, "tiles", "riichi-mahjong-tiles", "Regular")
if os.path.exists(tiles_path):
    app.mount("/tiles", StaticFiles(directory=tiles_path), name="tiles")

router = APIRouter(prefix="/api/battle")

# NOTE: router 在文件末尾 include 到 app（需在所有路由定义之后）

bots: Dict[int, Any] = {}
BOT_TYPE = os.environ.get("BOT_TYPE", "v5")


def _cleanup_bot(bot: Any) -> None:
    """释放 bot 占用的模型资源"""
    if hasattr(bot, "model") and bot.model is not None:
        del bot.model
    if hasattr(bot, "optimizer") and bot.optimizer is not None:
        del bot.optimizer


def _cleanup_all_bots() -> None:
    """清理所有 bot 并强制 GC"""
    global bots
    for bot in bots.values():
        _cleanup_bot(bot)
    bots.clear()
    gc.collect()


# 当前选用的 bot 模型名称（由 start_battle 设置）
current_bot_model: str = "modelv5"


def get_or_create_bot(bot_id: int) -> Any:
    if bot_id not in bots:
        import v5model

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(v5model.__file__))
        )
        model_name = current_bot_model
        if model_name in ("keqingv1", "keqingv2"):
            from keqingv1.bot import KeqingBot

            model_path = os.path.join(
                project_root, "artifacts", "models", model_name, "best.pth"
            )
            print(f"[Bot] Loading {model_name} model from: {model_path}")
            bots[bot_id] = KeqingBot(
                player_id=bot_id, model_path=model_path, device="cpu", verbose=False
            )
        else:
            from v5model.bot import V5Bot

            model_path = os.path.join(
                project_root, "artifacts", "models", model_name, "best.pth"
            )
            print(f"[Bot] Loading {model_name} model from: {model_path}")
            bots[bot_id] = V5Bot(
                player_id=bot_id, model_path=model_path, device="cpu", verbose=False
            )
    return bots[bot_id]


bot_driver = BotDriver(manager, get_or_create_bot)
_advance_lock = asyncio.Lock()


class PlayerInfo(BaseModel):
    id: str
    name: str
    type: str


class StartBattleRequest(BaseModel):
    player_name: str = "Player"
    bot_count: int = 3
    seed: Optional[int] = None
    bot_model: str = "modelv5"  # modelv5 / modelv5_naga / keqingv1 / keqingv2


class StartBattleResponse(BaseModel):
    game_id: str
    state: Dict[str, Any]


class GetStateResponse(BaseModel):
    state: Dict[str, Any]


class ActionRequest(BaseModel):
    game_id: str
    action: Dict[str, Any]


class ActionResponse(BaseModel):
    success: bool
    state: Dict[str, Any]
    bot_action: Optional[Dict[str, Any]] = None


@router.post("/start", response_model=StartBattleResponse)
async def start_battle(req: StartBattleRequest) -> StartBattleResponse:
    global current_bot_model
    current_bot_model = req.bot_model
    _cleanup_all_bots()

    players: List[PlayerInfo] = [
        PlayerInfo(id="human", name=req.player_name, type="human"),
    ]
    bot_names = ["Alpha", "Beta", "Gamma"]
    for i in range(req.bot_count):
        players.append(
            PlayerInfo(
                id=f"bot_{i}",
                name=bot_names[i] if i < len(bot_names) else f"Bot{i + 1}",
                type="bot",
            )
        )

    config = BattleConfig(player_count=4, players=[p.model_dump() for p in players])

    room = manager.create_room(config, seed=req.seed)
    room.human_player_id = 0

    manager.start_kyoku(room, seed=req.seed)

    for bot_id in range(1, 4):
        get_or_create_bot(bot_id).reset()

    state = manager.get_state_for_player(room, player_id=0)
    return StartBattleResponse(game_id=room.game_id, state=state)


@router.post("/close/{game_id}")
async def close_battle(game_id: str) -> Dict[str, bool]:
    """主动关闭游戏房间，释放内存"""
    manager.close_room(game_id)
    return {"success": True}


@router.post("/start_4bot")
async def start_4bot(seed: Optional[int] = None) -> Dict[str, Any]:
    """4 Bot对战模式，自动运行直到游戏结束"""
    _cleanup_all_bots()

    players: List[PlayerInfo] = [
        PlayerInfo(
            id=f"bot_{i}", name=["Alpha", "Beta", "Gamma", "Delta"][i], type="bot"
        )
        for i in range(4)
    ]

    config = BattleConfig(player_count=4, players=[p.model_dump() for p in players])
    room = manager.create_room(config, seed=seed)
    room.human_player_id = -1  # 表示无人类玩家

    for bot_id in range(4):
        get_or_create_bot(bot_id).reset()

    import asyncio

    asyncio.create_task(bot_driver.run_4bot_game(room.game_id, seed))

    state = manager.get_state_for_player(room, player_id=0)
    return {"game_id": room.game_id, "state": state}


@router.get("/state/{game_id}", response_model=GetStateResponse)
async def get_state(game_id: str, player_id: int = 0) -> GetStateResponse:
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    # 幂等摸牌：只有在人类玩家摸牌回合（非 chi/pon 后打牌）才摸牌
    if (
        room.phase == "playing"
        and room.state.actor_to_move == player_id
        and room.state.last_tsumo[player_id] is None
        and room.human_player_id == player_id
        and room.remaining_wall() >= 1
        and not room.state.last_discard  # 有弃牌时是响应弃牌阶段，不摸牌
    ):
        hand_size = sum(room.state.players[player_id].hand.values())
        if hand_size not in (11, 12):  # chi后11张，pon后12张，不摸牌
            manager.draw(room, player_id)

    state = manager.get_state_for_player(room, player_id=player_id)
    return GetStateResponse(state=state)


@router.get("/export_mjai/{game_id}")
async def export_mjai(game_id: str):
    """导出mjai格式JSONL"""
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    import json

    content = "\n".join(json.dumps(e, ensure_ascii=False) for e in room.events)
    return Response(content=content, media_type="application/jsonl")


@router.get("/export_tenhou6/{game_id}")
async def export_tenhou6(game_id: str):
    """导出tenhou6格式JSON"""
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6

    tenhou_data = convert_mjai_jsonl_to_tenhou6(room.events)
    return tenhou_data


class QuitResponse(BaseModel):
    success: bool


@router.post("/heartbeat/{game_id}")
async def heartbeat(game_id: str) -> Dict[str, Any]:
    """前端定期发送心跳，维持在线状态。"""
    import time

    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")
    room.last_heartbeat = time.time()
    room.disconnected = False
    return {"ok": True, "phase": room.phase}


@router.get("/reconnect/{game_id}")
async def reconnect(game_id: str, player_id: int = 0) -> Dict[str, Any]:
    """断线重连：返回当前完整游戏状态。"""
    import time

    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")
    room.last_heartbeat = time.time()
    room.disconnected = False
    # 若轮到人类但牌还没摸，先摸牌
    if (
        room.phase == "playing"
        and room.state.actor_to_move == player_id
        and room.human_player_id == player_id
        and room.state.last_tsumo[player_id] is None
        and not room.state.last_discard
        and room.remaining_wall() >= 1
    ):
        hand_size = sum(room.state.players[player_id].hand.values())
        if hand_size not in (11, 12):
            manager.draw(room, player_id)
    state = manager.get_state_for_player(room, player_id=player_id)
    return {"reconnected": True, "state": state}


@router.post("/quit/{game_id}", response_model=QuitResponse)
async def quit_battle(game_id: str) -> QuitResponse:
    """人类玩家主动退出，将其座位移交给 bot 接管，游戏继续（4-bot 模式）。"""
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")
    human_id = room.human_player_id
    room.human_player_id = -1  # 标记为全 bot 模式
    room.disconnected = True
    # 补充第 human_id 号 bot
    get_or_create_bot(human_id).reset()
    # 后台继续跑完对局
    if room.phase == "playing":
        asyncio.create_task(bot_driver.run_4bot_game_from_current(room))
    return QuitResponse(success=True)


@router.post("/action", response_model=ActionResponse)
async def do_action(req: ActionRequest) -> ActionResponse:
    room = manager.get_room(req.game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    action = req.action
    action_type = action.get("type")
    actor = action.get("actor", 0)

    # 委托给 BattleManager 处理游戏逻辑
    error = manager.process_human_action(room, actor, action)
    if error:
        raise HTTPException(status_code=400, detail=error)

    human_player_id = room.human_player_id
    state = manager.get_state_for_player(
        room, player_id=human_player_id if human_player_id >= 0 else 0
    )
    return ActionResponse(success=True, state=state, bot_action=None)


@router.post("/advance/{game_id}", response_model=ActionResponse)
async def advance_bot(game_id: str) -> ActionResponse:
    """推进一个 bot 步骤，前端轮询调用以依次显示每家出牌"""
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    human_player_id = room.human_player_id
    last_bot_action: Optional[Dict] = None

    async with _advance_lock:
        if room.phase != "ended":
            next_actor = room.state.actor_to_move
            if next_actor is not None and next_actor != human_player_id:
                last_bot_action = await bot_driver.take_turn(room, next_actor)

    state = manager.get_state_for_player(
        room, player_id=human_player_id if human_player_id >= 0 else 0
    )
    return ActionResponse(success=True, state=state, bot_action=last_bot_action)


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.api.battle:app",
        host="0.0.0.0",
        port=8000,
    )
