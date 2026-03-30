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


def get_or_create_bot(bot_id: int) -> Any:
    if bot_id not in bots:
        if BOT_TYPE == "keqingv1":
            from keqingv1.bot import KeqingBot
            import keqingv1

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(keqingv1.__file__))
            )
            model_path = os.path.join(
                project_root, "artifacts", "models", "keqingv1", "best.pth"
            )
            print(f"[Bot] Loading keqingv1 model from: {model_path}")
            bots[bot_id] = KeqingBot(
                player_id=bot_id, model_path=model_path, device="cpu", verbose=False
            )
        else:
            from v5model.bot import V5Bot
            import v5model

            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(v5model.__file__))
            )
            model_path = os.path.join(
                project_root, "artifacts", "models", "modelv5", "best.pth"
            )
            print(f"[Bot] Loading v5 model from: {model_path}")
            bots[bot_id] = V5Bot(
                player_id=bot_id, model_path=model_path, device="cpu", verbose=False
            )
    return bots[bot_id]


class PlayerInfo(BaseModel):
    id: str
    name: str
    type: str


class StartBattleRequest(BaseModel):
    player_name: str = "Player"
    bot_count: int = 3
    seed: Optional[int] = None


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

    asyncio.create_task(_run_4bot_game(room.game_id, seed))

    state = manager.get_state_for_player(room, player_id=0)
    return {"game_id": room.game_id, "state": state}


async def _run_4bot_game(game_id: str, seed: Optional[int]) -> None:
    """后台运行4 Bot游戏直到结束"""
    room = manager.get_room(game_id)
    if not room:
        return

    manager.start_kyoku(room, seed=seed)

    while True:
        while room.phase == "playing":
            actor = room.state.actor_to_move
            if actor is None:
                break

            if room.remaining_wall() < 1:
                manager.ryukyoku(room)
                break

            last_discard = room.state.last_discard
            is_opponent_discard = last_discard and last_discard.get("actor") != actor

            tsumo_pai = None
            if not is_opponent_discard:
                manager.draw(room, actor)
                tsumo_pai = room.state.last_tsumo[actor]

            snap = room.state.snapshot(actor)
            from mahjong_env.replay import _calc_shanten_waits

            hand_list = snap.get("hand", [])
            melds_list = (snap.get("melds") or [[], [], [], []])[actor]
            try:
                shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(
                    hand_list, melds_list
                )
            except Exception as e:
                print(f"[WARN] Bot {actor} shanten calc failed: {e}")
                shanten, waits_cnt, waits_tiles = 8, 0, [False] * 34
            snap["shanten"] = shanten
            snap["waits_count"] = waits_cnt
            snap["waits_tiles"] = waits_tiles
            snap["tsumo_pai"] = tsumo_pai

            if is_opponent_discard:
                event = {
                    "type": "dahai",
                    "actor": last_discard["actor"],
                    "pai": last_discard["pai"],
                }
            else:
                event = {"type": "tsumo", "actor": actor, "pai": tsumo_pai}

            try:
                bot = get_or_create_bot(actor)
                chosen = await asyncio.wait_for(
                    asyncio.to_thread(bot.react, event), timeout=5.0
                )
            except asyncio.TimeoutError:
                print(f"[ERROR] Bot {actor} timeout in _run_4bot_game")
                chosen = None
            except Exception as e:
                print(f"[ERROR] Bot {actor} exception: {e}")
                chosen = None

            if not chosen:
                chosen = {"type": "none", "actor": actor}

            action_type = chosen.get("type")

            legal = enumerate_legal_actions(snap, actor)
            legal_dict = {a.type: a for a in legal}

            if action_type == "dahai":
                pai = chosen.get("pai", "")
                tsumogiri = chosen.get("tsumogiri", False)
                legal_action = legal_dict.get("dahai")
                if legal_action and pai == legal_action.pai:
                    manager.discard(room, actor, pai, tsumogiri=tsumogiri)
                else:
                    print(f"[WARN] Bot {actor} invalid dahai: {pai}, fallback to none")
            elif action_type in ("pon", "chi", "daiminkan"):
                pai = chosen.get("pai", "")
                consumed = chosen.get("consumed", [])
                target = chosen.get("target")
                legal_action = legal_dict.get(action_type)
                if legal_action and pai == legal_action.pai:
                    manager.handle_meld(
                        room, action_type, actor, pai, consumed, target=target
                    )
                else:
                    print(f"[WARN] Bot {actor} invalid {action_type}: {pai}")
            elif action_type == "ankan":
                pai = chosen.get("pai", "")
                consumed = chosen.get("consumed", [pai] * 4)
                legal_action = legal_dict.get("ankan")
                if legal_action:
                    manager.handle_meld(room, "ankan", actor, pai, consumed)
                else:
                    print(f"[WARN] Bot {actor} invalid ankan: {pai}")
            elif action_type == "kakan":
                pai = chosen.get("pai", "")
                consumed = chosen.get("consumed", [])
                legal_action = legal_dict.get("kakan")
                if legal_action and pai == legal_action.pai:
                    manager.handle_meld(room, "kakan", actor, pai, consumed)
                else:
                    print(f"[WARN] Bot {actor} invalid kakan: {pai}")
            elif action_type == "reach":
                if shanten == 0 and tsumo_pai:
                    manager.reach(room, actor)
                    manager.discard(room, actor, tsumo_pai, tsumogiri=True)
                else:
                    print(f"[WARN] Bot {actor} invalid reach: shanten={shanten}")
            elif action_type == "hora":
                pai = chosen.get("pai", "")
                target = chosen.get("target", actor)
                is_tsumo = target == actor
                legal_action = legal_dict.get("hora")
                if legal_action and pai == legal_action.pai:
                    manager.hora(room, actor, target, pai, is_tsumo=is_tsumo)
                else:
                    print(f"[WARN] Bot {actor} invalid hora: {pai}")
            elif action_type == "none":
                if tsumo_pai and not is_opponent_discard:
                    manager.discard(room, actor, tsumo_pai, tsumogiri=True)

            await asyncio.sleep(0.1)

        if room.phase != "ended":
            break

        if manager.is_game_ended(room):
            room.events.append({"type": "end_game"})
            break

        if not manager.next_kyoku(room):
            room.events.append({"type": "end_game"})
            break

        manager.start_kyoku(room, seed=None)
        await asyncio.sleep(0.1)


@router.get("/state/{game_id}", response_model=GetStateResponse)
async def get_state(game_id: str, player_id: int = 0) -> GetStateResponse:
    room = manager.get_room(game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    # 幂等摸牌：只有在人类玩家回合 && last_tsumo 为空 时才摸牌，避免重复摸
    if (
        room.phase == "playing"
        and room.state.actor_to_move == player_id
        and room.state.last_tsumo[player_id] is None
        and room.human_player_id == player_id
        and room.remaining_wall() >= 1
    ):
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


async def _bot_take_turn(room: BattleRoom, actor: int) -> Optional[Dict]:
    """驱动单个 bot 完成一回合（摸牌或响应弃牌），返回 bot 选择的动作。"""
    from mahjong_env.replay import _calc_shanten_waits

    last_discard = room.state.last_discard
    # 只有当 last_discard 存在、且弃牌者不是自己、且自己不是弃牌者的下家（下家是摸牌者）时，才视为响应弃牌
    if last_discard:
        discarder = last_discard.get("actor")
        next_to_draw = (discarder + 1) % 4 if discarder is not None else None
        is_opponent_discard = discarder != actor and actor != next_to_draw
    else:
        is_opponent_discard = False

    tsumo_pai = None
    if not is_opponent_discard:
        if room.pending_rinshan:
            # 大明杠/暗杠/加杠后需要摸岭上牌
            if room.remaining_wall() < 1:
                manager.ryukyoku(room)
                return None
            manager.draw(room, actor)
            tsumo_pai = room.state.last_tsumo[actor]
        elif room.state.last_tsumo[actor] is None:
            # 普通摸牌回合（碰/吃后需要打牌，last_tsumo 也为 None，但手牌不足 14 张）
            hand_size = sum(room.state.players[actor].hand.values())
            if hand_size >= 14:
                # 已有摸牌（不应出现，保险处理）
                tsumo_pai = room.state.last_tsumo[actor]
            elif hand_size in (11, 12):  # 碰/吃后 11/12 张，直接打牌不摸牌
                tsumo_pai = None
            else:
                if room.remaining_wall() < 1:
                    manager.ryukyoku(room)
                    return None
                manager.draw(room, actor)
                tsumo_pai = room.state.last_tsumo[actor]
        else:
            tsumo_pai = room.state.last_tsumo[actor]

    snap = room.state.snapshot(actor)
    hand_list = snap.get("hand", [])
    melds_list = (snap.get("melds") or [[], [], [], []])[actor]
    try:
        shanten, waits_cnt, waits_tiles, _ = _calc_shanten_waits(hand_list, melds_list)
    except Exception as e:
        print(f"[WARN] Bot {actor} shanten calc failed: {e}")
        shanten, waits_cnt, waits_tiles = 8, 0, [False] * 34
    snap["shanten"] = shanten
    snap["waits_count"] = waits_cnt
    snap["waits_tiles"] = waits_tiles
    snap["tsumo_pai"] = tsumo_pai

    if is_opponent_discard:
        event = {
            "type": "dahai",
            "actor": last_discard["actor"],
            "pai": last_discard["pai"],
        }
    else:
        event = {"type": "tsumo", "actor": actor, "pai": tsumo_pai}

    print(f"[DEBUG] Bot {actor} calling react with event: {event}")

    try:
        bot = get_or_create_bot(actor)
        chosen = await asyncio.wait_for(
            asyncio.to_thread(bot.react, event), timeout=5.0
        )
        print(f"[DEBUG] Bot {actor} react returned: {chosen}")
        if not chosen:
            chosen = {"type": "none", "actor": actor}
    except asyncio.TimeoutError:
        print(f"[ERROR] Bot {actor} timeout after 5s")
        legal = enumerate_legal_actions(snap, actor)
        non_none = [a for a in legal if a.type != "none"]
        chosen = (
            (non_none[0] if non_none else legal[0])
            if legal
            else {"type": "none", "actor": actor}
        )
    except Exception as e:
        print(f"[ERROR] Bot {actor} error: {e}")
        import traceback

        traceback.print_exc()
        legal = enumerate_legal_actions(snap, actor)
        non_none = [a for a in legal if a.type != "none"]
        chosen = (
            (non_none[0] if non_none else legal[0])
            if legal
            else {"type": "none", "actor": actor}
        )

    action_type = chosen.get("type") if isinstance(chosen, dict) else chosen.type
    chosen_dict = (
        chosen
        if isinstance(chosen, dict)
        else {
            "type": chosen.type,
            "actor": chosen.actor,
            "pai": chosen.pai,
            "consumed": chosen.consumed,
            "target": chosen.target,
        }
    )

    legal_snap = room.state.snapshot(actor)
    legal_all = enumerate_legal_actions(legal_snap, actor)
    legal_dict = {a.type: a for a in legal_all}
    legal_dahai_list = [a for a in legal_all if a.type == "dahai"]

    def _norm(t: str) -> str:
        return {"5mr": "5m", "5pr": "5p", "5sr": "5s"}.get(t, t)

    print(
        f"[DEBUG] Bot {actor} chosen action: type={action_type}, pai={chosen_dict.get('pai')}"
    )

    if action_type == "dahai":
        pai = chosen_dict.get("pai", "")
        matched = next(
            (a for a in legal_dahai_list if a.pai == pai or _norm(a.pai) == _norm(pai)),
            None,
        )
        if matched:
            manager.discard(room, actor, matched.pai, tsumogiri=matched.tsumogiri)
        elif legal_dahai_list:
            print(
                f"[WARN] Bot {actor} invalid dahai: {pai}, falling back to {legal_dahai_list[0].pai}"
            )
            manager.discard(
                room,
                actor,
                legal_dahai_list[0].pai,
                tsumogiri=legal_dahai_list[0].tsumogiri,
            )
        else:
            print(f"[WARN] Bot {actor} no legal dahai found")
    elif action_type in ("pon", "chi", "daiminkan"):
        pai = chosen_dict.get("pai", "")
        consumed = chosen_dict.get("consumed", [])
        target = chosen_dict.get("target")
        legal_action = legal_dict.get(action_type)
        if legal_action:
            manager.handle_meld(room, action_type, actor, pai, consumed, target=target)
        else:
            print(f"[WARN] Bot {actor} invalid {action_type}: {pai}")
    elif action_type == "ankan":
        pai = chosen_dict.get("pai", "")
        consumed = chosen_dict.get("consumed", [pai] * 4)
        if legal_dict.get("ankan"):
            manager.handle_meld(room, "ankan", actor, pai, consumed)
        else:
            print(f"[WARN] Bot {actor} invalid ankan: {pai}")
    elif action_type == "kakan":
        pai = chosen_dict.get("pai", "")
        consumed = chosen_dict.get("consumed", [])
        legal_action = legal_dict.get("kakan")
        if legal_action:
            manager.handle_meld(room, "kakan", actor, pai, consumed)
        else:
            print(f"[WARN] Bot {actor} invalid kakan: {pai}")
    elif action_type == "reach":
        if shanten == 0 and tsumo_pai:
            manager.reach(room, actor)
            manager.discard(room, actor, tsumo_pai, tsumogiri=True)
        else:
            print(f"[WARN] Bot {actor} invalid reach: shanten={shanten}")
    elif action_type == "hora":
        pai = chosen_dict.get("pai", "")
        target = chosen_dict.get("target", actor)
        is_tsumo = target == actor
        legal_action = legal_dict.get("hora")
        if legal_action:
            manager.hora(room, actor, target, pai, is_tsumo=is_tsumo)
        else:
            print(f"[WARN] Bot {actor} invalid hora: {pai}")
    elif action_type == "none":
        if tsumo_pai and not is_opponent_discard:
            # 摸牌回合返回 none，强制摸切
            manager.discard(room, actor, tsumo_pai, tsumogiri=True)
        elif is_opponent_discard:
            # 响应他家弃牌选择 pass
            discarder = last_discard["actor"] if last_discard else None
            if discarder is not None:
                next_actor = (actor + 1) % 4
                if next_actor == discarder:
                    # 已经轮完一圈，所有人 pass，弃牌者下家摸牌
                    room.state.last_discard = None
                    room.state.actor_to_move = (discarder + 1) % 4
                else:
                    # 继续让下一家响应
                    room.state.actor_to_move = next_actor

    return chosen_dict


@router.post("/action", response_model=ActionResponse)
async def do_action(req: ActionRequest) -> ActionResponse:
    room = manager.get_room(req.game_id)
    if not room:
        raise HTTPException(status_code=404, detail="Game not found")

    action = req.action
    action_type = action.get("type")
    actor = action.get("actor", 0)

    print(
        f"[DEBUG] do_action called: actor={actor}, type={action_type}, current_actor_to_move={room.state.actor_to_move}, phase={room.phase}"
    )

    if room.state.actor_to_move != actor:
        raise HTTPException(status_code=400, detail="Not your turn")

    # 如果是 actor 的摸牌回合（非响应他家弃牌），根据场景决定是否摸牌
    last_discard = room.state.last_discard
    is_response_to_discard = last_discard and last_discard.get("actor") != actor
    if not is_response_to_discard and room.state.last_tsumo[actor] is None:
        hand_size = sum(room.state.players[actor].hand.values())
        need_draw = False
        if room.pending_rinshan:
            need_draw = True  # 杠后摸岭上牌
        elif hand_size not in (11, 12):  # 碰/吃后为 11/12 张，不需摸牌
            need_draw = True
        if need_draw:
            if room.remaining_wall() < 1:
                manager.ryukyoku(room)
                state = manager.get_state_for_player(room, player_id=actor)
                return ActionResponse(success=True, state=state, bot_action=None)
            manager.draw(room, actor)
            print(
                f"[DEBUG] Drew tile for actor {actor}, new hand_size={sum(room.state.players[actor].hand.values())}"
            )

    snap = room.state.snapshot(actor)
    legal = enumerate_legal_actions(snap, actor)
    legal_types = {
        (a.type, a.pai, tuple(a.consumed) if a.consumed else None, a.target)
        for a in legal
    }

    action_key = (
        action_type,
        action.get("pai"),
        tuple(action.get("consumed", [])) if action.get("consumed") else None,
        action.get("target"),
    )

    if action_type not in ("none", "reach") and action_key not in legal_types:
        raise HTTPException(status_code=400, detail="Illegal action")

    # 应用人类玩家的动作
    if action_type == "dahai":
        pai = action.get("pai", "")
        tsumogiri = action.get("tsumogiri", False)
        manager.discard(room, actor, pai, tsumogiri=tsumogiri)

    elif action_type in ("pon", "chi", "daiminkan"):
        pai = action.get("pai", "")
        consumed = action.get("consumed", [])
        target = action.get("target")
        manager.handle_meld(room, action_type, actor, pai, consumed, target=target)

    elif action_type == "ankan":
        pai = action.get("pai", "")
        consumed = action.get("consumed", [pai] * 4)
        manager.handle_meld(room, "ankan", actor, pai, consumed)

    elif action_type == "kakan":
        pai = action.get("pai", "")
        consumed = action.get("consumed", [])
        manager.handle_meld(room, "kakan", actor, pai, consumed)

    elif action_type == "reach":
        # reach 只设置立直标志，前端需要再发一次 dahai 宣言牌
        manager.reach(room, actor)

    elif action_type == "hora":
        pai = action.get("pai", "")
        target = action.get("target", actor)
        is_tsumo = target == actor
        manager.hora(room, actor, target, pai, is_tsumo=is_tsumo)

    elif action_type == "none":
        pass

    # 游戏已结束，直接返回
    if room.phase == "ended":
        state = manager.get_state_for_player(room, player_id=actor)
        return ActionResponse(success=True, state=state, bot_action=None)

    # 驱动各 bot 依次行动，直到轮到人类玩家或游戏结束
    last_bot_action: Optional[Dict] = None
    human_player_id = room.human_player_id  # 人类玩家 ID（-1 表示无人类）
    MAX_BOT_STEPS = 200  # 防止死循环（一局约60-80步）
    for _ in range(MAX_BOT_STEPS):
        if room.phase == "ended":
            break
        next_actor = room.state.actor_to_move
        if next_actor is None:
            break
        if next_actor == human_player_id:
            break
        last_bot_action = await _bot_take_turn(room, next_actor)
        if room.phase == "ended":
            break

    state = manager.get_state_for_player(
        room, player_id=human_player_id if human_player_id >= 0 else 0
    )
    return ActionResponse(success=True, state=state, bot_action=last_bot_action)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway.api.battle:app",
        host="0.0.0.0",
        port=8000,
    )
