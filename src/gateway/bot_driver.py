"""Bot 驱动层：封装 bot 单步执行、4Bot 自动对战、action 验证与分发。"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional

from gateway.battle import BattleManager, BattleRoom
from mahjong_env.legal_actions import enumerate_legal_action_specs, enumerate_legal_actions
from mahjong_env.types import ActionSpec, action_dict_to_spec, action_specs_match

logger = logging.getLogger(__name__)


class BotDriver:
    def __init__(
        self,
        manager: BattleManager,
        get_or_create_bot: Callable[[int], Any],
    ):
        self.manager = manager
        self.get_or_create_bot = get_or_create_bot

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def take_turn(self, room: BattleRoom, actor: int) -> Optional[Dict]:
        """驱动单个 bot 完成一回合（供前端 /advance 端点调用）。"""
        self.manager.prepare_turn(room, actor)
        if room.phase == "ended":
            return None

        snap, event = self._build_snap_and_event(room, actor)

        logger.debug("Bot %d calling react with event: %s", actor, event)
        await asyncio.sleep(0.5)

        chosen_dict = await self._call_bot(room, actor, snap, event)
        logger.debug(
            "Bot %d chosen: type=%s pai=%s",
            actor,
            chosen_dict.get("type"),
            chosen_dict.get("pai"),
        )

        applied = self.validate_and_apply(room, actor, chosen_dict, snap)
        if not applied:
            logger.warning(
                "Bot %d invalid action %s, falling back to none",
                actor,
                chosen_dict.get("type"),
            )
            self.validate_and_apply(room, actor, {"type": "none", "actor": actor}, snap)

        return chosen_dict

    async def run_4bot_game(self, game_id: str, seed: Optional[int]) -> None:
        """后台运行 4-Bot 游戏直到结束。"""
        room = self.manager.get_room(game_id)
        if not room:
            return

        self.manager.start_kyoku(room, seed=seed)

        while True:
            while room.phase == "playing":
                actor = room.state.actor_to_move
                if actor is None:
                    break

                if room.remaining_wall() < 1:
                    self.manager.ryukyoku(room)
                    break

                snap, event = self._build_snap_and_event(room, actor, do_draw=True)

                chosen_dict = await self._call_bot(room, actor, snap, event)
                applied = self.validate_and_apply(room, actor, chosen_dict, snap)
                if not applied:
                    logger.warning(
                        "Bot %d invalid action %s in run_4bot_game, falling back to none",
                        actor,
                        chosen_dict.get("type"),
                    )
                    self.validate_and_apply(
                        room, actor, {"type": "none", "actor": actor}, snap
                    )

                await asyncio.sleep(0.1)

            if room.phase != "ended":
                break
            if self.manager.is_game_ended(room):
                room.events.append({"type": "end_game"})
                break
            if not self.manager.next_kyoku(room):
                room.events.append({"type": "end_game"})
                break

            self.manager.start_kyoku(room, seed=None)
            await asyncio.sleep(0.1)

    async def run_4bot_game_from_current(self, room: BattleRoom) -> None:
        """从当前局面继续跑完 4-Bot 游戏（人类退出后接管）。"""
        while True:
            while room.phase == "playing":
                actor = room.state.actor_to_move
                if actor is None:
                    break

                if room.remaining_wall() < 1:
                    self.manager.ryukyoku(room)
                    break

                snap, event = self._build_snap_and_event(room, actor, do_draw=True)

                chosen_dict = await self._call_bot(room, actor, snap, event)
                applied = self.validate_and_apply(room, actor, chosen_dict, snap)
                if not applied:
                    self.validate_and_apply(
                        room, actor, {"type": "none", "actor": actor}, snap
                    )

                await asyncio.sleep(0.1)

            if room.phase != "ended":
                break
            if self.manager.is_game_ended(room):
                room.events.append({"type": "end_game"})
                break
            if not self.manager.next_kyoku(room):
                room.events.append({"type": "end_game"})
                break

            self.manager.start_kyoku(room, seed=None)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_snap_and_event(
        self, room: BattleRoom, actor: int, do_draw: bool = False
    ) -> tuple:
        """构建 snap（含 shanten）和触发事件。

        Args:
            do_draw: True 时主动摸牌（run_4bot_game 路径），
                     False 时假设 prepare_turn 已完成摸牌（take_turn 路径）。
        """
        last_discard = room.state.last_discard
        is_opponent_discard = bool(last_discard and last_discard.get("actor") != actor)

        tsumo_pai = None
        if not is_opponent_discard:
            if do_draw:
                self.manager.draw(room, actor)
            tsumo_pai = room.state.last_tsumo[actor]

        snap = self.manager.get_snap_with_shanten(room, actor)
        snap["tsumo_pai"] = tsumo_pai

        if is_opponent_discard:
            event = {
                "type": "dahai",
                "actor": last_discard["actor"],
                "pai": last_discard["pai"],
            }
        else:
            event = {"type": "tsumo", "actor": actor, "pai": tsumo_pai}

        return snap, event

    async def _call_bot(
        self, room: BattleRoom, actor: int, snap: Dict, event: Dict
    ) -> Dict:
        """调用 bot.react，超时或异常时返回 fallback 动作。"""
        try:
            bot = self.get_or_create_bot(actor)
            chosen = await asyncio.wait_for(
                asyncio.to_thread(bot.react, event), timeout=5.0
            )
            if not chosen:
                chosen = {"type": "none", "actor": actor}
        except asyncio.TimeoutError:
            logger.error("Bot %d timeout after 5s", actor)
            chosen = self._fallback_action(snap, actor)
        except Exception as e:
            logger.error("Bot %d error: %s", actor, e)
            traceback.print_exc()
            chosen = self._fallback_action(snap, actor)

        return self._normalize_chosen(chosen, actor)

    def _fallback_action(self, snap: Dict, actor: int) -> Dict:
        """枚举合法动作，优先返回非 none 动作。"""
        legal_specs = enumerate_legal_action_specs(snap, actor)
        non_none = [a for a in legal_specs if a.type != "none"]
        target = non_none[0] if non_none else (legal_specs[0] if legal_specs else None)
        if target is None:
            return {"type": "none", "actor": actor}
        return {
            "type": target.type,
            "actor": actor if target.actor is None else target.actor,
            "pai": target.pai,
            "consumed": list(target.consumed) if target.consumed else None,
            "target": target.target,
        }

    def _normalize_chosen(self, chosen: Any, actor: int) -> Dict:
        """统一将 bot 返回值（dict 或对象）转为 dict，强制覆盖 actor 为当前回合玩家。"""
        if isinstance(chosen, dict):
            result = dict(chosen)
        else:
            result = {
                "type": chosen.type,
                "actor": getattr(chosen, "actor", actor),
                "pai": getattr(chosen, "pai", None),
                "consumed": getattr(chosen, "consumed", None),
                "target": getattr(chosen, "target", None),
            }
        # 强制 actor 为当前回合玩家，防止 bot 返回错误的 actor
        result["actor"] = actor
        return result

    def validate_and_apply(
        self,
        room: BattleRoom,
        actor: int,
        action: Dict,
        snap: Dict,
    ) -> bool:
        """验证动作合法性并分发到对应的 BattleManager 方法。返回 True 表示成功。"""
        mgr = self.manager
        requested_spec = action_dict_to_spec(action, actor_hint=actor)
        legal_specs: List[ActionSpec] = enumerate_legal_action_specs(snap, actor)
        legal_by_type: Dict[str, List[ActionSpec]] = {}
        for spec in legal_specs:
            legal_by_type.setdefault(spec.type, []).append(spec)
        action_type = requested_spec.type

        if action_type == "dahai":
            pai = requested_spec.pai or ""
            legal_dahai = legal_by_type.get("dahai", [])
            matched = next((a for a in legal_dahai if a.pai == pai), None) or (
                legal_dahai[0] if legal_dahai else None
            )
            if matched:
                mgr.discard(room, actor, matched.pai, tsumogiri=matched.tsumogiri)
                return True

        elif action_type == "reach":
            pai = requested_spec.pai or ""
            tsumo_pai = snap.get("tsumo_pai")
            shanten = snap.get("shanten", 8)
            if shanten == 0 and tsumo_pai and legal_by_type.get("reach"):
                mgr.reach(room, actor)
                mgr.discard(room, actor, pai or tsumo_pai, tsumogiri=not bool(pai))
                return True

        elif action_type == "chi":
            target = requested_spec.target
            legal_chi = legal_by_type.get("chi", [])
            matched = next(
                (a for a in legal_chi if action_specs_match(a, requested_spec)), None
            ) or (legal_chi[0] if legal_chi else None)
            if matched:
                mgr.handle_meld(
                    room,
                    "chi",
                    actor,
                    matched.pai or "",
                    list(matched.consumed),
                    target=target,
                )
                return True

        elif action_type in ("pon", "daiminkan"):
            legal_same = legal_by_type.get(action_type, [])
            matched = next((a for a in legal_same if action_specs_match(a, requested_spec)), None)
            if matched:
                mgr.handle_meld(
                    room,
                    action_type,
                    actor,
                    matched.pai or "",
                    list(matched.consumed),
                    target=matched.target,
                )
                return True

        elif action_type == "ankan":
            legal_ankan = legal_by_type.get("ankan", [])
            matched = next((a for a in legal_ankan if action_specs_match(a, requested_spec)), None)
            if matched:
                mgr.handle_meld(
                    room,
                    "ankan",
                    actor,
                    matched.pai or "",
                    list(matched.consumed) or [matched.pai] * 4,
                )
                return True

        elif action_type == "kakan":
            legal_kakan = legal_by_type.get("kakan", [])
            matched = next((a for a in legal_kakan if action_specs_match(a, requested_spec)), None)
            if matched:
                mgr.handle_meld(
                    room,
                    "kakan",
                    actor,
                    matched.pai or "",
                    list(matched.consumed),
                )
                return True

        elif action_type == "hora":
            legal_hora = legal_by_type.get("hora", [])
            matched = next(
                (
                    a
                    for a in legal_hora
                    if a.target == requested_spec.target
                    and (requested_spec.pai is None or a.pai == requested_spec.pai)
                ),
                None,
            )
            if matched:
                target = actor if matched.target is None else matched.target
                mgr.hora(room, actor, target, matched.pai or "", is_tsumo=(target == actor))
                return True

        elif action_type == "none":
            # 委托给 apply_action 统一处理（含响应弃牌的 actor_to_move 推进）
            mgr.apply_action(room, actor, action)
            return True

        return False
