from __future__ import annotations

from typing import Any, Optional

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import apply_event

from inference.contracts import DecisionContext


class DefaultDecisionContextBuilder:
    def __init__(
        self,
        *,
        model_version: str,
        riichi_state,
        inject_shanten_waits,
        enumerate_legal_actions_fn=enumerate_legal_actions,
    ):
        self.model_version = model_version
        self.riichi_state = riichi_state
        self._inject_shanten_waits = inject_shanten_waits
        self._enumerate_legal_actions = enumerate_legal_actions_fn

    def build(self, state, actor: int, event: dict[str, Any]) -> Optional[DecisionContext]:
        etype = event.get("type", "")

        decision_base_snap: Optional[dict[str, Any]] = None
        pre_apply_hand: Optional[list] = None
        pre_apply_melds: Optional[list] = None

        if etype == "tsumo" and event.get("actor") == actor:
            decision_base_snap = state.snapshot(actor)
            pre_apply_hand = decision_base_snap.get("hand", [])
            pre_apply_melds = (
                (decision_base_snap.get("melds") or [[], [], [], []])[actor]
            )
            apply_event(state, event)
        elif etype == "dahai" and event.get("actor") != actor:
            pre_snap = state.snapshot(actor)
            pre_apply_hand = pre_snap.get("hand", [])
            pre_apply_melds = (pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
        elif etype == "kakan" and event.get("actor") != actor:
            pre_snap = state.snapshot(actor)
            pre_apply_hand = pre_snap.get("hand", [])
            pre_apply_melds = (pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
        elif etype in ("chi", "pon", "daiminkan") and event.get("actor") == actor:
            pre_snap = state.snapshot(actor)
            pre_apply_hand = pre_snap.get("hand", [])
            pre_apply_melds = (pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
        elif etype == "reach" and event.get("actor") == actor:
            pre_snap = state.snapshot(actor)
            pre_apply_hand = pre_snap.get("hand", [])
            pre_apply_melds = (pre_snap.get("melds") or [[], [], [], []])[actor]
            apply_event(state, event)
        else:
            apply_event(state, event)
            return None

        runtime_snap = state.snapshot(actor)
        injected = False
        if (
            self.riichi_state is not None
            and etype == "tsumo"
            and event.get("actor") == actor
            and decision_base_snap is None
            and self.model_version != "keqingv3"
        ):
            try:
                runtime_snap["shanten"] = int(self.riichi_state.shanten)
                runtime_snap["waits_count"] = int(sum(self.riichi_state.waits))
                runtime_snap["waits_tiles"] = list(self.riichi_state.waits)
                injected = True
            except Exception:
                pass
        if not injected:
            if pre_apply_hand is not None:
                if etype == "tsumo" and event.get("actor") == actor:
                    hand_list = runtime_snap.get("hand", [])
                    melds_list = (runtime_snap.get("melds") or [[], [], [], []])[actor]
                else:
                    hand_list = pre_apply_hand
                    melds_list = pre_apply_melds or []
            else:
                hand_list = runtime_snap.get("hand", [])
                melds_list = (runtime_snap.get("melds") or [[], [], [], []])[actor]
            self._inject_shanten_waits(
                runtime_snap,
                hand_list=hand_list,
                melds_list=melds_list,
                model_version=self.model_version,
            )

        self._inject_furiten(runtime_snap, state, actor)
        legal_actions = [a.to_mjai() for a in self._enumerate_legal_actions(runtime_snap, actor)]

        model_snap = runtime_snap
        if decision_base_snap is not None:
            model_snap = decision_base_snap
            self._inject_shanten_waits(
                model_snap,
                hand_list=pre_apply_hand or [],
                melds_list=pre_apply_melds or [],
                model_version=self.model_version,
            )
            model_snap["tsumo_pai"] = event.get("pai")
        else:
            model_snap["tsumo_pai"] = event.get("pai") if etype == "tsumo" else None

        return DecisionContext(
            actor=actor,
            event=event,
            runtime_snap=runtime_snap,
            model_snap=model_snap,
            legal_actions=legal_actions,
        )

    @staticmethod
    def _inject_furiten(runtime_snap: dict[str, Any], state, actor: int) -> None:
        waits_tiles = runtime_snap.get("waits_tiles")
        p = state.players[actor]
        if waits_tiles is not None:
            from mahjong_env.tiles import normalize_tile as _norm
            from mahjong_env.tiles import tile_to_34 as _t34

            wait_set = {i for i, w in enumerate(waits_tiles) if w}
            actor_disc_set = {
                _t34(_norm(d["pai"] if isinstance(d, dict) else d))
                for d in state.players[actor].discards
            }
            p.sutehai_furiten = bool(wait_set & actor_disc_set)
            if p.reached and state.last_tsumo[actor] is None:
                last_disc = p.discards[-1]["pai"] if p.discards else None
                if last_disc and _t34(_norm(last_disc)) in wait_set:
                    p.riichi_furiten = True
            p.furiten = p.sutehai_furiten or p.riichi_furiten or p.doujun_furiten
        runtime_snap["furiten"] = [pp.furiten for pp in state.players]
        runtime_snap["sutehai_furiten"] = [pp.sutehai_furiten for pp in state.players]
        runtime_snap["riichi_furiten"] = [pp.riichi_furiten for pp in state.players]
        runtime_snap["doujun_furiten"] = [pp.doujun_furiten for pp in state.players]
