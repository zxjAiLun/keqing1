#!/usr/bin/env python3
"""自对战脚本：加载 KeqingBot 跑 N 局全Bot对战，统计顺位/胜率，可选保存牌谱/.npz。

用法:
    python scripts/selfplay.py --model best.pth --games 100
    python scripts/selfplay.py --model best.pth --games 1000 --save-games 10 --output-dir selfplay_out
    python scripts/selfplay.py --model best.pth --games 100 --save-all-games --save-npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gateway.battle import BattleConfig, BattleManager, BattleRoom, _shuffle_wall
from keqingv1.bot import KeqingBot
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState


# ---------------------------------------------------------------------------
# Bot 决策辅助
# ---------------------------------------------------------------------------

def _bot_decide(bot: KeqingBot, event: dict) -> Optional[dict]:
    """将单条 mjai 事件喂给 bot，返回 bot 的响应动作（或 None）。"""
    return bot.react(event)


def _apply_bot_action(
    manager: BattleManager,
    room: BattleRoom,
    actor: int,
    action: dict,
) -> Optional[dict]:
    """将 bot 的决策动作应用到 room，返回需要广播给其他 bot 的事件（或 None）。

    返回 dict 代表产生了一个新事件（dahai/reach/meld），None 代表动作无效。
    """
    atype = action.get("type", "none")

    if atype == "none":
        return None

    if atype == "dahai":
        pai = action.get("pai", "")
        tsumogiri = action.get("tsumogiri", False)
        ok = manager.discard(room, actor, pai, tsumogiri=tsumogiri)
        if not ok:
            # 回退：打第一张手牌
            hand = list(room.state.players[actor].hand.keys())
            if hand:
                manager.discard(room, actor, hand[0])
        return room.events[-1] if room.events else None

    if atype == "reach":
        manager.reach(room, actor)
        return room.events[-1] if room.events else None

    if atype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
        consumed = action.get("consumed", [])
        pai = action.get("pai", "")
        target = action.get("target", None)
        ok = manager.handle_meld(room, actor, atype, pai, consumed, target=target)
        if not ok:
            return None
        return room.events[-1] if room.events else None

    if atype == "hora":
        target = action.get("target", actor)
        pai = action.get("pai", "")
        is_tsumo = (target == actor)
        manager.hora(room, actor, target, pai, is_tsumo=is_tsumo)
        return None

    if atype == "ryukyoku":
        manager.ryukyoku(room)
        return None

    return None


# ---------------------------------------------------------------------------
# 单局运行
# ---------------------------------------------------------------------------

MAX_TURNS = 300  # 防死循环


def run_one_kyoku(
    manager: BattleManager,
    room: BattleRoom,
    bots: List[KeqingBot],
    seed: Optional[int] = None,
) -> dict:
    """跑完一局（从 start_kyoku 到 hora/ryukyoku），返回结果 dict。"""
    manager.start_kyoku(room, seed=seed)

    # 重置所有 bot 状态，replay start_kyoku 事件
    for bot in bots:
        bot.game_state = GameState()
        bot.decision_log = []

    # 广播 start_game / start_kyoku 给所有 bot
    for ev in room.events:
        for bot in bots:
            bot.react(ev)

    oya = room.state.oya
    actor = oya
    turns = 0

    while room.phase == "playing" and turns < MAX_TURNS:
        turns += 1

        # 摸牌
        if room.pending_rinshan:
            tile = room.draw_rinshan() if hasattr(room, 'draw_rinshan') else room.draw_tile()
            if tile:
                room.state.players[actor].hand[tile] += 1
                import collections
                room.state.last_tsumo[actor] = tile
                room.state.last_tsumo_raw[actor] = tile
                room.events.append({"type": "tsumo", "actor": actor, "pai": tile})
            room.pending_rinshan = False
        else:
            tile = manager.draw(room, actor)

        if tile is None:
            # 牌山摸完，流局
            tenpai = []
            for pid in range(4):
                snap = room.state.snapshot(pid)
                from mahjong_env.replay import _calc_shanten_waits
                hand_list = snap.get("hand", [])
                melds_list = (snap.get("melds") or [[], [], [], []])[pid]
                shanten, _, _, _ = _calc_shanten_waits(hand_list, melds_list)
                if shanten <= 0:
                    tenpai.append(pid)
            manager.ryukyoku(room, tenpai=tenpai)
            break

        tsumo_ev = room.events[-1]

        # actor bot 响应摸牌
        snap = room.state.snapshot(actor)
        action = bots[actor].react(tsumo_ev)

        # 其他 bot 也收到 tsumo 事件（但不能响应）
        for pid in range(4):
            if pid != actor:
                bots[pid].react(tsumo_ev)

        if action is None:
            action = {"type": "dahai", "actor": actor, "pai": tile}

        atype = action.get("type", "none")

        # hora（自摸）
        if atype == "hora":
            manager.hora(room, actor, actor, tile, is_tsumo=True)
            hora_ev = room.events[-1]
            for bot in bots:
                bot.react(hora_ev)
            break

        # ryukyoku（九种九牌）
        if atype == "ryukyoku":
            manager.ryukyoku(room)
            break

        # reach
        if atype == "reach":
            manager.reach(room, actor)
            reach_ev = room.events[-1]
            for bot in bots:
                bot.react(reach_ev)
            # reach 后继续打牌
            action = bots[actor].react(reach_ev)
            if action is None or action.get("type") != "dahai":
                hand = list(room.state.players[actor].hand.keys())
                action = {"type": "dahai", "actor": actor, "pai": hand[0] if hand else tile}

        # dahai（含 reach 后打牌）
        if action.get("type") == "dahai":
            pai = action.get("pai", tile)
            tsumogiri = action.get("tsumogiri", False)
            ok = manager.discard(room, actor, pai, tsumogiri=tsumogiri)
            if not ok:
                hand = list(room.state.players[actor].hand.keys())
                if hand:
                    manager.discard(room, actor, hand[0])
            dahai_ev = room.events[-1]

            # 广播 dahai 给所有 bot
            for bot in bots:
                bot.react(dahai_ev)

            # 其他玩家响应（副露/荣和）
            responded = False
            for offset in range(1, 4):
                responder_id = (actor + offset) % 4
                resp = bots[responder_id].react(dahai_ev)
                if resp is None:
                    continue
                rtype = resp.get("type", "none")
                if rtype == "none":
                    continue

                if rtype == "hora":
                    discard_pai = dahai_ev.get("pai", "")
                    manager.hora(room, responder_id, actor, discard_pai, is_tsumo=False)
                    hora_ev = room.events[-1]
                    for bot in bots:
                        bot.react(hora_ev)
                    responded = True
                    break

                if rtype in ("chi", "pon", "daiminkan", "ankan", "kakan"):
                    consumed = resp.get("consumed", [])
                    resp_pai = resp.get("pai", "")
                    target = resp.get("target", actor)
                    ok = manager.handle_meld(room, responder_id, rtype, resp_pai, consumed, target=target)
                    if not ok:
                        continue
                    meld_ev = room.events[-1]
                    for bot in bots:
                        bot.react(meld_ev)

                    # 副露后打牌
                    meld_action = bots[responder_id].react(meld_ev)
                    if meld_action is None or meld_action.get("type") != "dahai":
                        hand = list(room.state.players[responder_id].hand.keys())
                        meld_action = {"type": "dahai", "actor": responder_id, "pai": hand[0] if hand else ""}
                    m_pai = meld_action.get("pai", "")
                    manager.discard(room, responder_id, m_pai)
                    m_dahai_ev = room.events[-1]
                    for bot in bots:
                        bot.react(m_dahai_ev)
                    actor = responder_id
                    responded = True
                    break

            if room.phase == "ended":
                break
            if not responded:
                actor = (actor + 1) % 4

    return {
        "scores": room.state.scores[:],
        "events": list(room.events),
        "turns": turns,
        "phase": room.phase,
    }


# ---------------------------------------------------------------------------
# NPZ / 牌谱 保存
# ---------------------------------------------------------------------------

def _events_to_npz(events: list) -> Optional[Tuple]:
    """从 events 生成训练样本，返回 (tile_feat, scalar, mask, action_idx, value) 或 None。"""
    try:
        from mahjong_env.replay import build_supervised_samples
        from keqingv1.action_space import action_to_idx, build_legal_mask
        from keqingv1.features import encode
        samples = build_supervised_samples(events)
    except Exception as e:
        return None

    rows_tile, rows_scalar, rows_mask, rows_action, rows_value = [], [], [], [], []
    for s in samples:
        try:
            tile_feat, scalar = encode(s.state, s.actor)
            mask = np.array(build_legal_mask(s.legal_actions), dtype=np.float32)
            action_idx = action_to_idx(s.label_action)
            if mask[action_idx] == 0:
                continue
            value = float(np.clip(s.value_target, -1.0, 1.0))
            rows_tile.append(tile_feat)
            rows_scalar.append(scalar)
            rows_mask.append(mask)
            rows_action.append(action_idx)
            rows_value.append(value)
        except Exception:
            continue
    if not rows_tile:
        return None
    return (
        np.stack(rows_tile).astype(np.float16),
        np.stack(rows_scalar).astype(np.float16),
        np.stack(rows_mask).astype(np.uint8),
        np.array(rows_action, dtype=np.int16),
        np.array(rows_value, dtype=np.float16),
    )


def _save_npz(path: Path, arrays: Tuple) -> None:
    tile_feat, scalar, mask, action_idx, value = arrays
    np.savez_compressed(
        path,
        tile_feat=tile_feat,
        scalar=scalar,
        mask=mask,
        action_idx=action_idx,
        value=value,
    )


def _save_mjson(path: Path, events: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 多局统计
# ---------------------------------------------------------------------------

def _rank_of_scores(scores: List[int]) -> List[int]:
    """返回每个 player 的顺位（1-4），同点同顺位。"""
    sorted_scores = sorted(enumerate(scores), key=lambda x: -x[1])
    ranks = [0] * 4
    rank = 1
    for i, (pid, sc) in enumerate(sorted_scores):
        if i > 0 and sc < sorted_scores[i - 1][1]:
            rank = i + 1
        ranks[pid] = rank
    return ranks


def _score_interest(events: list, scores: List[int]) -> float:
    """为牌谱评分，用于筛选'值得保存'的局。分越高越值得看。"""
    # 指标：有荣和/自摸事件 + 最终得分差距大
    hora_count = sum(1 for e in events if e.get("type") == "hora")
    score_spread = max(scores) - min(scores)
    return hora_count * 1000 + score_spread


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="KeqingBot 全Bot自对战")
    p.add_argument("--model", required=True, help="模型权重路径 (.pth)")
    p.add_argument("--games", type=int, default=100, help="总局数")
    p.add_argument("--save-games", type=int, default=0, metavar="N",
                   help="保存最值得看的 N 局牌谱（按 hora 数和得分差筛选）")
    p.add_argument("--save-all-games", action="store_true",
                   help="保存全部牌谱为 .mjson")
    p.add_argument("--save-npz", action="store_true",
                   help="将自对战数据保存为 .npz 训练样本")
    p.add_argument("--output-dir", default="selfplay_out", help="输出目录")
    p.add_argument("--beam-k", type=int, default=3, help="beam search k (0=禁用)")
    p.add_argument("--beam-lambda", type=float, default=1.0, help="value head 权重")
    p.add_argument("--device", default="cpu", help="推理设备")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    # 加载 4 个 bot（共享同一模型权重）
    print(f"加载模型: {args.model}")
    bots = [
        KeqingBot(
            player_id=i,
            model_path=args.model,
            device=args.device,
            beam_k=args.beam_k,
            beam_lambda=args.beam_lambda,
            verbose=args.verbose,
        )
        for i in range(4)
    ]
    print(f"开始自对战: {args.games} 局，beam_k={args.beam_k}，beam_lambda={args.beam_lambda}")

    manager = BattleManager()
    config = BattleConfig(
        player_count=4,
        players=[{"id": i, "name": f"Bot{i}", "type": "bot"} for i in range(4)],
    )

    rank_counts = defaultdict(int)   # rank -> count
    total_scores = [0] * 4
    hora_total = 0
    game_results = []  # [{game_id, scores, ranks, interest, events}]

    for game_idx in range(args.games):
        room = manager.create_room(config)
        room.human_player_id = -1
        room.state.bakaze = "E"
        room.state.kyoku = 1
        room.state.honba = 0
        room.state.oya = 0
        room.state.scores = [25000, 25000, 25000, 25000]
        room.state.kyotaku = 0

        seed = random.randint(0, 2**31) if args.seed is None else args.seed + game_idx
        result = run_one_kyoku(manager, room, bots, seed=seed)

        scores = result["scores"]
        events = result["events"]
        ranks = _rank_of_scores(scores)
        interest = _score_interest(events, scores)
        hora_count = sum(1 for e in events if e.get("type") == "hora")
        hora_total += hora_count

        for pid in range(4):
            rank_counts[ranks[pid]] += 1
            total_scores[pid] += scores[pid]

        game_results.append({
            "game_id": game_idx,
            "scores": scores,
            "ranks": ranks,
            "interest": interest,
            "events": events,
        })

        if (game_idx + 1) % max(1, args.games // 10) == 0 or game_idx == args.games - 1:
            done = game_idx + 1
            avg_rank = sum(r * c for r, c in rank_counts.items()) / max(1, done * 4)
            print(f"  [{done}/{args.games}] 平均顺位={avg_rank:.2f} hora/局={hora_total/done:.2f}")

    # 最终统计
    print("\n=== 自对战结果 ===")
    for rank in range(1, 5):
        print(f"  {rank}位: {rank_counts[rank]} 次 ({rank_counts[rank]/args.games*100:.1f}%)")
    print(f"  平均得分: {[s//args.games for s in total_scores]}")
    print(f"  hora/局: {hora_total/args.games:.2f}")

    # 保存牌谱
    if args.save_all_games or args.save_games > 0:
        mjson_dir = output_dir / "replays"
        mjson_dir.mkdir(exist_ok=True)

        if args.save_all_games:
            to_save = game_results
        else:
            to_save = sorted(game_results, key=lambda x: x["interest"], reverse=True)[:args.save_games]

        for r in to_save:
            fname = mjson_dir / f"game_{r['game_id']:05d}.mjson"
            _save_mjson(fname, r["events"])
        print(f"  保存 {len(to_save)} 局牌谱 -> {mjson_dir}")

    # 保存 npz
    if args.save_npz:
        npz_dir = output_dir / "npz"
        npz_dir.mkdir(exist_ok=True)
        saved = 0
        for r in game_results:
            arrays = _events_to_npz(r["events"])
            if arrays is None:
                continue
            fname = npz_dir / f"game_{r['game_id']:05d}.npz"
            _save_npz(fname, arrays)
            saved += 1
        print(f"  保存 {saved} 局 .npz 训练样本 -> {npz_dir}")

    # 保存统计 json
    stats = {
        "games": args.games,
        "rank_counts": dict(rank_counts),
        "avg_scores": [s / args.games for s in total_scores],
        "hora_per_game": hora_total / args.games,
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计 -> {output_dir}/stats.json")


if __name__ == "__main__":
    main()
