from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from mahjong_env.replay import ReplaySample


def summarize_samples(samples: Iterable[ReplaySample]) -> Dict:
    samples_list: List[ReplaySample] = list(samples)
    by_actor = Counter(s.actor for s in samples_list)
    by_name = Counter(s.actor_name for s in samples_list)
    by_action = Counter(s.label_action["type"] for s in samples_list)
    return {
        "num_samples": len(samples_list),
        "by_actor": dict(sorted(by_actor.items())),
        "by_name": dict(sorted(by_name.items())),
        "by_action_type": dict(sorted(by_action.items())),
    }

