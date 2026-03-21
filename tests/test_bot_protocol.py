import json

from bot.mjai_bot import MjaiPolicyBot
from train.train_sl import train


class _Args:
    log_path = "log.jsonl"
    config = "configs/mvp.yaml"
    out_dir = "artifacts/sl"


def test_bot_react_json() -> None:
    train(_Args())
    bot = MjaiPolicyBot(player_id=2, checkpoint_path="artifacts/sl/best.npz")
    out = bot.react('{"type":"start_game"}')
    obj = json.loads(out)
    assert "type" in obj

