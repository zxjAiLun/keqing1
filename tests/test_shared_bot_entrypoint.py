from inference.runtime_bot import RuntimeBot
from replay import bot as replay_bot


def test_replay_bot_classes_use_shared_runtime_bot():
    assert replay_bot._BOT_CLASSES["xmodel1"] is RuntimeBot
    assert replay_bot._BOT_CLASSES["keqingv4"] is RuntimeBot
