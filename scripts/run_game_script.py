"""运行当前主线模型的本地镜像对局脚本"""
import sys, json
sys.path.insert(0, 'src')

import torch
from riichienv import RiichiEnv
from inference.runtime_bot import RuntimeBot

# 加载模型
MODEL_PATH = "artifacts/models/xmodel1/best.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 创建4个agent
print("创建4个Agent...")
bots = {}
for pid in range(4):
    bots[pid] = RuntimeBot(
        player_id=pid,
        model_path=MODEL_PATH,
        device=DEVICE
    )
    print(f"Bot {pid} 创建完成")

print("\n开始对局...")
env = RiichiEnv(game_mode="4p-red-half")
obs_dict = env.reset()

step_count = 0
while not env.done():
    actions = {}
    for pid, obs in obs_dict.items():
        new_events = obs.new_events()
        mjai_action = None
        for evt_str in new_events:
            evt = json.loads(evt_str)
            result = bots[pid].react(evt)
            if result is not None:
                mjai_action = result
        if mjai_action is None:
            actions[pid] = obs.legal_actions()[0]
        else:
            renv_action = obs.select_action_from_mjai(json.dumps(mjai_action))
            actions[pid] = renv_action if renv_action is not None else obs.legal_actions()[0]
    obs_dict = env.step(actions)
    step_count += 1
    if step_count % 100 == 0:
        print(f"已进行 {step_count} 步, 当前得分: {env.scores()}")

print(f"\n对局结束! 共 {step_count} 步")
print(f"最终排名: {env.ranks()}")
print(f"最终得分: {env.scores()}")

# 获取 replay 数据用于可视化
replay = env.get_replay()
print(f"\nReplay 数据已生成，长度: {len(json.dumps(replay))} 字符")
