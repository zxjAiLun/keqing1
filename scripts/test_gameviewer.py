#!/usr/bin/env python3
"""
GameViewer测试脚本

测试GameViewer是否正确安装和配置

Usage:
    python scripts/test_gameviewer.py
"""

import sys
from pathlib import Path

# 添加src到path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from riichienv import RiichiEnv
from riichienv.agents import RandomAgent


def test_basic_viewer():
    """测试基本的GameViewer功能"""
    print("\n" + "=" * 70)
    print("测试1: 基本GameViewer功能")
    print("=" * 70)
    
    # 创建环境
    env = RiichiEnv(game_mode="4p-red-single")
    agent = RandomAgent()
    
    # 运行对局
    obs_dict = env.reset()
    step = 0
    max_steps = 30
    
    print(f"运行{ max_steps}步对局...")
    
    while not env.done() and step < max_steps:
        actions = {pid: agent.act(obs) for pid, obs in obs_dict.items()}
        obs_dict = env.step(actions)
        step += 1
    
    print(f"✓ 对局完成，共{step}步")
    
    # 测试GameViewer
    print("\n创建GameViewer...")
    viewer = env.get_viewer()
    
    if viewer:
        print("✓ GameViewer创建成功")
        print(f"  类型: {type(viewer)}")
        print(f"  方法: {dir(viewer)[:10]}...")
        return viewer
    else:
        print("✗ GameViewer创建失败")
        return None


def test_viewer_methods(viewer):
    """测试GameViewer的方法"""
    if not viewer:
        print("\n⚠ 跳过方法测试")
        return
    
    print("\n" + "=" * 70)
    print("测试2: GameViewer方法")
    print("=" * 70)
    
    # 检查可用方法
    methods = [m for m in dir(viewer) if not m.startswith('_')]
    print(f"可用方法: {methods}")
    
    # 测试save方法（如果存在）
    if hasattr(viewer, 'save'):
        print("✓ save() 方法可用")
        print("  可以保存HTML回放")
    else:
        print("⚠ save() 方法不可用")
    
    # 测试show方法（如果存在）
    if hasattr(viewer, 'show'):
        print("✓ show() 方法可用")
        print("  可以在Jupyter中显示")
    else:
        print("⚠ show() 方法不可用")


def test_summary(viewer):
    """测试GameViewer的summary方法"""
    if not viewer:
        print("\n⚠ 跳过summary测试")
        return
    
    print("\n" + "=" * 70)
    print("测试3: Viewer摘要信息")
    print("=" * 70)
    
    if hasattr(viewer, 'summary'):
        try:
            summary = viewer.summary()
            print(f"✓ summary() 成功")
            print(f"  摘要类型: {type(summary)}")
            if isinstance(summary, list):
                print(f"  局数: {len(summary)}")
        except Exception as e:
            print(f"⚠ summary() 调用失败: {e}")
    else:
        print("⚠ summary() 方法不可用")


def main():
    """主函数"""
    print("=" * 70)
    print("GameViewer功能测试")
    print("=" * 70)
    
    try:
        # 测试基本功能
        viewer = test_basic_viewer()
        
        # 测试方法
        test_viewer_methods(viewer)
        
        # 测试摘要
        test_summary(viewer)
        
        print("\n" + "=" * 70)
        print("测试完成")
        print("=" * 70)
        print("\nGameViewer使用方法:")
        print("1. 在Jupyter Notebook中:")
        print("   from riichienv import RiichiEnv")
        print("   env = RiichiEnv()")
        print("   # 运行对局...")
        print("   viewer = env.get_viewer()")
        print("   viewer.show()  # 在Notebook中显示3D回放")
        print("\n2. 保存为HTML:")
        print("   viewer.save('replay.html')  # 保存为HTML文件")
        print("\n3. 在浏览器中打开:")
        print("   viewer.save('replay.html')")
        print("   # 然后用浏览器打开replay.html")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
