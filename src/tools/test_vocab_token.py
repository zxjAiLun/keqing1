#!/usr/bin/env python3
"""
测试 vocab.py 中的 token 转换是否正确

用法:
    python -m tools.test_vocab_token
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.vocab import build_action_vocab, action_to_token


def test_action_to_token():
    test_cases = [
        ({"type": "chi", "pai": "3m"}, "chi:3m"),
        ({"type": "chi", "pai": "7p"}, "chi:7p"),
        ({"type": "pon", "pai": "5p"}, "pon:5p"),
        ({"type": "pon", "pai": "9s"}, "pon:9s"),
        ({"type": "daiminkan", "pai": "7s"}, "daiminkan:7s"),
        ({"type": "ankan", "pai": "2m"}, "ankan:2m"),
        ({"type": "kakan", "pai": "4p"}, "kakan:4p"),
        ({"type": "dahai", "pai": "5m"}, "dahai:5m"),
        ({"type": "dahai", "pai": "5pr"}, "dahai:5pr"),
        ({"type": "reach"}, "reach"),
        ({"type": "hora"}, "hora"),
        ({"type": "ryukyoku"}, "ryukyoku"),
        ({"type": "none"}, "none"),
    ]

    all_passed = True
    for action, expected in test_cases:
        result = action_to_token(action)
        if result != expected:
            print(f"❌ FAIL: {action} -> {result}, expected {expected}")
            all_passed = False
        else:
            print(f"✅ PASS: {action} -> {result}")

    return all_passed


def test_vocab_completeness():
    actions, stoi = build_action_vocab()

    fuuro_types = ["chi", "pon", "daiminkan", "ankan", "kakan"]
    all_suits = ["m", "p", "s", "z"]

    all_passed = True

    for fuuro_type in fuuro_types:
        count = sum(1 for a in actions if a.startswith(f"{fuuro_type}:"))
        expected_count = 34
        if count != expected_count:
            print(f"❌ FAIL: {fuuro_type}:* token 数量 {count}, 期望 {expected_count}")
            all_passed = False
        else:
            print(f"✅ PASS: {fuuro_type}:* token 数量 = {count}")

    dahai_count = sum(1 for a in actions if a.startswith("dahai:"))
    if dahai_count != 34:
        print(f"❌ FAIL: dahai:* token 数量 {dahai_count}, 期望 34")
        all_passed = False
    else:
        print(f"✅ PASS: dahai:* token 数量 = {dahai_count}")

    print(f"\n词汇表总大小: {len(actions)}")

    return all_passed


def test_token_in_vocab():
    actions, stoi = build_action_vocab()

    test_tokens = [
        "chi:3m", "chi:7p", "chi:5s",
        "pon:5p", "pon:9s", "pon:1m",
        "daiminkan:7s", "daiminkan:E",
        "ankan:2m", "ankan:5p",
        "kakan:4p", "kakan:9s",
        "dahai:5m", "dahai:E", "dahai:9s",
    ]

    all_passed = True
    for token in test_tokens:
        if token not in stoi:
            print(f"❌ FAIL: token '{token}' 不在词汇表中")
            all_passed = False
        else:
            print(f"✅ PASS: token '{token}' 在词汇表中 (idx={stoi[token]})")

    return all_passed


def main():
    print("=" * 60)
    print("测试 action_to_token 函数")
    print("=" * 60)
    result1 = test_action_to_token()

    print("\n" + "=" * 60)
    print("测试词汇表完整性")
    print("=" * 60)
    result2 = test_vocab_completeness()

    print("\n" + "=" * 60)
    print("测试 token 在词汇表中")
    print("=" * 60)
    result3 = test_token_in_vocab()

    print("\n" + "=" * 60)
    if result1 and result2 and result3:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
