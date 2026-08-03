"""Test-suite env isolation.

用户级 `setx` 可能设置了 ``KEQING_LADDER_DATA_ROOT`` / ``KEQING_LADDER_CONFIG_DIR``
（正式 runtime 使用）。这些全局环境变量会污染依赖相对路径解析的测试
（``resolve_report_dir`` 优先使用 data root），因此每个测试开始前默认清除。

需要显式设置 env 的测试用 ``monkeypatch.setenv`` 覆盖（pytest 的 monkeypatch
是 per-test 共享实例，autouse fixture 的清理先于测试函数执行）。
"""

from __future__ import annotations

import pytest

_ENV_KEYS = ("KEQING_LADDER_DATA_ROOT", "KEQING_LADDER_CONFIG_DIR")


@pytest.fixture(autouse=True)
def _isolate_external_ladder_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
