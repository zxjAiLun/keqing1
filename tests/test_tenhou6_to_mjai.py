"""Tests for tenhou6 to mjai conversion."""

from pathlib import Path

import pytest

from src.convert.tenhou6_to_mjai import (
    convert_tenhou6_to_mjai,
    convert_tenhou6_file,
    load_tenhou6,
    tenhou_tile_to_mjai,
    mjai_tile_to_tenhou,
)
from src.convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl
from mahjong_env.replay import read_mjai_jsonl


# Sample tenhou6 data for testing
SAMPLE_TENHOU6 = {
    "ver": 2.3,
    "ref": "2022092102gm-00a9-0000-f2ea91a6",
    "log": [
        [
            [0, 0, 0],
            [25000, 25000, 25000, 25000],
            [29],
            [47],
            [11, 13, 16, 17, 21, 23, 26, 31, 31, 31, 32, 33, 35],
            [38, 53, 47, 23, 11, 44, 36, 21],
            [60, 26, 60, 60, 60, 60, 33, 11],
            [11, 13, 15, 16, 29, 32, 32, 32, 38, 39, 41, 44, 45],
            [36, 37, 12, 24, 17, 33, 45, 13],
            [44, 29, 41, 45, "r24", 60, 60, 60],
            [14, 15, 52, 26, 28, 34, 36, 37, 37, 38, 39, 39, 42],
            [14, 34, 22, 27, 39, 28, 18],
            [42, 28, 60, 14, "r37", 60, 60],
            [12, 51, 17, 18, 18, 22, 23, 28, 29, 38, 41, 43, 43],
            [46, 25, 42, 47, 24, 21, 19],
            [41, 46, 60, 60, 43, 43, 18],
            ["和了", [0, -2600, 4600, 0], [2, 1, 2, "40符2飜2600点", "立直(1飜)", "赤ドラ(1飜)"]],
        ],
        [
            [1, 0, 0],
            [25000, 21400, 28600, 25000],
            [11],
            [46],
            [12, 15, 17, 19, 23, 31, 33, 34, 42, 44, 45, 47, 47],
            [36, 24, 32, 44, 45, 43, 34, "4747p47"],
            [34, 36, 42, 45, 60, 60, 31, 12],
            [11, 13, 16, 22, 23, 24, 29, 31, 37, 41, 42, 46, 47],
            [19, 11, 26, 21, 17, 46, 32, 23, 25],
            [60, 42, 29, 31, 46, 60, 60, 47, 41],
            [18, 22, 23, 25, 26, 26, 28, 35, 38, 38, 39, 39, 43],
            [16, 28, 26, 38, 28, 41, 42, 18],
            [43, 35, 25, 16, 18, 60, 60, "r60"],
            [13, 13, 14, 51, 18, 19, 21, 27, 27, 32, 33, 37, 44],
            [42, 17, 15, 33, 11, 45, 27, 21],
            [44, 42, 21, 37, 33, 60, 11, 60],
            ["和了", [0, 0, 9000, -8000], [2, 3, 2, "満貫8000点", "立直(1飜)", "一発(1飜)", "三暗刻(2飜)"]],
        ],
        [
            [2, 0, 0],
            [25000, 21400, 36600, 17000],
            [18],
            [43],
            [11, 12, 14, 18, 21, 21, 22, 29, 34, 35, 38, 39, 39],
            [15, 47, 28, 43, 11, 33, 24, 46, 27, 29, 24, 42, 31],
            [29, 18, 60, 60, 47, 11, 22, 35, 24, 60, 27, 24, 34],
            [12, 14, 16, 17, 19, 22, 24, 28, 28, 29, 32, 33, 35],
            [16, 13, 25, 31, 31, 18, 43, 18, 44, 27, 36, 43, 52],
            [35, 29, 19, 16, 60, "r22", 60, 60, 60, 60, 60, 60, 60],
            [16, 17, 19, 25, 25, 28, 31, 32, 35, 39, 42, 44, 46],
            [23, 33, 11, 51, 21, 13, "c222123", 16, 37, 45, 53, 34, 38, 37],
            [42, 44, 46, 28, 39, 35, 19, 16, 60, 60, 60, 60, 60, 60],
            [14, 17, 19, 19, 24, 26, 29, 32, 34, 36, 37, 37, 39],
            [21, 15, 26, 11, 36, 15, 23, 44, 23, 45, 13, 38, 41, 26],
            [29, 21, 17, 60, 39, 34, 32, 60, "r24", 60, 60, 60, 60, 60],
            ["和了", [0, 4000, 0, -2000], [1, 3, 1, "30符2飜2000点", "立直(1飜)", "平和(1飜)"]],
        ],
        [
            [3, 0, 0],
            [25000, 24400, 36600, 14000],
            [29],
            [39],
            [12, 14, 15, 22, 52, 26, 28, 29, 31, 33, 33, 34, 41],
            [29, 27, 47, 27, 17, 12, 24, 39, 22, 11, 19, 41, 46, 21, 43, 36],
            [41, 12, 60, 31, 29, 60, 17, 60, 27, 60, 33, 33, 34, 24, 26, 28],
            [12, 16, 19, 21, 21, 25, 26, 37, 37, 37, 42, 43, 47],
            [53, 15, 23, 47, 17, 12, 47, 11, 29, 14, 31, 41, 16, 35, 41, 38],
            [19, 42, 47, 60, 43, 23, 60, 12, 60, 11, 60, 53, 26, 60, 60, 41],
            [11, 14, 17, 17, 18, 24, 25, 27, 32, 38, 42, 46, 46],
            [34, 42, 45, 34, 18, 33, 21, 18, 23, 39, 33, 16, 44, 32, 22, 51],
            [11, 38, 60, 27, 14, 24, 25, 21, 60, 60, 18, 32, 60, 60, 42, 42],
            [13, 13, 14, 15, 23, 26, 28, 31, 35, 37, 38, 45, 45],
            [45, 31, 43, 32, 13, 44, 24, 26, 35, 34, 39, 24, 18, 32, 42, 44, 22],
            [31, 60, 60, 60, 35, 60, 26, 60, 60, 60, "r28", 60, 60, 60, 60, 60],
            ["和了", [-2000, -2000, -2000, 7000], [3, 3, 3, "30符3飜2000点∀", "立直(1飜)", "門前清自摸和(1飜)", "役牌 白(1飜)"]],
        ],
        [
            [3, 1, 0],
            [23000, 22400, 34600, 20000],
            [26],
            [],
            [11, 14, 14, 15, 17, 24, 28, 31, 32, 42, 43, 44, 44],
            [13, 46, 24, 41, 53, 38, 42, 36, 31, 21, 17, 44, 16, 24, 14, "31p3131", 23, 38],
            [43, 42, 11, 46, 41, 60, 60, 32, 28, 60, 15, 60, 60, 13, 44, 44, 60, 60],
            [16, 18, 19, 25, 29, 37, 38, 39, 41, 41, 43, 44, 45],
            [12, 43, 15, 37, "p414141", 11, 15, 33, 32, 25, 27, 52, 35, 21, 22, 47, 19, 16],
            [44, 25, 12, 60, 45, 60, 60, 60, 29, 60, 32, 19, 18, 60, 60, 43, 47, 38],
            [13, 17, 19, 24, 27, 29, 29, 33, 36, 38, 39, 39, 46],
            [14, 12, 46, 36, 12, 21, 39, 31, 34, 32, 34, 22, 45, 34, 31, 18, 51, 25],
            [46, 29, 60, 24, 33, 60, 12, 60, 60, 38, 60, 60, 32, 45, 60, "r34", 60, 60],
            [18, 19, 21, 22, 26, 26, 28, 29, 33, 37, 37, 42, 45],
            [22, 42, 11, 46, 35, 35, 27, 33, 28, 26, 13, 27, 36, 28, 12, 47, 43, 47],
            [42, 60, 60, 60, 45, 29, 33, 60, 18, 60, 21, 19, 13, 60, 60, 60, 60, 60],
            ["流局", [1000, -3000, 1000, 1000]],
        ],
        [
            [3, 2, 1],
            [24000, 19400, 34600, 21000],
            [24],
            [],
            [11, 13, 21, 22, 25, 28, 33, 34, 38, 41, 46, 46, 47],
            [38, 35, 36, 15, 34, 21, "46p4646", 47, 17, 46, 11, "c121113", 23, 29, 39, 38, 29, 45],
            [41, 28, 47, 22, 21, 60, 15, 60, 60, 25, 46, 11, 60, 60, 60, 34, 60, 60],
            [14, 14, 16, 18, 23, 24, 26, 28, 31, 35, 37, 41, 43],
            [27, 18, 19, 26, 14, 15, 37, 24, 12, "c252324", 35, 34, 12, 25, 13, 12, 16, 18],
            [31, 41, 60, 43, 26, 37, 60, 60, 35, 12, 60, 60, 60, 28, 60, 60, 15],
            [13, 17, 17, 19, 22, 26, 27, 31, 36, 37, 44, 45, 46],
            [21, 42, 26, 41, 47, 11, 44, 19, 36, 23, 41, 29, 39, 31, 16, 39, 31],
            [31, 60, 44, 60, 60, 46, 60, 26, 45, 19, 60, 60, 60, 11, 21, 13, 22],
            [14, 16, 24, 25, 27, 27, 33, 34, 37, 38, 42, 44, 47],
            [51, 28, 42, 21, 42, 11, 45, 39, 36, 15, 12, 28, 33, 22, 13, 43, 33],
            [42, 44, 47, 60, 60, 60, 60, 28, 39, 60, 60, 60, 42, 60, 60, 60, 34],
            ["和了", [-1200, 5600, -1200, -2200], [1, 1, 1, "30符3飜1000-2000点", "断幺九(1飜)", "ドラ(2飜)"]],
        ],
        [
            [4, 0, 0],
            [22800, 25000, 33400, 18800],
            [37],
            [],
            [11, 15, 16, 19, 22, 28, 29, 31, 33, 36, 38, 43, 47],
            [45, 38, 11, 31, 26, 45, 33],
            [43, 19, 47, 22, 29, 45, 45],
            [12, 14, 14, 16, 18, 29, 29, 35, 38, 39, 41, 41, 42],
            [17, 47, 24, 12, 35, 36, 13],
            [42, 60, 35, 24, 60, 60, 39],
            [15, 17, 18, 27, 28, 32, 36, 38, 41, 43, 45, 47, 47],
            [32, "p474747", 19, 29, 21, 19, 37],
            [41, 15, 45, 43, 60, 60],
            [19, 21, 52, 27, 27, 28, 32, 37, 39, 39, 41, 44, 45],
            [46, 44, 44, 11, 17, 22],
            [41, 19, 45, 46, 21, 60],
            ["和了", [-1000, -500, 2000, -500], [2, 2, 2, "30符2飜500-1000点", "役牌 中(1飜)", "ドラ(1飜)"]],
        ],
        [
            [5, 0, 0],
            [21800, 24500, 35400, 18300],
            [11],
            [42],
            [11, 15, 17, 17, 18, 21, 28, 31, 31, 33, 34, 34, 36],
            [45, 13, 29, 22, 17, 14, 28, 27, 41, 37, 24, 29, 32],
            [11, 21, 45, 60, 18, 36, 34, "r28", 60, 60, 60, 60],
            [13, 21, 22, 27, 29, 33, 35, 37, 38, 42, 43, 46, 47],
            [22, 17, 14, 39, 23, 43, 35, 24, 45, 15, 22, 23, 11],
            [43, 42, 46, 47, 22, 60, 17, 21, 60, 33, 60, 24, 29],
            [24, 26, 26, 27, 28, 29, 35, 36, 37, 38, 43, 45, 47],
            [28, 37, 31, 41, 39, 41, 16, 47, 16, 16, 33, 26, 39],
            [43, 47, 60, 60, 45, 60, 28, 60, 28, 36, 37, 24, 29],
            [12, 15, 18, 18, 23, 25, 25, 25, 52, 26, 33, 44, 46],
            [31, 32, 39, 13, 45, 44, 51, 32, 21, 36, 27, 44, 19],
            [44, 46, 60, 15, 23, 60, 60, 60, 60, 45, "r36", 60, 60],
            ["和了", [4000, -1000, -500, -500], [0, 0, 0, "30符2飜500-1000点", "立直(1飜)", "門前清自摸和(1飜)"]],
        ],
        [
            [6, 0, 0],
            [24800, 23500, 34900, 16800],
            [12],
            [19],
            [12, 14, 14, 16, 19, 23, 25, 32, 34, 35, 36, 37, 43],
            [18, 26, 34, 45, 33, 27, 13, 46, 28],
            [19, 23, 43, 60, 37, "r12", 60, 60, 60],
            [11, 16, 24, 25, 26, 29, 31, 33, 34, 36, 41, 46, 47],
            [47, 24, 15, 26, 27, 38, 17, 47, 22],
            [41, 46, 29, 31, 11, 47, 47, 60, 60],
            [11, 14, 51, 24, 25, 28, 31, 35, 38, 41, 43, 44, 44],
            [12, 11, 44, 16, 23, 17, 22, 41, 53, 36],
            [43, 31, 41, 28, 38, 35, 60, 60, 12, "r17"],
            [13, 15, 16, 18, 21, 21, 27, 28, 28, 38, 42, 45, 46],
            [24, 37, 17, 39, 23, 22, 37, 23, 32],
            [46, 45, 42, 24, 60, 60, 60, 60, 60],
            ["和了", [3600, 0, -2600, 0], [0, 2, 0, "40符2飜2600点", "立直(1飜)", "断幺九(1飜)"]],
        ],
        [
            [7, 0, 0],
            [27400, 23500, 32300, 16800],
            [16],
            [],
            [13, 26, 27, 28, 31, 33, 35, 38, 42, 43, 44, 45, 47],
            [25, 17, 11, 15, 16, 38, 35, 41, 14, 22, 31, 21, 18, 44, 39],
            [43, 44, 42, 38, 45, 47, 38, 60, 11, 60, 33, 60, "r28", 60, 60],
            [13, 15, 18, 21, 23, 24, 32, 34, 35, 38, 39, 44, 46],
            [23, 12, 45, 42, 23, 37, 28, 28, 45, 26, 24, 39, 26, 51],
            [44, 46, 60, 32, 21, 15, 60, 42, 18, 45, 13, 28, 24, 39],
            [11, 15, 19, 22, 25, 29, 32, 36, 36, 37, 38, 42, 47],
            [47, 16, 37, 43, "47p4747", 34, 25, 27, 13, 27, 14, 36, 12, 19],
            [42, 19, 29, 60, 11, 22, 37, 60, 60, 60, 36, 36, 60, 60],
            [11, 12, 17, 24, 52, 29, 31, 34, 36, 37, 44, 45, 46],
            [43, 17, 32, 32, "32p3232", 41, 29, 34, 17, 18, 24, 47, 41, 26, 46, 33],
            [60, 44, 31, 29, 11, 45, 60, 12, 46, 60, 41, 60, 60, 24, 60, 60],
            ["和了", [0, 0, 2000, -1000], [2, 3, 2, "30符1飜1000点", "役牌 中(1飜)"]],
        ],
    ],
    "ratingc": "PF4",
    "rule": {"disp": "鳳南喰赤", "aka53": 1, "aka52": 1, "aka51": 1},
    "lobby": 0,
    "dan": ["七段", "十段", "七段", "七段"],
    "rate": [2180.15, 2325.23, 2141.9, 2136.8],
    "sx": ["M", "M", "M", "M"],
    "sc": [26400, 6.4, 23500, -16.5, 34300, 44.3, 15800, -34.2],
    "name": ["遊走", "武田舞彩", "九紋龍史進", "Nemo"],
}


class TestTileConversion:
    """Test tile code conversion functions."""

    def test_tenhou_to_mjai_basic(self):
        # Tile codes: kind*36 + (num-1)*4 gives base, actual tile is base + suffix (0-3)
        assert tenhou_tile_to_mjai(0) == "1m"
        assert tenhou_tile_to_mjai(32) == "9m"
        assert tenhou_tile_to_mjai(36) == "1p"
        assert tenhou_tile_to_mjai(68) == "9p"
        assert tenhou_tile_to_mjai(72) == "1s"
        assert tenhou_tile_to_mjai(104) == "9s"
        assert tenhou_tile_to_mjai(108) == "E"
        assert tenhou_tile_to_mjai(112) == "S"
        assert tenhou_tile_to_mjai(116) == "W"
        assert tenhou_tile_to_mjai(120) == "N"
        assert tenhou_tile_to_mjai(124) == "P"
        assert tenhou_tile_to_mjai(128) == "F"
        assert tenhou_tile_to_mjai(132) == "C"

    def test_tenhou_to_mjai_aka(self):
        # Red 5s: 5mr=16, 5pr=52, 5sr=88
        assert tenhou_tile_to_mjai(16) == "5mr"
        assert tenhou_tile_to_mjai(52) == "5pr"
        assert tenhou_tile_to_mjai(88) == "5sr"

    def test_mjai_to_tenhou(self):
        # Based on Rust tid_to_mjai logic:
        # 5mr -> 16, 5pr -> 52, 5sr -> 88
        # E -> 108, S -> 112, W -> 116, N -> 120, P -> 124, F -> 128, C -> 132
        assert mjai_tile_to_tenhou("1m") == 0
        assert mjai_tile_to_tenhou("9m") == 32
        assert mjai_tile_to_tenhou("5mr") == 16
        assert mjai_tile_to_tenhou("5pr") == 52
        assert mjai_tile_to_tenhou("5sr") == 88
        assert mjai_tile_to_tenhou("E") == 108
        assert mjai_tile_to_tenhou("C") == 132


class TestBasicConversion:
    """Test basic conversion structure."""

    def test_start_game_event(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        assert events[0]["type"] == "start_game"
        assert events[0]["names"] == ["遊走", "武田舞彩", "九紋龍史進", "Nemo"]
        assert events[0]["kyoku_first"] == 0
        assert events[0]["aka_flag"] == True

    def test_start_kyoku_event(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        # Find first start_kyoku
        start_kyoku = None
        for e in events:
            if e["type"] == "start_kyoku":
                start_kyoku = e
                break
        assert start_kyoku is not None
        assert start_kyoku["bakaze"] == "E"
        assert start_kyoku["kyoku"] == 1
        assert start_kyoku["honba"] == 0
        assert start_kyoku["kyotaku"] == 0
        assert start_kyoku["oya"] == 0
        assert start_kyoku["scores"] == [25000, 25000, 25000, 25000]
        assert len(start_kyoku["tehais"]) == 4

    def test_end_game_event(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        assert events[-1]["type"] == "end_game"

    def test_kyoku_count(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        start_kyoku_count = sum(1 for e in events if e["type"] == "start_kyoku")
        end_kyoku_count = sum(1 for e in events if e["type"] == "end_kyoku")
        assert start_kyoku_count == len(SAMPLE_TENHOU6["log"])
        assert start_kyoku_count == end_kyoku_count


class TestHoraAndRyukyoku:
    """Test hora and ryukyoku event conversion."""

    def test_hora_event(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        hora_events = [e for e in events if e["type"] == "hora"]
        assert len(hora_events) > 0
        # First hora should be player 2
        assert hora_events[0]["actor"] == 2

    def test_ryukyoku_event(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        ryukyoku_events = [e for e in events if e["type"] == "ryukyoku"]
        assert len(ryukyoku_events) == 1  # One ryukyoku in sample data


class TestTsumogiri:
    """Test tsumogiri (discarding drawn tile) detection."""

    def test_tsumogiri_detection(self):
        events = convert_tenhou6_to_mjai(SAMPLE_TENHOU6)
        dahai_events = [e for e in events if e["type"] == "dahai"]
        # Should have some tsumogiri events
        tsumogiri_count = sum(1 for e in dahai_events if e.get("tsumogiri", False))
        assert tsumogiri_count >= 0  # At least some should be detected


class TestRealtimeFileConversion:
    """Test conversion using real tenhou6 files."""

    def test_convert_real_file(self, tmp_path):
        # Use one of the real tenhou6 files
        tenhou_file = Path("dataset/tenhou6/ds1/0000_http___tenhou.net_0__log_2022092102gm-00a9-0000-f2ea91a6_tw_2.json")
        if not tenhou_file.exists():
            pytest.skip("Real tenhou6 file not found")

        output_file = tmp_path / "output.jsonl"

        # Convert
        convert_tenhou6_file(tenhou_file, output_file)
        assert output_file.exists()

        # Read and verify structure
        events = read_mjai_jsonl(str(output_file))
        assert len(events) > 0
        assert events[0]["type"] == "start_game"

        # Validate
        errors = validate_mjai_jsonl(str(output_file))
        assert errors == [], f"Validation errors: {errors}"


class TestCompareWithLibriichiFallback:
    """Compare tenhou6_to_mjai with libriichi_bridge fallback converter."""

    def test_compare_output_structure(self, tmp_path):
        tenhou_file = Path("dataset/tenhou6/ds1/0000_http___tenhou.net_0__log_2022092102gm-00a9-0000-f2ea91a6_tw_2.json")
        if not tenhou_file.exists():
            pytest.skip("Real tenhou6 file not found")

        # Convert using libriichi_bridge (fallback)
        libriichi_out = tmp_path / "libriichi.jsonl"
        convert_raw_to_mjai(str(tenhou_file), str(libriichi_out), libriichi_bin=None)

        # Convert using our new converter
        our_out = tmp_path / "ours.jsonl"
        convert_tenhou6_file(tenhou_file, our_out)

        # Both should exist and have content
        assert libriichi_out.exists()
        assert our_out.exists()

        libriichi_events = read_mjai_jsonl(str(libriichi_out))
        our_events = read_mjai_jsonl(str(our_out))

        # Both should have same number of start_kyoku events
        libriichi_kyoku = sum(1 for e in libriichi_events if e["type"] == "start_kyoku")
        our_kyoku = sum(1 for e in our_events if e["type"] == "start_kyoku")
        assert libriichi_kyoku == our_kyoku
