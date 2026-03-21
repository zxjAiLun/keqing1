from convert.link_converter import parse_log_url


def test_parse_tenhou_url() -> None:
    u = "https://tenhou.net/0/?log=2019050417gm-0029-0000-4f2a8622&tw=2"
    out = parse_log_url(u)
    assert out["site"] == "tenhou"
    assert out["log_id"] == "2019050417gm-0029-0000-4f2a8622"
    assert out["tw"] == "2"


def test_parse_mjsoul_url() -> None:
    u = "https://game.mahjongsoul.com/?paipu=abc"
    out = parse_log_url(u)
    assert out["site"] == "mjsoul"

