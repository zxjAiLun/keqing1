from .decoder import Meld


class State:
    def __init__(self, name: str = 'NoName', room: str = '0_0'):
        self.name: str = name
        self.room: str = room.replace('_', ',')
        # 手牌(天鳳インデックス)
        self.hand: list[int] = []
        # 立直をかけているか
        self.in_riichi: bool = False
        # 壁牌の枚数
        self.live_wall: int | None = None
        # 副露のリスト
        self.melds: list[Meld] = []
        # 待ち
        self.wait: set[int] = set()
        # 天凤分配的实际座位号（从 UN 消息解析）
        self.seat: int = 0
        # 已发出荣和/自摸请求，等待天凤确认，忽略后续事件
        self.hora_pending: bool = False
