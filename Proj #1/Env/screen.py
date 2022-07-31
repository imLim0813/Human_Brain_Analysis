import pygame as G
from Env.parameters import Parameters


class Screen(Parameters):
    # 게임 콘솔 창 정의를 위한 클래스
    # Method : overwrite
    def __init__(self):
        super().__init__()
        G.init()
        self.screen = G.display.set_mode([self.width, self.height])

    def overwrite(self):
        # 콘솔 창을 검은색으로 덮어쓰기
        self.screen.fill(self.black)