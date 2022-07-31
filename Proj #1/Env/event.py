from Env.parameters import Parameters
import pygame as G


class Event(Parameters):
    # 이벤트 정의를 위한 클래스
    # Method : hit_target

    def __init__(self):
        super().__init__()

    def hit_target(self, screen, target):
        # 커서가 타겟을 맞추면 타겟의 색을 빨간색으로 변경
        G.draw.ellipse(screen.screen, self.red, target.rect, 0)

