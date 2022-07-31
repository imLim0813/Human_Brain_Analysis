from pygame.rect import *
from parameters import Parameters
import pygame as G


class Target(Parameters):
    # 타겟에 관한 클래스
    # Method : move, display, pos
    def __init__(self):
        super().__init__()
        self.rect = Rect(self.init_x, self.init_y, self.target_diameter, self.target_diameter)
        self.target = None

    def move(self, path, idx):
        # 정해진 경로에 따라 타겟을 이동시키기 위한 함수
        self.rect = Rect(path[idx][0], path[idx][1], self.target_diameter, self.target_diameter)

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.width:
            self.rect.right = self.width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self.height:
            self.rect.bottom = self.height

    def display(self, mode, screen):
        # 동그라미 모양의 타겟을 콘솔 창에 디스플레이 하기 위한 함수
        if mode == 'base':
            self.target = G.draw.ellipse(screen.screen, self.gray, self.rect, 0)
        elif mode == 'adapt':
            self.target = G.draw.ellipse(screen.screen, self.blue, self.rect, 0)
        elif mode == 'reverse':
            self.target = G.draw.ellipse(screen.screen, self.yellow, self.rect, 0)

    def pos(self):
        return [list(self.rect.copy())[0], list(self.rect.copy())[1]]

