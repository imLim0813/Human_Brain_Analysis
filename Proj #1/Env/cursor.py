import pygame as G
import numpy as np
from Env.parameters import Parameters
from Env.func import base, adapt, reverse


class Cursor(Parameters):
    # 커서에 관한 클래스
    # Method : display, update

    def __init__(self):
        super().__init__()
        self.x = self.init_x
        self.y = self.init_y
        self.max_x = self.width - self.target_diameter
        self.max_y = self.height - self.target_diameter

    def display(self, screen):
        # 십자가 모양의 커서를 콘솔 창에 디스플레이 하기 위한 함수
        # 길이 : 20 pixel

        G.draw.line(screen.screen, self.white, (self.x + 25, self.y + 35), (self.x + 45, self.y + 35), 3)
        G.draw.line(screen.screen, self.white, (self.x + 35, self.y + 25), (self.x + 35, self.y + 45), 3)

    def move(self, mode: str, action):
        # 상태로부터 구해진 행동을 이용하여 커서의 좌표를 이동시키기 위한 함수
        # (r, theta) -> (x, y)

        act_x = action[0] * np.cos(np.deg2rad((action[1])))
        act_y = action[0] * np.sin(np.deg2rad(action[1]))

        if mode == 'base':
            self.x, self.y = base(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'adapt':
            self.x, self.y = adapt(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'reverse':
            self.x, self.y = reverse(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        else:
            print('Choose the mode among...')
            RuntimeError()

        return [int(self.x), int(self.y)] # int값을 이용하여 콘솔창에 디스플레이 하기 때문에