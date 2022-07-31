import gym
import os
import cv2
import pygame as G
import matplotlib.pyplot as plt
import numpy as np
import time

from abc import ABC
from Env.func import *
from Env.parameters import Parameters
from Env.screen import Screen
from Env.target import Target
from Env.event import Event
from Env.cursor import Cursor
from Env.func import flip

path_2 = np.load('./Data/total_path.npy').astype('int')

os_driver(True)


class Load(Parameters, gym.Env, ABC):
    def __init__(self):
        super().__init__()
        self.clock = G.time.Clock()
        self.screen = Screen()
        self.target = Target()
        self.cursor = Cursor()
        self.event = Event()

        visible_mouse(False)

        self.done = False
        self.count = 0

        # Action space 정의
        act_high = 1.0

        self.action_r = gym.spaces.Box(low=np.float(0), high=np.float(act_high), shape=(1,))
        self.action_theta = gym.spaces.Box(low=np.float(-act_high), high=np.float(act_high), shape=(1,))

        # Observation 정의
        obs_high = np.array([1920., 1080.], dtype=np.float)
        obs_low = np.array([0., 0.], dtype=np.float)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(2,), dtype=np.float)

        # State space 정의
        self.state = np.array([self.init_x, self.init_y,
                               self.init_x, self.init_y], dtype=np.float)

        self.action = None

    def step(self, r, theta, path=path_2):
        # 상호작용
        # 타겟을 움직이고, 그 상태를 바탕으로 강화학습이 출력한 행동으로 커서를 움직인 뒤, 현재상태에 저장.
        self.target.move(path, idx=self.count)
        self.action = np.array([r[0] * 6, theta[0] * 180])
        tmp = self.cursor.move('base', self.action)
        self.state = np.array([tmp[0], tmp[1], self.target.pos()[0], self.target.pos()[1]], dtype=np.float)

        # 보상을 정의.
        # 타겟과 커서 간 유클리디안 거리를 구하고, 가우시안 함수의 입력으로 사용해 보상 출력.
        dt = euclidean_distance(self.state)
        reward = distance_reward(dt, sigma=100)

        # Reset 정의
        # 1) 경로를 모두 수행 2) 타겟과 커서 간 거리가 350 pixel 이상.
        if self.duration == self.count:
            self.done = True
        if dt >= 350:
            self.done = True

        # 인포메이션 정의
        info = {}

        self.count += 1

        # 실제 실험에는 Trial 존재하므로 고려하기 위함.
        # 1 trial 끝나면 커서를 정중앙으로 위치시킴.
        if self.count != 0 and self.count % 1500 == 0:
            self.cursor = Cursor()

        return self.state, reward, self.done, info

    def reset(self):
        self.count = 0
        self.done = False
        self.target = Target()
        self.cursor = Cursor()
        self.event = Event()

        self.state = np.array([self.init_x, self.init_y, self.init_x, self.init_y], dtype=np.float)

        return self.state

    def to_frame(self, width, height):
        # 콘솔 창으로부터 프레임을 얻고, 이를 3차원 array로 변환
        self.render()
        string_image = G.image.tostring(self.screen.screen, 'RGB')
        temp_surf = G.image.fromstring(string_image, (self.width, self.height), 'RGB')
        tmp = G.surfarray.array3d(temp_surf)
        tmp = tmp.transpose((1, 0, 2))

        # 이미지 사이즈 조절
        # 1) Padding 2) Cropping 3) Resize
        img_pad = np.pad(tmp, ((448, 448), (448, 448), (0, 0)))
        cropped_img = img_pad[int(self.state[1]): int(self.state[1]) + 896, int(self.state[0]):int(self.state[0]) + 896,
                      :]
        image = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_AREA)

        # RGB scale -> Gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def render(self):

        # If don't use this, the console window is not the same with the state.
        G.event.pump()

        # Set the hertz
        clock_tick(self.hertz)

        # Fill the window with black ground.
        self.screen.overwrite()

        # Display target.
        self.target.display('base', self.screen)

        # If hit, then the target will show red color.
        if hit(self.state):
            self.event.hit_target(self.screen, self.target)
        else:
            pass

        # Display cursor
        self.cursor.display(self.screen)

        # Update the console window.
        flip()