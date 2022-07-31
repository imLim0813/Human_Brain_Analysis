import pygame as G


class Parameters:
    # 여러 클래스에 거쳐 사용될 파라미터를 정리해놓은 클래스

    def __init__(self):
        self.red = G.Color(255, 0, 0)
        self.gray = G.Color(128, 128, 128)
        self.green = G.Color(0, 255, 0)
        self.white = G.Color(255, 255, 255)
        self.black = G.Color(0, 0, 0)
        self.blue = G.Color(0, 0, 255)
        self.yellow = G.Color(255, 255, 0)

        self.width = 1920
        self.height = 1080
        self.cursor_diameter = 20
        self.target_diameter = 70

        self.hertz = 60
        self.time = 25
        self.trial = 20
        self.duration = self.hertz * self.time * self.trial

        self.init_x = 960
        self.init_y = 540

