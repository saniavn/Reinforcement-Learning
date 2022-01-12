'''
This is the small_gridworld_final.py file that implements 
a small gridworld environment as part of the final project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 12/6/2021
'''
import numpy as np
import pygame as pg
import os


# Collision matrix
Coll = np.array([[0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 0, 2],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])

RIGHT = [0, 1]
LEFT = [0, -1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, RIGHT, LEFT]
NA = len(ACTIONS)
ACTION_IDX = [0, 1, 2, 3]
#Constants
WIDTH = 450     # width of the environment (px)
HEIGHT = 450    # height of the environment (px)
TS = 60         # delay in msec
NC, NR = np.shape(Coll)

# define colors
goal_color = pg.Color(0, 100, 0)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(0, 0, 0)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(120,120,0)
wall_color = pg.Color(140, 140, 140)


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    d = WIDTH // NC
    for i in range(NR):
        for j in range(NC):
            idx = Coll[i, j]
            if idx == 1:  #wall
                pg.draw.rect(scr, wall_color, (j*d, i*d ,d, d))            
            elif idx == 2: #lava
                pg.draw.rect(scr, bad_color, (j*d, i*d ,d, d))            
            elif idx == 3: #goal
                pg.draw.rect(scr, goal_color, (j*d, i*d ,d, d))            
    # Horizontal lines
    for i in range(NR+1):
        pg.draw.line(scr, line_color, (0, i*d), (WIDTH, i*d), 2)
    # Vertical lines
    for i in range(NC+1):
        pg.draw.line(scr, line_color, (i*d, 0), (i*d, HEIGHT), 2)


class Agent:
    '''the agent class '''
    def __init__(self, scr):
        self.w = WIDTH//NC
        self.h = HEIGHT//NR
        self.idx = [14, 2]
        self.x = self.idx[1]*self.w
        self.y = self.idx[0]*self.h
        self.scr = scr
        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))

    def update(self, idx):
        self.idx = idx
        self.show(bg_color)
        pg.time.wait(TS*20)
        self.x = self.idx[1]*self.w
        self.y = self.idx[0]*self.h
        self.show(agent_color)

    def show(self, color):
        self.my_rect = pg.Rect((self.x+2,self.y+2), (self.w-2, self.h-2))
        pg.draw.rect(self.scr, color, self.my_rect)

    def is_move_valid(self, a):
        '''checking for the validity of moves'''
        if (0 <= self.idx[0]+a[0] <= NC-1) and (0 <= self.idx[1]+a[1] <= NR-1):
            if Coll[self.idx[0]+a[0], self.idx[1]+a[1]] != 1:
                return True
        else:
            return False

    def reward(self):
        if Coll[self.idx[0], self.idx[1]] == 3:
            return 0
        elif Coll[self.idx[0], self.idx[1]] == 2:
            return -5.0
        else:
            return -0.1

    def move(self, a):
        '''move the agent'''
        if self.is_move_valid(a):
            pg.time.wait(TS)
            self.show(bg_color)
            self.idx[0] += a[0]
            self.idx[1] += a[1]
            self.x = self.idx[1]*self.w
            self.y = self.idx[0]*self.h
            self.show(agent_color)
            print("state: "+str(self.idx)+" --- reward: "+str(self.reward()))            

def main(mode,T=[]):
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)
    pg.init() # initialize pygame
    screen = pg.display.set_mode((WIDTH, HEIGHT))   # set up the screen

    pg.display.set_caption("COMP4600/5500: Small Gridworld")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    agent = Agent(screen)                               # instantiate an agent
    agent.show(agent_color)
    pg.display.flip()
    if mode == 'keyboard':
        run = True
        while run:
            clock.tick(60)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                    agent.move(RIGHT)
                elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                    agent.move(LEFT)
                elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                    agent.move(UP)
                elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                    agent.move(DOWN)

            screen.blit(bg, (0,0))
            draw_grid(screen)
            agent.show(agent_color)
            pg.display.flip()
            pg.display.update()
        pg.quit()

    elif mode=='trajectory':
        for i in range(len(T)):
            s = T[i]
            agent.update(s)
            screen.blit(bg, (0,0))
            draw_grid(screen)
            agent.show(agent_color)
            pg.display.flip()
            pg.display.update()
        pg.quit()        


if __name__ == "__main__":
    #main(mode='keyboard')
    main(mode='trajectory', T=[[10, 2], [10, 3], [11, 3], [10, 3], [11, 3], [11, 4], [10, 4], [10, 3], [9, 3], [9, 2], [8, 2]])