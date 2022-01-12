'''
This is the large_gridworld_final.py file that implements 
a large gridworld environment as part of the final project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 12/6/2021
'''
import numpy as np
import pygame as pg

# Collision matrix
Coll = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 4, 4, 2, 2, 2, 4, 4, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 4, 2, 2, 2, 2, 2, 4, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 2, 4, 4, 4, 4, 4, 2, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 2, 2, 2, 1, 0, 0, 1, 4, 2, 4, 4, 4, 4, 4, 2, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
'''
t= 0 
for i in range(30):
    for j in range(30):
        if Coll[i,j] == 1:
            t+=1

print(t)
'''
RIGHT = [0, 1]
LEFT = [0, -1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, RIGHT, LEFT]
NA = len(ACTIONS)
ACTION_IDX = [0, 1, 2, 3]

#Constants
WIDTH = 900     # width of the environment (px)
HEIGHT = 900    # height of the environment (px)
TS = 10         # delay in msec
NC, NR = np.shape(Coll)

# define colors
goal_color = pg.Color(0, 100, 0)
bad_color = pg.Color(100, 0, 0)
bg_color = pg.Color(0, 0, 0)
line_color = pg.Color(128, 128, 128)
agent_color = pg.Color(120,120,0)
wall_color = pg.Color(140, 140, 140)
ice_color = pg.Color(0, 0, 100)

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
            elif idx == 4: #ice
                pg.draw.rect(scr, ice_color, (j*d, i*d ,d, d))            

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
        self.idx = [0, 0]
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
        if Coll[self.idx[0], self.idx[1]] == 4:
            a_idx = ACTIONS.index(a)
            p = 0.1*np.ones(NA)
            p[a_idx] = 0.7
            a_idx = np.random.choice(ACTION_IDX, p=p)
            a = ACTIONS[a_idx]        
        if (0 <= self.idx[0]+a[0] <= NC-1) and (0 <= self.idx[1]+a[1] <= NR-1):
            if Coll[self.idx[0]+a[0], self.idx[1]+a[1]] != 1:
                self.idx[0] += a[0]
                self.idx[1] += a[1]
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
            self.x = self.idx[1]*self.w
            self.y = self.idx[0]*self.h
            self.show(agent_color)
            print("state: "+str(self.idx)+" --- reward: "+str(self.reward()))            

def animate(mode, T=[]):
    pg.init() # initialize pygame
    screen = pg.display.set_mode((WIDTH, HEIGHT))   # set up the screen
    pg.display.set_caption("COMP4600/5500: Large Gridworld")              # add a caption
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
    #animate(mode='keyboard')
    animate(mode='trajectory', T=[[10, 2], [10, 1], [11, 1], [12, 1], [12, 0], [12, 0], [12, 0], [13, 0], [13, 1], [13, 2], [13, 2]])