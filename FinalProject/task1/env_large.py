'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Late modified: 10/19/2021
'''
import numpy as np
import pygame as pg
import time

#Constants
WIDTH = 800    # width of the environment (px)
HEIGHT = 800    # height of the environment (px)
TS = 10         # delay in msec
NR = 40      # number of cells in the environment
NC = 40
map=np.zeros((NR,NC))

map[39,39]=6
map[7,7] =6
map[10,10]=6
map[0,0]=1

map[0,3]=1
map[0,4]=4

map[0,18]=2
map[0,19]=2
map[1,18]=2
map[1,19]=2

map[4,10]=2
map[4,11]=2

map[4,13]=2
map[4,14]=2
map[4,15]=2

map[5,13]=2
map[5,14]=2
map[5,15]=2

map[5,20]=1
map[5,21]=2

map[6,16]=2
map[6,17]=2
map[6,18]=2
map[7,16]=2
map[7,17]=2
map[7,18]=2

map[8,2]=1
map[8,7]=1
map[8,8]=3
map[9,8]=1
map[9,7]=4

map[9,9]=4

map[8,9]=1
map[8,11]=1
map[8,12]=1

map[9,18]=2
map[9,17]=2
map[10,17]=2
map[10,18]=2

map[11,1]=1
map[11,8]=1
map[11,10]=1

map[12,0]=1

map[12, 8]=4
map[12,7]=4

map[13,3]=2
map[13,4]=2
map[13,5]=2
map[13,6]=2
map[14,3]=2
map[14,4]=2
map[14,5]=2
map[14,6]=2
map[15,3]=2
map[15,4]=1
map[15,5]=2
map[15,6]=2

map[15,7]=4
map[15,9]=1

map[13,11]=2
map[13,12]=2
map[13,13]=2
map[14,11]=2
map[14,12]=2
map[14,13]=2
map[15,11]=2
map[15,12]=2
map[15,13]=2



map[19,0]=4
map[19,1]=4
map[18,0]=4
map[18,1]=1

map[19,15]=2
map[19,14]=2
map[19,13]=2
map[19,12]=2


map[18,7]=1
map[17,9]=1

for i in (np.arange(25, NR - 10)):
    for j in np.arange(25, NR - 10):
        map[j,i]=4


#map[20,30]=1
map[35,10]=1
map[35,12]=1

map[37,14]=4
map[38,15]=4
map[38,18]=4

map[15,33]=1
map[15,38]=1

map[21,12]=1
map[22,14]=1
map[23,13]=1
map[21,30]=1

#for i in np.arange(2, 3):
#    for j in np.arange(19, 20):
 #       map[j,i]=4

for i in np.arange(9, 12):
    for j in np.arange(30, 33):
        map[j,i]=1

for i in np.arange(30, 33):
    for j in np.arange(9, 12):
        map[j,i]=1


for i in np.arange(25, 30):
    for j in np.arange(0,5 ):
        map[j,i]=2



for i in (np.arange(30,NR-2)):
    for j in np.arange(30,NR-2):
        map[j,i]=1

for i in np.arange(17, 21):
    for j in np.arange(21, 24):
        map[j,i]=5


for i in np.arange(0, 5):
    for j in np.arange(35, 40):
        map[j,i]=2

for i in np.arange(24, 26):
    for j in np.arange(12, 16):
        map[j,i]=5

for i in np.arange(0, 7):
    for j in np.arange(27, 29):
        map[j,i]=5

for i in np.arange(0, 4):
    for j in np.arange(19, 20):
        map[j,i]=4

for i in np.arange(25, 27):
    for j in np.arange(19, 22):
        map[j,i]=4
for i in np.arange(35, 37):
    for j in np.arange(19, 22):
        map[j,i]=2

for i in (np.arange(0, 4)):
    for j in np.arange(36, 39):
        map[j,i]=1

for i in (np.arange(8, 13)):
    for j in np.arange(34, 40):
         map[j,i]=4

for i in (np.arange(9, 12)):
    for j in np.arange(28, 32):
        map[j, i] = 1

for i in np.arange(19, 22):
    for j in np.arange(13, 16):
        map[j, i] = 1

for i in np.arange(19, 24):
     for j in np.arange(36, 40):
         map[j, i] = 4


for i in np.arange(33, 36):

    for j in np.arange(25,28):
        map[j,i]= 5

bg_color = pg.Color(250, 250, 250)
line_color = pg.Color(128, 128, 128)

def draw_grid(scr):
    '''a function to draw gridlines and other objects'''

    for i in range(NR+1):
        pg.draw.line(scr, line_color,(0,i*WIDTH//NR),(WIDTH, i*WIDTH//NR), 2)

    for j in range(NC+1):
        pg.draw.line(scr, line_color, (j*WIDTH//NC,0), (j*WIDTH//NR, HEIGHT), 2)

    # Vertical lines
        for k in range(0, WIDTH, WIDTH//NC):
            for b in range(0, HEIGHT, HEIGHT//NR):
                rect = pg.Rect(k,b, WIDTH//NC,HEIGHT//NR)
                pg.draw.rect(scr,bg_color , rect,1)

class Agent:
    '''the agent class '''
    def __init__(self, scr):
        self.w = WIDTH//(NR)
        self.h = WIDTH//(NC)
        self.x = 10*self.w
        self.y = HEIGHT - self.h
        self.scr = scr
        self.my_rect = pg.Rect((self.x, self.y), (self.w, self.h))

    def show(self, color):
        self.my_rect = pg.Rect((self.x,self.y), (self.w, self.h))
        pg.draw.rect(self.scr, color, self.my_rect)

    def is_move_validC(self, a):
        '''checking for the validity of moves'''
        if 0 <= self.x + a < WIDTH:
            return True
        else:
            return False

    def is_move_validR(self, a):
        '''checking for the validity of moves'''
        #if
        if 0 <= self.y + a < HEIGHT:
            return True
        else:
            return False

    def moveC(self, a):
        '''move the agent'''
        s_n=(np.random.choice([1,-1]), np.random.choice([1,-1]))
        s_t= (np.random.choice([1,-1]), np.random.choice([1,-1]))

        if self.is_move_validC(a) and map[int((self.y)/(HEIGHT//NR)), int((self.x+a)/(WIDTH//NC))]!=2:

            pg.time.wait(TS)

            #self.show(bg_color)
            if (map[int((self.y)/(HEIGHT//NR)), int((self.x+a)/(WIDTH//NC))]==1 and
                self.is_move_validC(s_n[0] * (WIDTH//NC)) and self.is_move_validR(s_n[1] * (WIDTH//NC))):
                self.x += (s_n[0]* (WIDTH//NC))
                self.y += (s_n[1]* (WIDTH//NC))
            if (map[int((self.y)/(HEIGHT//NR)), int((self.x+a)/(WIDTH//NC))]==4 and
                self.is_move_validC(s_t[0] * (WIDTH//NC)) and self.is_move_validR(s_t[1] * (WIDTH//NC))):
                self.x += (s_t[0]* (WIDTH//NC))
                self.y += (s_t[1]* (WIDTH//NC))
            else:
                self.x += a
                #self.show(agent_color)
                self.show(bg_color)



    def moveR(self, a):
        '''move the agent'''
        if self.is_move_validR(a) and map[int((self.y+a)/(HEIGHT//NR)), int((self.x)/(WIDTH//NC))]!=2:
            pg.time.wait(TS)
            #self.show(bg_color)
            self.y += a
            #self.show(agent_color)
            self.show(bg_color)

def main():
    tree=pg.image.load('tree07.png')
    tree1=pg.transform.scale(tree, (WIDTH//NC,HEIGHT//NR))

    water=pg.image.load('water.png')
    water1=pg.transform.scale(water, (WIDTH//NC,HEIGHT//NR))
    ##cat
    cat = pg.image.load('cat.png')
    cat1 = pg.transform.scale(cat, (WIDTH // NC, HEIGHT // NR))



    #### obstacles#####
    lava=pg.image.load('rock.png')
    lava1=pg.transform.scale(lava, (WIDTH//NC,HEIGHT//NR))

    #### snake###
    snake=pg.image.load('snake.png')
    snake1= pg.transform.scale(snake, (WIDTH // NC, HEIGHT // NR))

    ###agent##
    Agnt=pg.image.load('down.png')
    Agnt1=pg.transform.scale(Agnt, (WIDTH//NC,HEIGHT//NR))

    ####Goal###
    Goal=pg.image.load('bag_of_berries.png')
    Goal1=pg.transform.scale(Goal, (WIDTH//NC,HEIGHT//NR))


    pg.init() # initialize pygame
    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Large environment")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    agent = Agent(screen)                               # instantiate an agent
    # agent.show(agent_color)
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                agent.show(bg_color)
                agent.moveC(WIDTH//NC)
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                agent.show(bg_color)
                agent.moveC(-WIDTH//NC)
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                agent.show(bg_color)
                agent.moveR(-HEIGHT//NR)
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                agent.show(bg_color)
                agent.moveR(HEIGHT//NR)

        ### mushrooms negative points###
        screen.blit(tree1, (1,4))
        screen.blit(tree1, (2*WIDTH//NC,8*HEIGHT//NR))
        #screen.blit(tree1, (15*WIDTH//NC,8*HEIGHT//NR))
        screen.blit(tree1, (9*WIDTH//NC,8*HEIGHT//NR))
        screen.blit(tree1, (10*WIDTH//NC,11*HEIGHT//NR))
        screen.blit(tree1, (20*WIDTH//NC,5*HEIGHT//NR))
        screen.blit(lava1, (21 * WIDTH // NC, 5 * HEIGHT // NR))

       # screen.blit(tree1, (18*WIDTH//NC,8*HEIGHT//NR))
        screen.blit(tree1, (11 * WIDTH // NC, 8 * HEIGHT // NR))
        screen.blit(tree1, (12 * WIDTH // NC, 8 * HEIGHT // NR))



        screen.blit(tree1, (7*WIDTH//NC,8*HEIGHT//NR))
        screen.blit(tree1, (7*WIDTH//NC,18*HEIGHT//NR))
        screen.blit(tree1, (9*WIDTH//NC,17*HEIGHT//NR))
        screen.blit(tree1, (9*WIDTH//NC,15*HEIGHT//NR))

        screen.blit(tree1, (12 * WIDTH // NC, 35 * HEIGHT // NR))
        screen.blit(tree1, (10 * WIDTH // NC, 35 * HEIGHT // NR))

        screen.blit(tree1, (20 * WIDTH // NC, 30 * HEIGHT // NR))


        screen.blit(tree1, (0*WIDTH//NC,12*HEIGHT//NR))
        screen.blit(tree1, (1*WIDTH//NC,11*HEIGHT//NR))
        screen.blit(tree1, (3* WIDTH // NC, 0* HEIGHT // NR))

        screen.blit(tree1, (8*WIDTH//NC,9*HEIGHT//NR))
        screen.blit(water1, (9*WIDTH//NC,9*HEIGHT//NR))
        screen.blit(water1, (7*WIDTH//NC,9*HEIGHT//NR))

        screen.blit(water1, (4* WIDTH// NC, 0* HEIGHT // NR))

        screen.blit(water1, (9*WIDTH//NC,12*HEIGHT//NR))
        screen.blit(water1, (8*WIDTH//NC,12*HEIGHT//NR))
        screen.blit(water1, (7*WIDTH//NC,15*HEIGHT//NR))

        screen.blit(water1, (18 * WIDTH // NC, 38 * HEIGHT // NR))
        screen.blit(water1, (15 * WIDTH // NC, 38 * HEIGHT // NR))
        screen.blit(water1, (14 * WIDTH // NC, 37 * HEIGHT // NR))

        screen.blit(tree1, (8 * WIDTH // NC, 11* HEIGHT // NR))



        ####### obstacle####

##########water
        screen.blit(water1, (0*WIDTH//NC,19*HEIGHT//NR))
        screen.blit(water1, (0*WIDTH//NC,18*HEIGHT//NR))
        screen.blit(tree1, (1*WIDTH//NC,18*HEIGHT//NR))
        screen.blit(water1, (1*WIDTH//NC,19*HEIGHT//NR))
        screen.blit(lava1, (12 * WIDTH // NC, 19 * HEIGHT // NR))
        screen.blit(lava1, (13 * WIDTH // NC, 19 * HEIGHT // NR))
        screen.blit(lava1, (14 * WIDTH // NC, 19 * HEIGHT // NR))
        screen.blit(lava1, (15* WIDTH // NC, 19 * HEIGHT // NR))

        screen.blit(tree1, (38 * WIDTH // NC, 15 * HEIGHT // NR))
        screen.blit(tree1, (33 * WIDTH // NC, 15 * HEIGHT // NR))


        ####randomTree
        screen.blit(tree1, (13 * WIDTH // NC, 23 * HEIGHT // NR))
        screen.blit(tree1, (14 * WIDTH // NC, 22 * HEIGHT // NR))
        screen.blit(tree1, (15 * WIDTH // NC, 25 * HEIGHT // NR))
        screen.blit(tree1, (12 * WIDTH // NC, 21 * HEIGHT // NR))

        screen.blit(lava1, (14*WIDTH//NC,4*HEIGHT//NR))
        screen.blit(lava1, (13*WIDTH//NC,4*HEIGHT//NR))
        #screen.blit(lava1, (12*WIDTH//NC,4*HEIGHT//NR))
        screen.blit(lava1, (11*WIDTH//NC,4*HEIGHT//NR))
        screen.blit(lava1, (10*WIDTH//NC,4*HEIGHT//NR))
       # screen.blit(lava1, (15*WIDTH//NC,4*HEIGHT//NR))

        screen.blit(lava1, (15*WIDTH//NC,5*HEIGHT//NR))
        screen.blit(lava1, (14*WIDTH//NC,5*HEIGHT//NR))
        screen.blit(lava1, (13*WIDTH//NC,5*HEIGHT//NR))

        screen.blit(lava1, (16*WIDTH//NC,6*HEIGHT//NR))
        screen.blit(lava1, (16*WIDTH//NC,7*HEIGHT//NR))
        screen.blit(lava1, (17*WIDTH//NC,7*HEIGHT//NR))
        screen.blit(lava1, (17*WIDTH//NC,6*HEIGHT//NR))
        screen.blit(lava1, (18*WIDTH//NC,6*HEIGHT//NR))
        screen.blit(lava1, (18*WIDTH//NC,7*HEIGHT//NR))
        screen.blit(lava1, (17*WIDTH//NC,9*HEIGHT//NR))
        screen.blit(lava1, (17*WIDTH//NC,10*HEIGHT//NR))
        screen.blit(lava1, (18*WIDTH//NC,10*HEIGHT//NR))
        screen.blit(lava1, (18*WIDTH//NC,9*HEIGHT//NR))

        screen.blit(lava1, (3*WIDTH//NC,15*HEIGHT//NR))
        #######Tree####
        screen.blit(tree1, (4*WIDTH//NC,15*HEIGHT//NR))
       #######################treee#################
        screen.blit(lava1, (15* WIDTH // NC, 4 * HEIGHT // NR))

        screen.blit(lava1, (5*WIDTH//NC,15*HEIGHT//NR))
        screen.blit(lava1, (6*WIDTH//NC,15*HEIGHT//NR))
        screen.blit(lava1, (11*WIDTH//NC,15*HEIGHT//NR))
        screen.blit(lava1, (12*WIDTH//NC,15*HEIGHT//NR))
        screen.blit(lava1, (13*WIDTH//NC,15*HEIGHT//NR))


        screen.blit(lava1, (3 * WIDTH // NC, 14 * HEIGHT//NR))
        screen.blit(lava1, (4 * WIDTH // NC, 14 * HEIGHT//NR))
        screen.blit(lava1, (5 * WIDTH // NC, 14 * HEIGHT//NR))
        screen.blit(lava1, (6 * WIDTH // NC, 14 * HEIGHT//NR))
        screen.blit(lava1, (11*WIDTH//NC,14*HEIGHT//NR))
        screen.blit(lava1, (12 * WIDTH // NC, 14 * HEIGHT // NR))
        screen.blit(lava1, (13 * WIDTH // NC, 14 * HEIGHT // NR))


        screen.blit(snake1, (39*WIDTH//NC,39*HEIGHT//NR))
        screen.blit(snake1, (7*WIDTH//NC,7*HEIGHT//NR))
        screen.blit(snake1, (10 * WIDTH // NC, 10 * HEIGHT // NR))


        screen.blit(lava1, (3*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (4*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (5*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (6*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (11*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (12*WIDTH//NC,13*HEIGHT//NR))
        screen.blit(lava1, (13*WIDTH//NC,13*HEIGHT//NR))

        for i in (np.arange(25, NR - 10)):
            for j in np.arange(25, NR - 10):
                screen.blit(water1, (j * WIDTH // NC, i * HEIGHT // NR))

        for i in (np.arange(0, 4)):
            for j in np.arange(36, 39):
                screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))

        for i in (np.arange(8, 13)):
            for j in np.arange(34, 40):
                screen.blit(water1, (j * WIDTH // NC, i * HEIGHT // NR))

        for i in (np.arange(9, 12)):
            for j in np.arange(28, 32):
                screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))


        for i in np.arange(25, 30):
            for j in np.arange(0, 5):
                screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in (np.arange(30, NR - 2)):
            for j in np.arange(30, NR - 2):
                screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))



        for i in np.arange(17, 21):
            for j in np.arange(21, 24):
                screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(0, 5):
            for j in np.arange(35, 40):
                screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(24, 26):
            for j in np.arange(12, 16):
                screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(0, 7):
            for j in np.arange(27, 29):
                screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(19, 22):
            for j in np.arange(13, 16):
                screen.blit(tree1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(19, 24):
            for j in np.arange(36, 40):
                screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

        #for i in np.arange(2, 4):
        #    for j in np.arange(19, 20):
         #       screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(25, 27):
            for j in np.arange(19, 22):
                screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(35, 37):
            for j in np.arange(19, 22):
                screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(9, 12):
            for j in np.arange(30,33 ):
                screen.blit(tree1, (i * WIDTH // NC, j * HEIGHT // NR))

        for i in np.arange(33, 36):
            for j in np.arange(25,28):
                screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))









        ### Agent###
        screen.blit(Agnt1, (agent.x, agent.y))
        ### Goal###
        screen.blit(Goal1, (8*WIDTH//NC,8*HEIGHT//NR))
        # screen.blit(bg, (0,0))
        draw_grid(screen)
        # agent.show(agent_color)
        pg.display.flip()
        pg.display.update()
    pg.quit()

def animate(Trajectory):

    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Large environment")              # add a caption
    tree=pg.image.load('tree07.png')
    tree1=pg.transform.scale(tree, (WIDTH//NC,HEIGHT//NR))

    water=pg.image.load('water.png')
    water1=pg.transform.scale(water, (WIDTH//NC,HEIGHT//NR))

    ###cat###
    cat = pg.image.load('cat.png')
    cat1 = pg.transform.scale(cat, (WIDTH // NC, HEIGHT // NR))

    #### snake###
    snake=pg.image.load('snake.png')
    snake1= pg.transform.scale(snake, (WIDTH // NC, HEIGHT // NR))


    #### obstacles#####
    lava=pg.image.load('rock.png')
    lava1=pg.transform.scale(lava, (WIDTH//NC,HEIGHT//NR))

     ###agent##
    Agnt=pg.image.load('down.png')
    Agnt1=pg.transform.scale(Agnt, (WIDTH//NC,HEIGHT//NR))

    ####Goal###
    Goal=pg.image.load('bag_of_berries.png')
    Goal1=pg.transform.scale(Goal, (WIDTH//NC,HEIGHT//NR))


    pg.init() # initialize pygame
    #screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    #pg.display.set_caption("Saniya Vahedian Movahed")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    agent = Agent(screen)                               # instantiate an agent
    # agent.show(agent_color)
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        for state in Trajectory:

            ### mushrooms negative points###
            screen.blit(tree1, (1,4))
            screen.blit(tree1, (2*WIDTH//NC,8*HEIGHT//NR))
            #screen.blit(tree1, (15*WIDTH//NC,8*HEIGHT//NR))
            screen.blit(tree1, (9*WIDTH//NC,8*HEIGHT//NR))
            screen.blit(tree1, (10*WIDTH//NC,11*HEIGHT//NR))
            screen.blit(tree1, (20*WIDTH//NC,5*HEIGHT//NR))
            screen.blit(lava1, (21 * WIDTH // NC, 5 * HEIGHT // NR))
        #    screen.blit(tree1, (18*WIDTH//NC,8*HEIGHT//NR))
            screen.blit(tree1, (11 * WIDTH // NC, 8 * HEIGHT // NR))
            screen.blit(tree1, (12 * WIDTH // NC, 8 * HEIGHT // NR))

            screen.blit(tree1, (7*WIDTH//NC,8*HEIGHT//NR))
            screen.blit(tree1, (7*WIDTH//NC,18*HEIGHT//NR))
            screen.blit(tree1, (9*WIDTH//NC,17*HEIGHT//NR))
            screen.blit(tree1, (9*WIDTH//NC,15*HEIGHT//NR))

            screen.blit(tree1, (0*WIDTH//NC,12*HEIGHT//NR))
            screen.blit(tree1, (1*WIDTH//NC,11*HEIGHT//NR))
            screen.blit(tree1, (3* WIDTH // NC, 0* HEIGHT // NR))

            screen.blit(tree1, (8*WIDTH//NC,9*HEIGHT//NR))
            screen.blit(water1, (9*WIDTH//NC,9*HEIGHT//NR))
            screen.blit(water1, (7*WIDTH//NC,9*HEIGHT//NR))

            screen.blit(water1, (4* WIDTH// NC, 0* HEIGHT // NR))

            screen.blit(water1, (9*WIDTH//NC,12*HEIGHT//NR))
            screen.blit(water1, (8*WIDTH//NC,12*HEIGHT//NR))
            screen.blit(water1, (7*WIDTH//NC,15*HEIGHT//NR))

            screen.blit(tree1, (8 * WIDTH // NC, 11* HEIGHT // NR))

            #Zscreen.blit(tree1, (2*WIDTH//NC,15*HEIGHT//NR))
             #screen.blit(tree1, (1*WIDTH//NC,15*HEIGHT//NR))

            screen.blit(tree1, (12 * WIDTH // NC, 35 * HEIGHT // NR))
            screen.blit(tree1, (10 * WIDTH // NC, 35 * HEIGHT // NR))

            screen.blit(tree1, (20 * WIDTH // NC, 30 * HEIGHT // NR))

            ######treemiddle Lava

            screen.blit(tree1, (4 * WIDTH // NC, 15 * HEIGHT // NR))





             ### tree postive points###
             #screen.blit(tree1, (1,15))
            screen.blit(water1, (18 * WIDTH // NC, 38 * HEIGHT // NR))
            screen.blit(water1, (15 * WIDTH // NC, 38 * HEIGHT // NR))
            screen.blit(water1, (14 * WIDTH // NC, 37 * HEIGHT // NR))

            ####### obstacle####
            #screen.blit(lava1, (19*WIDTH//NC,0*HEIGHT//NR))
            #screen.blit(lava1, (19*WIDTH//NC,1*HEIGHT//NR))
            ##########water
            screen.blit(water1, (0 * WIDTH // NC, 19 * HEIGHT // NR))
            screen.blit(water1, (0 * WIDTH // NC, 18 * HEIGHT // NR))
            screen.blit(tree1, (1 * WIDTH // NC, 18 * HEIGHT // NR))
            screen.blit(water1, (1 * WIDTH // NC, 19 * HEIGHT // NR))



            screen.blit(lava1, (12 * WIDTH // NC, 19 * HEIGHT // NR))
            screen.blit(lava1, (13 * WIDTH // NC, 19 * HEIGHT // NR))
            screen.blit(lava1, (14 * WIDTH // NC, 19 * HEIGHT // NR))
            screen.blit(lava1, (15 * WIDTH // NC, 19 * HEIGHT // NR))

            screen.blit(lava1, (14*WIDTH//NC,4*HEIGHT//NR))
            screen.blit(lava1, (13*WIDTH//NC,4*HEIGHT//NR))
            #screen.blit(lava1, (12*WIDTH//NC,4*HEIGHT//NR))
            screen.blit(lava1, (11*WIDTH//NC,4*HEIGHT//NR))
            screen.blit(lava1, (10*WIDTH//NC,4*HEIGHT//NR))
            # screen.blit(lava1, (15*WIDTH//NC,4*HEIGHT//NR))

            screen.blit(lava1, (15*WIDTH//NC,5*HEIGHT//NR))
            screen.blit(lava1, (14*WIDTH//NC,5*HEIGHT//NR))
            screen.blit(lava1, (13*WIDTH//NC,5*HEIGHT//NR))

            screen.blit(lava1, (16*WIDTH//NC,6*HEIGHT//NR))
            screen.blit(lava1, (16*WIDTH//NC,7*HEIGHT//NR))
            screen.blit(lava1, (17*WIDTH//NC,7*HEIGHT//NR))
            screen.blit(lava1, (17*WIDTH//NC,6*HEIGHT//NR))
            screen.blit(lava1, (18*WIDTH//NC,6*HEIGHT//NR))
            screen.blit(lava1, (18*WIDTH//NC,7*HEIGHT//NR))
            screen.blit(lava1, (17*WIDTH//NC,9*HEIGHT//NR))
            screen.blit(lava1, (17*WIDTH//NC,10*HEIGHT//NR))
            screen.blit(lava1, (18*WIDTH//NC,10*HEIGHT//NR))
            screen.blit(lava1, (18*WIDTH//NC,9*HEIGHT//NR))

            screen.blit(lava1, (3*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(tree1, (4*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (5*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (6*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (11*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (12*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (13*WIDTH//NC,15*HEIGHT//NR))
            screen.blit(lava1, (4*WIDTH//NC,15*HEIGHT//NR))

            screen.blit(lava1, (13*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (12*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (11*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (6*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (5*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (4*WIDTH//NC,14*HEIGHT//NR))
            screen.blit(lava1, (3*WIDTH//NC,14*HEIGHT//NR))

            screen.blit(lava1, (3*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (4*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (5*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (6*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (11*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (12*WIDTH//NC,13*HEIGHT//NR))
            screen.blit(lava1, (13*WIDTH//NC,13*HEIGHT//NR))


            screen.blit(snake1, (39*WIDTH//NC,39*HEIGHT//NR))
            screen.blit(snake1, (7*WIDTH//NC,7*HEIGHT//NR))
            screen.blit(snake1, (10 * WIDTH // NC, 10 * HEIGHT // NR))

            screen.blit(lava1, (15 * WIDTH // NC, 4 * HEIGHT // NR))

            screen.blit(tree1, (38 * WIDTH // NC, 15 * HEIGHT // NR))
            screen.blit(tree1, (33 * WIDTH // NC, 15 * HEIGHT // NR))

            ####randomTree
            screen.blit(tree1, (13 * WIDTH // NC, 23 * HEIGHT // NR))
            screen.blit(tree1, (14 * WIDTH // NC, 22 * HEIGHT // NR))
            screen.blit(tree1, (15 * WIDTH // NC, 25 * HEIGHT // NR))
            screen.blit(tree1, (12 * WIDTH // NC, 21 * HEIGHT // NR))

            for i in (np.arange(25, NR - 10)):
                for j in np.arange(25, NR - 10):
                    screen.blit(water1, (j * WIDTH // NC, i * HEIGHT // NR))

            for i in (np.arange(0, 4)):
                for j in np.arange(36, 39):
                    screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))

            for i in (np.arange(8, 13)):
                for j in np.arange(34, 40):
                    screen.blit(water1, (j * WIDTH // NC, i * HEIGHT // NR))

            for i in (np.arange(9, 12)):
                for j in np.arange(28, 32):
                    screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))

            for i in np.arange(25, 30):
                for j in np.arange(0, 5):
                    screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in (np.arange(30, NR - 2)):
                for j in np.arange(30, NR - 2):
                    screen.blit(tree1, (j * WIDTH // NC, i * HEIGHT // NR))

            for i in np.arange(17, 21):
                for j in np.arange(21, 24):
                    screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(0, 5):
                for j in np.arange(35, 40):
                    screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(24, 26):
                for j in np.arange(12, 16):
                    screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(0, 7):
                for j in np.arange(27, 29):
                    screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(19, 22):
                for j in np.arange(13, 16):
                    screen.blit(tree1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(19, 24):
                for j in np.arange(36, 40):
                    screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

            # for i in np.arange(2, 4):
            #    for j in np.arange(19, 20):
            #       screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(25, 27):
                for j in np.arange(19, 22):
                    screen.blit(water1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(35, 37):
                for j in np.arange(19, 22):
                    screen.blit(lava1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(9, 12):
                for j in np.arange(30, 33):
                    screen.blit(tree1, (i * WIDTH // NC, j * HEIGHT // NR))

            for i in np.arange(33, 36):
                for j in np.arange(25, 28):
                    screen.blit(cat1, (i * WIDTH // NC, j * HEIGHT // NR))

            ### Goal###
            screen.blit(Goal1, (8*WIDTH//NC,8*HEIGHT//NR))
            # screen.blit(bg, (0,0))
            draw_grid(screen)

            # agent.show(agent_color)

            ### Agent###
            screen.blit(Agnt1, (state[1]*WIDTH//NC, state[0]*HEIGHT//NR))
            pg.display.flip()
            pg.display.update()
            #time.sleep(0.5)
            screen.blit(bg, (state[1] * WIDTH // NC, state[0] * HEIGHT // NR))

        run= False


    pg.quit()

if __name__ == "__main__":
    main()
