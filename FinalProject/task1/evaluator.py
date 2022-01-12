####### for evaluating because of the variables in both small and big environment have the same name
#### so denpends on what evaluation polts you need you should command and uncommand import
#from env_small import *
#from env_large import map
######################

import numpy as np
import matplotlib.pyplot as plt
import h5py


from env_small import *
#from env_large import map
#from gridworld_template import map
import statistics as st
# from learner_template import *

def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))


def epsilonGreedy(q, action, epsilon=0.05):
    if np.random.random() < epsilon:
        return np.random.randint(len(action))
    else:
        return greedy(q)
# agent = Agent(screen)

START = (19,10)  # start state
GOAL = (8,8)
# Actions
RIGHT = 0
LEFT = 1
FORWARD = 2
BACKWARD = 3
ACTIONS = [RIGHT, LEFT, FORWARD, BACKWARD]
nA = len(ACTIONS)
gamma = 0.99



def states(s):
    if map[tuple(s)]== 4 or map[tuple(s)]== 1:
        a= (np.random.choice([1,-1]),np.random.choice([1,-1]))
        return a
    else:
        return (0,0)


def transition(s, a):
    s_t = states(s)
    global gameOver

    nr = s[0]
    nc = s[1]

    if (nr + s_t[0] >= np.shape(map)[0] or nc + s_t[1] >= np.shape(map)[1] or nr + s_t[0] < 0 or nc + s_t[1] < 0):
        s_t = (0, 0)
        S_prime = s

    if a == 0:
        S_prime = (nr + s_t[0], min(np.shape(map)[1] - 1, abs(nc + 1 + s_t[1])))
    elif a == 1:
        S_prime = (nr + s_t[0], max(0, nc - 1 + s_t[1]))
    elif a == 3:
        S_prime = (max(0, nr - 1 + s_t[0]), nc + s_t[1])
    elif a == 2:
        S_prime = (min(np.shape(map)[0] - 1, abs(nr + 1 + s_t[0])), nc + s_t[1])
    else:
        S_prime = s
    if map[S_prime] == 2:
        S_prime = s
    if S_prime == GOAL:
        return S_prime, 100
    elif map[S_prime] == 1:
        return S_prime, -2
    elif map[S_prime] == 4:
        return S_prime, -3
    elif map[S_prime] == 5:
        return S_prime, -6
    elif map[S_prime] == 6:
        gameOver = True
        return S_prime, -100
    else:
        return S_prime, -1

large = h5py.File('LargeEvalution.hdf5', 'r')
#TrajectoryL= large['Trajectory']
Rlarg = large['rewards']
stepLarge = large['steps']
Qlarge = large['Q']


small = h5py.File('samllEvaluation.hdf5', 'r')
#TrajectoryS = small['Trajectory']
Rsmall = small['rewards']
stepSmall = small['steps']
Qsmall = small['Q']



def play_task(startPoistion, init_idx):
    R = []
    ######## chosing the smal/large environment####
    #optimalQ = Qlarge
    optimalQ = Qsmall
    s = startPoistion
    while True:
        a = greedy(optimalQ[s[0], s[1], :, init_idx])
        # train_animate(s, a, wait_time=0.1)
        sp, re = transition(s, a)
        R.append(re)
        print(sp)
        s = sp

        if map[sp[0], sp[1]] == 3:
            break
    return R


idx = np.random.randint(len(np.asarray((np.where(map == 0)[0]))))
startState = tuple(np.asarray((np.where(map == 0)[0], np.where(map == 0)[1]))[:, idx])

iHistory = []
rewards = []
for ss in range(100):
    print( str(ss+1) + ': ' + str(startState ))
    if startState  in iHistory:
        idx = np.random.randint(len(np.asarray((np.where(map == 0)[0]))))
        startState  = tuple(np.asarray((np.where(map == 0)[0], np.where(map == 0)[1]))[:, idx])
    else:
        iHistory.append(startState )
    reward = play_task(startState, idx)
    rewards.append(np.sum(reward))

plt.plot(range(1, 101), np.ones(100)*100, label='max reward', color='green')
plt.plot(range(1, 101), rewards, label='rewards', color='blue')
plt.xlabel('number tests')
plt.ylabel('total reward values (100)')
plt.title('Task 1 (Large environment)')
plt.legend()
plt.show()
large.close()
small.close()


###### computing, Standard Deviation, Minimum Error, Maximum Error, Mean Squared Error
#std = st.stdev(rewards, 100.0)
#minError = np.min(np.subtract(rewards, np.ones(100)*100.0))
#maxError = np.max(np.subtract(rewards, np.ones(100)*100.0))
#mse = np.square(np.subtract(np.ones(100)*100.0, rewards)).mean()
#print('Standard Deviation = ', std, '\n', 'Minimum Error = ', minError, '\n', 'Maximum Error = ', maxError, '\