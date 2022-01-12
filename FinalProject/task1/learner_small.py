'''
This is the learner_template.py file that implements
a RL components for a 2D gridworld and is part of the
mid-term project in the COMP4600/5500-Reinforcement Learning
course - Fall 2021
Late modified: 10/19/2021
'''
import numpy as np
import matplotlib.pyplot as plt
from env_small import map
from env_small import animate
import h5py

START = (19,10)  # start state
GOAL = (8,8)  # goal state
# Actions
RIGHT = 0
LEFT = 1
DOWN=2
UP=3


ACTIONS = [LEFT, RIGHT, UP, DOWN]
nA = len(ACTIONS)
NR=20
NC=20
numOFepisode = 500
gamma = 1



def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))

def epsilonGreedy(q, action, epsilon=0.05):
    if np.random.random() < epsilon:
        return np.random.randint(len(action))
    else:
        return greedy(q)

### random intialization start state
def states(s):
    if map[tuple(s)]== 4 or map[tuple(s)]== 1:
        a= (np.random.choice([1,-1]),np.random.choice([1,-1]))
        return a
    else:
        return (0,0)


def transition(s, a):
    s_t=states(s)
    # S_prime=s
    #print(S_prime)

    nr=s[0]
    nc=s[1]

    if (nr+s_t[0]>= np.shape(map)[0] or nc +s_t[1] >= np.shape(map)[1] or nr+s_t[0]<0 or nc+ s_t[1]<0):
        s_t=(0,0)
        S_prime=s

    if a==0:
        S_prime = (nr + s_t[0], min(np.shape(map)[1] - 1, abs(nc + 1 + s_t[1])))
    elif a== 1:
        S_prime = (nr + s_t[0], max(0, nc - 1 + s_t[1]))
    elif a == 3:
        S_prime = (max(0, nr - 1 + s_t[0]), nc + s_t[1])
    elif a== 2:
        S_prime = (min(np.shape(map)[0] - 1, abs(nr + 1 + s_t[0])), nc + s_t[1])
    else:
        S_prime = s
    if map[S_prime] ==2:
        S_prime=s
    if S_prime== GOAL:
        return S_prime,100
    elif map[S_prime]== 1:
        return S_prime, -2
    elif map[S_prime]==4:
        return S_prime, -3
    else:
        return S_prime, -1

def random_agent():
    s = START
    T = [s]
    R = []
    while s != (8, 8):
        a = np.random.choice(ACTIONS)
        sp, re = transition(s, a)
        print(sp, re)
        R.append(re)
        T.append(sp)
        s = sp
        if s== (8, 8):
            break
    return T, R


def rl_agent(alpha=0.5, epsilon=0.1):
    startState = np.asarray((np.where(map == 0)[0], np.where(map == 0)[1]))
    startStateLen = len(startState[0])
    Q = np.zeros((NR, NC, nA))
    Qstar = np.zeros((NR, NC, nA, startStateLen))
    steps = []
    rewards = []

    #for init in range(startStateLen):
     #   print(str(init + 1) + '/ ' + str(startStateLen))
    init= 281
    startPosition = tuple(startState[:, init])

    for e in range(numOFepisode):
        s = startPosition

        T = [s]
        R = []
        A = []
        timeSteps = 0

        while True:
            timeSteps += 1

            a = epsilonGreedy(Q[s[0], s[1], :], ACTIONS, epsilon)
            A.append(a)

            sp, r = transition(s, a)
            T.append(sp)
            R.append(r)

            Q[s[0], s[1], a] += alpha * (r + gamma * np.max(Q[sp[0], sp[1], :]) - Q[s[0], s[1], a])

            s = sp

            if map[sp[0], sp[1]] == 3:
                Q[sp[0], sp[1], :] = 0
                steps.append(timeSteps)
                rewards.append(sum(R))
                print(str(e + 1) + '/' + str(numOFepisode) +
                        ' done in' + str(timeSteps + 1) + ' steps')
                break
    Qstar[:, :, :, init] = Q
    return T, rewards, steps, Qstar


if __name__=="__main__":
    # Trajectory, Reward = random_agent()
    # print(Trajectory)
    # plt.plot(Reward)
    # plt.show()
    ################## for running 50 times ################
    # steps = []
    # r = []
    # runs = 50
    # for run in range(runs):
    #    Trajectory, R, stepsQ, QL = rl_agent(alpha=0.5, epsilon=0.2)
    #    steps.append(stepsQ)
    #    r.append(R)
    ##################################################################
    Trajectory, R, stepsQ, QL = rl_agent(alpha=0.5, epsilon=0.2)
    # final = h5py.File('smalEnv1.hdf5', "w")
    # final.create_dataset('rewards', data=R)
    # final.create_dataset('steps', data=stepsQ)
    # final.create_dataset('Trajectory', data=Trajectory)
    #final.create_dataset('steps', data=steps)
    #final.create_dataset('Q', data=QL)
    # final = h5py.File('SmallPlotsNew.hdf5', "w")
    # final.create_dataset('rewards', data=r)
   #final.create_dataset('steps', data=steps)
   #final.create_dataset('steps', data=steps)
    animate(Trajectory)
    #plt.figure('Q-Learning over 50 runs-rewards')
    #plt.subplot(2, 1, 1)
    #plt.plot(np.mean(np.asarray(R), axis=0), color='blue')
   #  plt.plot(range(1, numOFepisode + 1), steps, label='DQ_learning', color='blue')
   #  plt.xlabel('episodes')
   #  plt.ylabel('average rewards')
   #  plt.legend()
   #  plt.figure('Q-Learning over 50 runs-steps')
   #
   #  plt.subplot(2, 1, 2)
   #
   # # plt.plot(np.mean(np.asarray(steps), axis=0), color='blue')
   #  plt.plot(range(1, numOFepisode + 1), np.cumsum(R), label='Double Q-learning', color='blue')
   #  plt.xlabel('episodes')
   #  plt.ylabel('average steps')
   #  plt.legend()
   #  plt.show()
   #
# plt.figure()
#     plt.subplot(2, 1, 1)
#     plt.plot(range(1, NUM_EPISODES + 1), steps_q, label='Q-learning', color='blue')
#     plt.title("Q-Learning")
#     plt.xlabel('episodes')
#     plt.ylabel('number of Steps')
#     plt.legend()
#     plt.subplot(2, 1, 2)
#     plt.plot(range(1, NUM_EPISODES + 1), np.cumsum(Reward_q), label='Q-learning', color='blue')
#     plt.title("Q-Learning")
#     plt.xlabel('episodes')
#     plt.ylabel('cumulative reward')
#
#     plt.legend()
#
#
#     plt.show()


