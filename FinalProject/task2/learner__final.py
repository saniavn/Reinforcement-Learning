'''
This is the learner_template.py file that implements
a RL components for two 2D gridworld environments as
part of the final project in the COMP4600/5500-Reinforcement Learning
course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 12/6/2021
'''
import numpy as np
import matplotlib.pyplot as plt

import h5py
import small_gridworld_final as sm
import large_gridworld_final as lg


numOfEpisode = 500
RUNS = 50
gamma = .99
ENV = 1 #0 for small and 1 for large
# Collision matrix for the small environment
Coll_small = np.array([[0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 3, 0, 2],
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

# Collision matrix for the large environment
Coll_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 2, 2, 2, 2, 2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

Colls = [Coll_small, Coll_large]
Coll = Colls[ENV]

# Actions
RIGHT = [0, 1]
LEFT = [0, -1]
UP = [-1, 0]
DOWN = [1, 0]
ACTIONS = [UP, DOWN, RIGHT, LEFT]
NA = len(ACTIONS)
ACTION_IDX = [0, 1, 2, 3]
NC, NR = np.shape(Coll)

forbidden_areas = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5],
                   [23, 3], [23, 4], [23, 5], [24, 3], [24, 4], [24, 5], [25, 3], [25, 4], [25, 5],
                   [3, 23], [3, 24], [3, 25], [4, 23], [4, 24], [4, 25], [5, 23], [5, 24], [5, 25],
                   [23, 23], [23, 24], [23, 25], [24, 23], [24, 24], [24, 25], [25, 23], [25, 24], [25, 25]]



def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))


def epsilonGreedy(q, actions, epsilon=0.05):
    if np.random.random() < epsilon:
        idx = np.random.randint(len(actions))
    else:
        idx = greedy(q)
    return idx


def transition(s, a):
    '''transition function'''
    # check for stochasticity
    if Coll[s[0], s[1]] == 4:
        a_idx = ACTIONS.index(a)
        p = 0.1*np.ones(NA)
        p[a_idx] = 0.7
        a_idx = np.random.choice(ACTION_IDX, p=p)
        a = ACTIONS[a_idx]

    sp = [0, 0]
    if (0 <= s[0]+a[0] <= NC-1) and (0 <= s[1]+a[1] <= NR-1):
        if Coll[s[0]+a[0], s[1]+a[1]] != 1:
            sp[0] = s[0]+a[0]
            sp[1] = s[1]+a[1]
            return sp
        else:
            return s
    else:
        return s

def reward(s):
    '''reward function'''
    if Coll[s[0], s[1]] == 3:
        return 0
    elif Coll[s[0], s[1]] == 2:
        return -5.0
    else:
        return -0.1

def random_agent(s0, num_steps=10):
    '''this is a random walker
    your smart algorithm will replace this'''
    s = s0
    T = [s0]
    R = [reward(s)]
    for i in range(num_steps):
        a_idx = np.random.choice(ACTION_IDX)
        a = ACTIONS[a_idx]
        sp = transition(s, a)
        print(sp)
        re = reward(sp)
        R.append(re)
        T.append(sp)
        s = sp
    return T, R


def q_learning(alpha=0.5, epsilon=0.1):
    IState = np.asarray((np.where(Coll == 0)[0], np.where(Coll == 0)[1]))

    IStateSize = len(IState[0])
    Q = np.zeros((NR, NC, NA))
    QStart = np.zeros((NR, NC, NA, IStateSize))

    steps = []
    rewards = []
    # for init in range(IStateSize):
    init = 5
    initStart = tuple(IState[:, init])
    print('start' + str(init + 1) + ' /' + str(IStateSize) + ' ' + str(initStart))
    # if ENV and (list(initStart) in forbidden_areas):
    #     print('Forbidden Area!')
    #     continue
    for e in range(numOfEpisode):
        s = initStart
        T = [s]
        R = []
        t_step = 0
        while True:
            t_step += 1
            a_idx = epsilonGreedy(Q[s[0], s[1], :], ACTION_IDX, epsilon)
            a = ACTIONS[a_idx]
            sp = transition(s, a)
            T.append(sp)
            re = reward(sp)
            R.append(re)
            Q[s[0], s[1], a_idx] += alpha * (re + gamma * np.max(Q[sp[0], sp[1], :]) - Q[s[0], s[1], a_idx])
            s = sp
            if Coll[sp[0], sp[1]] == 3:
                Q[sp[0], sp[1], :] = 0
                steps.append(t_step)
                rewards.append(sum(R))
                print(str(e+ 1) + '/' + str(numOfEpisode) + ' done in' + str(t_step + 1) + ' steps')
                break
    QStart[:, :, :, init] = Q
    return T, rewards, steps, QStart


def animate():
    '''
    a function that can pass information to the
    pygame gridworld environment for visualizing
    agent's moves
    '''
    pass


if __name__=="__main__":
    trajectory, rewards, steps, Q = q_learning(alpha=.5, epsilon=.1)
    #
    # r= []
    # numrun = []
    # trajectory, rewards, steps, Q = q_learning(alpha=.5, epsilon=.1)
    # for run in range(RUNS):
    #     print(str(run+1) + ' /' + str(RUNS))
    #     trajecory, rewards, steps, Q = q_learning(alpha=.5, epsilon=.1)
    #     r.append(rewards)
    #     numrun.append(steps)

# animation
    lg.main('trajectory', T=trajectory)
    # sm.main('trajectory', T=trajectory)
#
#     data = h5py.File("smallFinal.hdf5", "w")
#     data.create_dataset('QStart', data=Q)

#ploting cumulative Rewrad and steps
#     plt.figure()
#     plt.title('Task 2 small grid')
#     plt.plot(np.asarray(rewards), label='Q-learning', color='blue')
#     plt.ylabel('accumulated rewards per episode')
#     plt.xlabel('episodes')
#     plt.figure()
#     plt.title('Task 2 small grid')
#     plt.plot(steps, color='blue')
#     plt.ylabel('number of steps')
#     plt.xlabel('episodes')
#     plt.show()
#
