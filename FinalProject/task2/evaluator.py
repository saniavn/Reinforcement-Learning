import numpy as np
import matplotlib.pyplot as plt
import h5py
from learner_template_final import *
import statistics as st

###### for changing the environment you should also change the Evn variable in learner file.
def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))

def epsilonGreedy(q, action, epsilon=0.05):
    if np.random.random() < epsilon:
        return np.random.randint(len(action))
    else:
        return greedy(q)

large = h5py.File('Task2-largeenv.hdf5', 'r')
QLarge = large['Q']


small = h5py.File('Task2-smallenv.hdf5', 'r')
QSmall = small['Q']


def play_task(initial_state, init_idx):
    R = []
    optimalQ = QLarge
    #optimalQ = QSmall
    s = initial_state
    while True:
        a_idx = greedy(optimalQ[s[0], s[1], :, init_idx])
        a = ACTIONS[a_idx]
        t.append(sp)
        sp = transition(s, a)
        r = reward(sp)
        R.append(r)
        s = sp
        if len(t) > 3 and t[-1] == t[-3]:
            sindex = np.random.randint(len(np.asarray((np.where(Coll == 0)[0]))))
            s = tuple(np.asarray((np.where(Coll == 0)[0], np.where(Coll == 0)[1]))[:, sindex])
            continue
        if Coll[sp[0], sp[1]] == 3:
            break
    return R


idx = np.random.randint(len(np.asarray((np.where(Coll == 0)[0]))))
init_pose = tuple(np.asarray((np.where(Coll == 0)[0], np.where(Coll == 0)[1]))[:, idx])
iHistory = []
idxHistory = []
rewards = []

while len(iHistory) < 100:
    if (startState in iHistory) or (list(startState) in ban):
        idx = np.random.randint(len(np.asarray((np.where(Coll == 0)[0]))))
        startState = tuple(np.asarray((np.where(Coll == 0)[0], np.where(Coll == 0)[1]))[:, idx])
    else:
        iHistory.append(startState)
        idxHistory.append(idx)

for ss in range(100):
    startState = iHistory[ss]
    print(str(ss + 1) + ':' + str(startState))
    rew = play_task(startState, idxHistory[ss])
    rewards.append(np.sum(rew))



plt.plot(range(1, 101), np.zeros(100), label='maximum reward', color='green')
plt.plot(range(1, 101), rewards, label='rewards for an experiment', color='red')
plt.xlabel('number of tests')
plt.ylabel('Total Reward Values (100)')
plt.title('Task 2 (large environment)' )
plt.legend()
plt.show()
large.close()
small.close()

###### computing, Standard Deviation, Minimum Error, Maximum Error, Mean Squared Error
#std = st.stdev(rewards, 100.0)
#minError = np.min(np.subtract(rewards, np.ones(100)*100.0))
#maxError = np.max(np.subtract(rewards, np.ones(100)*100.0))
#mse = np.square(np.subtract(np.ones(100)*100.0, rewards)).mean()
#print('Standard Deviation = ', std, '\n', 'Minimum Error = ', minError, '\n', 'Maximum Error = ', maxError, '\n','Mean Squared Error = ', mse)