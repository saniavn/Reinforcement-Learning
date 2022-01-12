import numpy as np
import matplotlib.pyplot as plt
import h5py


runs = 50
numOfepisode = 500


large = h5py.File('largeEvn1.hdf5', 'r')
rewardsL = large['rewards']
stepsL = large['steps']

small = h5py.File('smalEnv1.hdf5', 'r')
rewardsS = small['rewards']
stepsS = small['steps']

plt.figure('Q-Learning over 50 runs-rewards')
plt.plot(range(1,numOfepisode + 1), np.cumsum(np.mean(np.asarray(rewardsS), axis=0)), color='blue', label='small environment')
plt.plot(range(1, numOfepisode + 1), np.cumsum(np.mean(np.asarray(rewardsL), axis=0)), color='red', label='large environment')
plt.ylabel('average rewards over 50 runs')

plt.xlabel('episodes')


plt.figure('Q-Learning over 50 runs-steps')
plt.plot(range(1, numOfepisode + 1), np.mean(np.asarray(stepsS), axis=0), color='blue', label='small environment')
plt.plot(range(1,numOfepisode + 1), np.mean(np.asarray(stepsL), axis=0), color='red', label='large environment')
plt.ylabel('average steps over 50 runs')
plt.xlabel('episodes')
plt.legend()
plt.show()

large.close()
small.close()