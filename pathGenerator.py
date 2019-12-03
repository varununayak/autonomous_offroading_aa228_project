#!usr/bin/env/python3
'''
Generates a random change of gradient np array
'''

import numpy as np
import matplotlib.pyplot as plt 
from params import *

'''
generateRandomPath(lenth)

inputs:
length of path in meters, default 100 

returns tuple of size three containing:
numpy array of change of gradients at intervals of 1 meter
stepSize (default is 1m)
totalLength (default is 100)
'''
def generateRandomPath(length = PATH_LENGTH):
    minHeight = PATH_MIN_HEIGHT
    maxHeight = PATH_MAX_HEIGHT # implies that max change of gradient can be 2 times this value
    stepSize = STEP_SIZE
    totalLength = length
    heights = np.random.randint(minHeight, maxHeight, length + 2)
    gradients = np.zeros((length + 1,), dtype = int)
    for i in range(length + 1):
        gradients[i] = heights[i+1] - heights[i]
    changeOfGradients = np.zeros((length,), dtype = int)
    for i in range(length):
        changeOfGradients[i] = gradients[i+1] - gradients[i]
    randomPath = np.abs(changeOfGradients)
    return (randomPath, stepSize, totalLength)

# For debugging only
if __name__ == "__main__":
    for i in range(1, NUM_RANDOM_PATHS + 1):
        randomPath, stepSize, totalLength = generateRandomPath()
        np.savetxt(f"Standard/standardRandomPath{i}.csv", randomPath)
    print(randomPath, stepSize, totalLength)
    plt.plot(randomPath)
    plt.show()
