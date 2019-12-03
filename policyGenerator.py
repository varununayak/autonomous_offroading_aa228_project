#!usr/bin/env/python3
'''
Generates a random policy
'''

import numpy as np
import matplotlib.pyplot as plt 
from params import *

'''
generateConstantVelocity profile

inputs:
velocityMagnitude (integer)
'''
def generateConstantVelocityProfile(velocityMagnitude, length = PATH_LENGTH):
    velocityMagnitude = np.clip(velocityMagnitude, MIN_VEL, MAX_VEL)
    velocityMagnitude = int(velocityMagnitude)
    return np.multiply(velocityMagnitude, np.ones((length,), dtype = int))

def generateRandomVelocityProfile(length = PATH_LENGTH):
    return np.random.randint(MIN_VEL, MAX_VEL, length)

# For debugging only
if __name__ == "__main__":
    print(generateRandomVelocityProfile())


