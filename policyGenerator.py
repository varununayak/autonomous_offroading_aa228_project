#!usr/bin/env/python3
'''
Generates a random policy
'''

import numpy as np
import matplotlib.pyplot as plt 

'''
generateConstantVelocity profile

inputs:
velocityMagnitude (integer)
'''
def generateConstantVelocityProfile(velocityMagnitude, length = 100):
    minVel = 1
    maxVel = 10
    velocityMagnitude = np.clip(velocityMagnitude, minVel, maxVel)
    velocityMagnitude = int(velocityMagnitude)
    return np.multiply(velocityMagnitude, np.ones((length,), dtype = int))

def generateRandomVelocityProfile(length = 100):
    minVel = 1
    maxVel = 10
    return np.random.randint(minVel, maxVel, length)

# For debugging only
if __name__ == "__main__":
    print(generateRandomVelocityProfile())


