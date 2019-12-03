#!usr/bin/env/python3

import numpy as np
import matplotlib.pyplot as plt
from pathGenerator import *
from policyGenerator import *
from params import *
from reward import *
import random
import pandas as pd
'''
DATA SAMPLER

This program take a path and reward function and returns samples to be used for learning
'''

def sample_generator(pathData,velocityData, saveIdx):
    pathArray,stepSize,totalLength = pathData
    S = []
    A = []
    R = []
    Sp = []

    # State format = [velocity, thisGradient, nextGradient, d2GoalBinned]
    initialState = [0, pathArray[0], pathArray[1], D2GOAL_BIN_RES]    # Start with zero velocity
    s = initialState
    for index in range(len(pathArray) - 2):
        a = velocityData[index]
         # Bin the d2Goal before adding to state
        d2GoalBinnedNext = int(round((totalLength-((index+1)*stepSize))/10))
        # Compute next state (next velocity is a result of current velocity and current action)
        sNext = [getNextVelocity(s[0], a), pathArray[index + 1], pathArray[index + 2], d2GoalBinnedNext]
        r = CalculateReward(s, a) 
        S.append(s)
        A.append(a)
        R.append(r)
        Sp.append(sNext)
        # Set s to sNext before next iteration
        s = sNext

    S = np.array(S)
    Sp = np.array(Sp)

    DataCombined = np.transpose(np.vstack((S[:,0],S[:,1],S[:,2],S[:,3],A,R,Sp[:,0],Sp[:,1],Sp[:,2],Sp[:,3])))
    np.savetxt(f"Standard/standardSamples{saveIdx}.csv", DataCombined, delimiter=",")

def generateAndSaveStandardFiles():
    saveIdx = 1
    for i in range(1,NUM_RANDOM_PATHS + 1):
        pathData = (np.squeeze(pd.read_csv(f"Standard/standardRandomPath{i}.csv")), PATH_STEP_SIZE, PATH_LENGTH)
        for velocity in range(1,NUM_RANDOM_VELOCITY_PROFILES + 1):
            velocityData = generateRandomVelocityProfile()
            sample_generator(pathData,velocityData, saveIdx)
            saveIdx += 1
        pass
    pass

def getNextVelocity(V, Vdes):
	cap_range = 2 		#+/- range to cap velocity
	new_V = np.clip(Vdes, a_min = V - cap_range, a_max = V + cap_range)
	noise = random.randint(-VELOCITY_DYNAMICS_NOISE_AMP, VELOCITY_DYNAMICS_NOISE_AMP)
	return np.clip(new_V + noise, a_min = MIN_VEL, a_max = MAX_VEL)

def main():
    generateAndSaveStandardFiles()

if __name__	== "__main__":
    main()