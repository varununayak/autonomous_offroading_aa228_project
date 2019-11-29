#!usr/bin/env/python3

import numpy as np
import matplotlib.pyplot as plt
from pathGenerator import *
from policyGenerator import *
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

    for index in range(len(pathArray)-1):
        # Bin the d2Goal before adding to state
        d2GoalBinnedNext = int(round((totalLength-((index+1)*stepSize))/10))
        # Create next state
        sNext = [pathArray[index+1], d2GoalBinnedNext]
        # Current velocity 
        vCurrent = velocityData[index]
        # Compute reward given current desired velocity (which is next actual velocity) THIS WILL CHANGE in future
        nextReward = CalculateReward(sNext, vCurrent)
        # Bin the d2Goal before adding to state
        d2GoalBinned = int(round((totalLength-((index)*stepSize))/10))
        # Append all required variables to list
        S.append([pathArray[index], d2GoalBinned])
        A.append(vCurrent)
        R.append(nextReward)
        Sp.append(sNext)

    S = np.array(S)
    Sp = np.array(Sp)

    DataCombined = np.transpose(np.vstack((S[:,0], S[:,1],A,R,Sp[:, 0], Sp[:, 1])))
    np.savetxt(f"Standard/standardSamplesConstantVelocity{saveIdx}.csv", DataCombined, delimiter=",")

def generateAndSaveStandardFiles():
    saveIdx = 1
    for i in range(1,11):
        pathData = (np.squeeze(pd.read_csv(f"Standard/standardRandomPath{i}.csv")), 1, 100)
        for velocity in range(1,11):
            velocityData = generateRandomVelocityProfile()
            sample_generator(pathData,velocityData, saveIdx)
            saveIdx += 1
        pass
    pass

def getNextVelocity(V, Vdes):
	cap_range = 2 		#+/- range to cap velocity
	new_V = np.clip(Vdes, a_min = V - cap_range, a_max = V + cap_range)
	noise = random.randint(-1,1)
	return (new_V + noise)

def main():
    generateAndSaveStandardFiles()

if __name__	== "__main__":
    main()