#!usr/bin/env/python3

import numpy as np
import matplotlib.pyplot as plt
from pathGenerator import *
from policyGenerator import *
from reward import *
import pandas as pd
'''
DATA SAMPLER

This program take a path and reward function and returns samples to be used for learning
'''

def sample_generator(pathData,velocityData):
    pathArray,stepSize,totalLength = pathData
    S = []
    A = []
    R = []
    Sp = []

    for index in range(len(pathArray)-1):
        s_next = [pathArray[index+1],(totalLength-((index+1)*stepSize))]
        v_next = velocityData[index+1]
        current_reward = CalculateReward(s_next, v_next)
        S.append([pathArray[index],(totalLength-((index)*stepSize))])
        A.append(velocityData[index+1])
        R.append(current_reward)
        Sp.append(s_next)

    S = np.array(S)
    Sp = np.array(Sp)

    DataCombined = np.transpose(np.vstack((S[:,0], S[:,1],A,R,Sp[:, 0], Sp[:, 1])))
    np.savetxt(f"Standard/standardSamplesConstantVelocity{velocityData[0]}.csv", DataCombined, delimiter=",")

def generateAndSaveStandardFiles():
    pathData = (np.squeeze(pd.read_csv("Standard/standardRandomPath.csv")), 1, 100)
    for velocity in range(1,11):
        velocityData = generateConstantVelocityProfile(velocity)
        sample_generator(pathData,velocityData)
    pass


def main():
    generateAndSaveStandardFiles()

if __name__	== "__main__":
    main()