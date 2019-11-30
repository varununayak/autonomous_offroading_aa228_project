#!usr/bin/env/python3
'''
Main function for calling functions
that learn the policy and testing agains some random policy
'''
import numpy as np
import pandas as pd
from buildQ import *
from getOptimumAction import *
from reward import *
import matplotlib.pyplot as plt
from sampler import generateAndSaveStandardFiles, getNextVelocity
from pathGenerator import generateRandomPath

def main():
    # Recreate the standard csvs
    generateAndSaveStandardFiles()
    qBuilder = QBuilder()
    # For each dataset, learn Q(s,a)
    learn(qBuilder)    
    # Get the path to test on
    path = np.squeeze(pd.read_csv("Test/testRandomPath.csv"))
    totalLength = 100
    stepSize = 1
    # Initialize policy and rewards collected to store and compare
    optimumRewards = []
    randomRewards = []
    optimumPolicy = []
    randomPolicy = []
    optimumVelocity = []
    randomVelocity = []
    # Initialize state before testing 
    initialState = [0, int(round(path[0])), int(round(path[1])), 10]
    s = initialState
    # Simulate for Optimum Policy
    for index in range(1, len(path) - 2):
        a = getOptimumAction(qBuilder.getQ(), s)
         # Bin the d2Goal before adding to state
        d2GoalBinnedNext = int(round((totalLength-((index+1)*stepSize))/10))
        # Compute next state (next velocity is a result of current velocity and current action)
        sNext = [getNextVelocity(s[0], a), int(round(path[index + 1])), int(round(path[index + 2])), d2GoalBinnedNext]
        r = CalculateReward(s, a)
        s = sNext
        optimumRewards.append(r)
        optimumPolicy.append(a)
        optimumVelocity.append(s[0])
    # Simulate for Random Policy
    s = initialState
    for index in range(1, len(path) - 2):
        a = np.random.randint(1,10)
         # Bin the d2Goal before adding to state
        d2GoalBinnedNext = int(round((totalLength-((index+1)*stepSize))/10))
        # Compute next state (next velocity is a result of current velocity and current action)
        sNext = [getNextVelocity(s[0], a), int(round(path[index + 1])), int(round(path[index + 2])), d2GoalBinnedNext]
        r = CalculateReward(s, a)
        s = sNext
        randomRewards.append(r)
        randomPolicy.append(a)
        randomVelocity.append(s[0])
    # Print and plot results
    print("Sum of rewards by following Optimal Policy: ",sum(optimumRewards))
    print("Sum of rewards by following Random Policy: ",sum(randomRewards))
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(path)
    plt.plot(optimumVelocity)
    plt.legend(["Path Change Of Grads","Optimum Velocity"])
    plt.subplot(2,1,2)
    plt.plot(path)
    plt.plot(randomVelocity)
    plt.legend(["Path Change Of Grads","Rand Velocity"])
    plt.show()

def learn(qBuilder):
    numOfPasses = 4
    for j in range(numOfPasses):
        print(f"Learning.... {j/numOfPasses*100}%")
        for i in range(1,101):
            filename = f"Standard/standardSamples{i}.csv"
            data = np.squeeze(pd.read_csv(filename))
            qBuilder.learnFromDataSARSA(data)
        pass
    return




if __name__ == "__main__":
    main()