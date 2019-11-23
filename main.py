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
from sampler import generateAndSaveStandardFiles
from pathGenerator import generateRandomPath

def main():
    # Recreate the standard csvs
    generateAndSaveStandardFiles()
    qBuilder = QBuilder()
    # For each dataset, learn Q(s,a)
    learn(qBuilder)    
    # Get the path to test on
    path = np.squeeze(pd.read_csv("Test/testRandomPath.csv"))
    # Initialize state (change of gradient, d2goal)
    initialState = (path[0], 10)
    # Initialize policy and rewards collected to store and compare
    optimumRewards = []
    randomRewards = []
    optimumPolicy = []
    randomPolicy = []
    # Initialize state before testing 
    state = initialState
    # Get first action
    optimumAction = getOptimumAction(qBuilder.getQ(), state)
    randomAction = np.random.randint(1,10)
    for i in range(1, len(path)):
        # Bin the d2goal i.e mapping from (100,1) to (10,1)
        d2GoalBinned = int(round((state[1] - 1)/10))
        # Next state
        state = (path[i], d2GoalBinned)
        # Compute Rewards for PREVIOUS action and this state since previous actions is current velocity (this will change)
        # Store them as well
        optimumRewards.append(CalculateReward(state, optimumAction))
        randomRewards.append(CalculateReward(state, randomAction))
        # Get the optimum action given the state
        optimumAction = getOptimumAction(qBuilder.getQ(), state)
        # Random policy samples action randomly
        randomAction = np.random.randint(1,10)
        # For storage of rewards 
        optimumPolicy.append(optimumAction)
        randomPolicy.append(randomAction)
    # Print and plot results
    print("Sum of rewards by following Optimal Policy: ",sum(optimumRewards))
    print("Sum of rewards by following Random Policy: ",sum(randomRewards))
    plt.figure()
    plt.plot(path)
    plt.plot(optimumPolicy)
    plt.plot(randomPolicy)
    plt.legend(["Path Change Of Grads", "Optimum Vel", "Random Vel"])
    plt.show()

def learn(qBuilder):
    numOfPasses = 3
    for j in range(numOfPasses):
        print(f"Learning.... {j/numOfPasses*100}%")
        for i in range(1,101):
            filename = f"Standard/standardSamplesConstantVelocity{i}.csv"
            data = np.squeeze(pd.read_csv(filename))
            qBuilder.learnFromDataQLearning(data)
        pass
    return




if __name__ == "__main__":
    main()