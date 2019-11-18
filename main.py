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
    # Initialize state
    initialState = (path[0], 100)
    optimumRewards = []
    randomRewards = []
    optimumPolicy = []
    randomPolicy = []
    state = initialState
    optimumAction = getOptimumAction(qBuilder.getQ(), state)
    randomAction = np.random.randint(1,10)
    for i in range(1, len(path)):
        # Next state
        state = (path[i], state[1]-1)
        # Compute Rewards for previous action and this state
        optimumRewards.append(CalculateReward(state, optimumAction))
        randomRewards.append(CalculateReward(state, randomAction))
        # Get the optimum action given the state
        optimumAction = getOptimumAction(qBuilder.getQ(), state)
        randomAction = np.random.randint(1,10)
        # For storage of rewards
        optimumPolicy.append(optimumAction)
        randomPolicy.append(randomAction)
    print("Sum of rewards by following Optimal Policy: ",sum(optimumRewards))
    print("Sum of rewards by following Random Policy: ",sum(randomRewards))
    plt.figure()
    plt.plot(path)
    plt.plot(optimumPolicy)
    plt.plot(randomPolicy)
    plt.legend(["Path Change Of Grads", "Optimum Vel", "Random Vel"])
    plt.show()

def learn(qBuilder):
    numOfPasses = 10
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