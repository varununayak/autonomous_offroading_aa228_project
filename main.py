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

def main():
    # Recreate the standard csvs
    generateAndSaveStandardFiles()
    qBuilder = QBuilder()
    # For each dataset, learn Q(s,a)
    for i in range(1,11):
        filename = f"Standard/standardSamplesConstantVelocity{i}.csv"
        data = np.squeeze(pd.read_csv(filename))
        qBuilder.learnFromData(data)

    # Get the path to test on
    path = np.squeeze(pd.read_csv("Standard/standardRandomPath.csv"))
    # Initialize state
    initialState = (path[0], 100)
    optimumRewards = []
    randomRewards = []
    optimumPolicy = []
    randomPolicy = []
    state = initialState
    for i in range(1, len(path)):
        prevState = state
        # Next state
        state = (path[i], state[1]-1)
        # Get the optimum action given the state
        optimumAction = getOptimumAction(qBuilder.getQ(), state)
        randomAction = np.random.randint(1,10)
        optimumPolicy.append(optimumAction)
        randomPolicy.append(randomAction)
        # Compute Rewards
        optimumRewards.append(CalculateReward(prevState, optimumAction))
        randomRewards.append(CalculateReward(prevState, randomAction))
    print("Sum of rewards by following Optimal Policy: ",sum(optimumRewards))
    print("Sum of rewards by following Random Policy: ",sum(randomRewards))
    plt.figure()
    plt.plot(path)
    plt.plot(optimumPolicy)
    plt.plot(randomPolicy)
    plt.legend(["Path Change Of Grads", "Optimum Vel", "Random Vel"])
    plt.show()





if __name__ == "__main__":
    main()