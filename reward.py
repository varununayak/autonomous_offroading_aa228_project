#Function for the reward model
import numpy as np

def CalculateReward(s, a):
    usingKshitijsFunction = True
    d_goal = s[3]
    Del2H = s[1]
    if (usingKshitijsFunction):
        a1 = 0.5
        a2 = 0.9
        reward = a1*d_goal*np.abs(a) - (a2*np.abs(Del2H))/np.abs(a)
    else:
        a1 = 0.05
        a2 = 0.5
        reward = a1*d_goal*np.abs(a) + a2*Del2H/np.abs(a)
    return reward