#Function for the reward model
import numpy as np

def CalculateReward(s, a):
    usingKshitijsFunction = True
    Del2H1 = s[1]
    Del2H2 = s[2]
    curr_vel = s[0]
    d_goal = s[3]
    if (usingKshitijsFunction):
        a1 = 0.5
        a2 = 1.0
        a3 = 0.8
        if curr_vel == 0:
            return 0
        # penalized current velocity as well as intended velocity using the current and next gradient
        reward = a1*d_goal*np.abs(a) - (a2*np.abs(Del2H1))/np.abs(curr_vel) - (a3*np.abs(Del2H2))/np.abs(a)
    else:
        a1 = 0.05
        a2 = 0.5
        reward = a1*d_goal*np.abs(a) + a2*Del2H/np.abs(a)
    return reward