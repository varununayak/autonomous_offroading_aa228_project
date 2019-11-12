#Function for the reward model
import numpy as np

def CalculateReward(Snext_t, Vnext):
	#Snext_t is the next state vector which has [Del2H, d_goal]
	#Weights for the terms in reward
	a1 = 0.5
	a2 = 0.5
	d_goal = Snext_t[1]
	Del2H = Snext_t[0]
	reward = a1*d_goal*np.abs(Vnext) + a2*Del2H/np.abs(Vnext)
	return reward