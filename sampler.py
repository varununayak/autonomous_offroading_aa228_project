import numpy as np
import matplotlib.pyplot as plt
from pathGenerator import *
from policyGenerator import *
from reward import *
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

	print(np.shape(S),np.shape(A),np.shape(R),np.shape(Sp))

	S = np.array(S)
	Sp = np.array(Sp)

	DataCombined = np.transpose(np.vstack((S[:,0], S[:,1],A,R,Sp[:, 0], Sp[:, 1])))
	np.savetxt("samples.csv", DataCombined, delimiter=",")

	# for 



def main():
	pathData = generateRandomPath()
	velocityData = generateConstantVelocityProfile(1)
	sample_generator(pathData,velocityData)


if __name__	== "__main__":
	main()