import numpy as np
import matplotlib.pyplot as plt
from pathGenerator.py import generateRandomPath
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

	for index in range(len(path)-1):
		s_next = [pathArray[index+1],(totalLength-((index+1)*stepSize))]
		v_next = velocityData[index+1]
		current_reward = CalculateReward(s_next, v_next)
		S.append([pathArray[index],(totalLength-((index)*stepSize))])
		A.append(velocityData[index+1])
		R.append(current_reward)
		Sp.append(s_next)



def main():
	pathData = generateRandomPath()
	sample_generator(pathData,velocityData)


if __name__	== "__main__":
	main()