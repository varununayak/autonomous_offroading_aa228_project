import numpy as np
import pandas as pd
import time

def ImportData(infile):
	data = pd.read_csv(infile)
	return data

def GetQ(Q,s,a):
	if (s,a) not in Q:
		return 0
	else:
		return Q[(s,a)]

def BuildQ(data):
	num_iters = 20
	Q = {}
	alpha = 0.1
	gamma = 1
	s_t = np.zeros(2)
	s_p = np.zeros(2)
	for k in range(num_iters):
		for i in range(data.shape[0]-1):
			s_t[0],s_t[1], a_t, r_t, s_p[0], s_p[1] = data.values[i]
			_, _, a_t1, _, _, _ = data.values[i+1]
			Q[(s_t,a_t)] = GetQ(Q,s_t,a_t) + alpha*(r_t + gamma*GetQ(Q,s_p,a_t1) - GetQ(Q,s_t,a_t))
			
	return Q

