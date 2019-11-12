import numpy as np
import pandas as pd
import time

class BuildQ(object):
    """ Represents a class to build a dictionary Q with the key as (s,a)"""
    def __init__(self, data):
        self.Q = {}
        self.data = data

	def GetQ(self,s,a):
		if (s,a) not in self.Q:
			return 0
		else:
			return self.Q[(s,a)]

	def ReturnQ(self, num_iters, alpha, gamma):
		#alpha = 0.1
		#gamma = 1
		s_t = np.zeros(2)
		s_p = np.zeros(2)
		for k in range(num_iters):
			for i in range(self.data.shape[0]-1):
				s_t[0],s_t[1], a_t, r_t, s_p[0], s_p[1] = self.data.values[i]
				_, _, a_t1, _, _, _ = self.data.values[i+1]
				self.Q[(s_t,a_t)] = self.GetQ(Q,s_t,a_t) + alpha*(r_t + gamma*self.GetQ(Q,s_p,a_t1) - self.GetQ(Q,s_t,a_t))
		return self.Q

