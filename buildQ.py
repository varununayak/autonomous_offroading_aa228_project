import numpy as np
import pandas as pd
import time


class QBuilder(object):
	""" Represents a class to build a dictionary Q with the key as (s,a)"""

	def __init__(self):
		self.Q = {}

	def getQValue(self, s, a):
		s = (int(s[0]), int(s[1]))		
		if (s, a) not in self.Q:
		   return 0
		else:
			return self.Q[(s, a)]
		pass

	def learnFromData(self, data, num_iters=100, alpha=0.1, gamma=1):
		s_t = np.zeros(2, dtype=int)
		s_p = np.zeros(2, dtype=int)
		for k in range(num_iters):
			for i in range(data.shape[0]-1):
				s_t[0], s_t[1], a_t, r_t, s_p[0], s_p[1] = data.values[i]
				_, _, a_t1, _, _, _ = data.values[i+1]
				s_t_key = (s_t[0], s_t[1])
				self.Q[(s_t_key, a_t)] = self.getQValue(s_t, a_t) + alpha * \
						(r_t + gamma*self.getQValue(s_p, a_t1) - self.getQValue(s_t, a_t))
			pass
		pass

	def getQ(self):
		return self.Q
