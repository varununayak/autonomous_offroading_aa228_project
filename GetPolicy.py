import pandas as pd
import numpy as np

def GetPolicy(Q, num_states):
	policy = []	
	for s in range(1,num_states):
		Q_max = float('-inf')
		for st,at in Q.keys():
			if (s == st):
				Q_val = Q[st,at]
				if (Q_val > Q_max):
					optimal_action = at
					Q_max = Q_val
		policy.append(optimal_action)

	return policy