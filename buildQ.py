import numpy as np
import pandas as pd
import time


class QBuilder(object):
    """ Represents a class to build a dictionary Q with the key as (s,a)"""

    def __init__(self):
        self.Q = {}

    def getQValue(self, s, a):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])))
        a = int(round(a))		
        if (s, a) not in self.Q:
           return 0
        else:
            return self.Q[(s, a)]
        pass

    def learnFromDataSARSA(self, data, num_iters=100, alpha=0.5, gamma=0.7):
        s_t = np.zeros(3, dtype=int)
        s_p = np.zeros(3, dtype=int)
        for k in range(num_iters):
            for i in range(data.shape[0]-1):
                s_t[0], s_t[1], s_t[2], a_t, r_t, s_p[0], s_p[1], s_p[2] = data.values[i]
                _, _,_, a_t1, _, _,_, _ = data.values[i+1]
                s_t_key = (int(round(s_t[0])), int(round(s_t[1])), int(round(s_t[2])))
                a_t_key = int(round(a_t))
                self.Q[(s_t_key, a_t_key)] = self.getQValue(s_t, a_t) + alpha * \
                        (r_t + gamma*self.getQValue(s_p, a_t1) - self.getQValue(s_t, a_t))
            pass
        pass
    
    def getMaxOverAQValue(self, s):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])))
        maxQOverA = -100000
        for a in range(1,11):
            if (s, a) in self.Q:
                value = self.Q[(s,a)]
                if value > maxQOverA:
                    maxQOverA = value
                pass
            pass
        return maxQOverA            
    
    def learnFromDataQLearning(self, data, num_iters=100, alpha=0.5, gamma=0.7):
        s_t = np.zeros(3, dtype=int)
        s_p = np.zeros(3, dtype=int)
        for k in range(num_iters):
            for i in range(data.shape[0]):
                s_t[0], s_t[1], s_t[2], a_t, r_t, s_p[0], s_p[1], s_p[2] = data.values[i]
                s_t_key = (int(round(s_t[0])), int(round(s_t[1])), int(round(s_t[2])))
                a_t_key = int(round(a_t))
                self.Q[(s_t_key, a_t_key)] = self.getQValue(s_t, a_t) + alpha * \
                        (r_t + gamma*self.getMaxOverAQValue(s_p) - self.getQValue(s_t, a_t))
            pass
        pass

    def getQ(self):
        return self.Q
