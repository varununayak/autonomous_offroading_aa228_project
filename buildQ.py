import numpy as np
import pandas as pd
import time
import itertools

def toKey(val):
    return int(round(val))

def space(x, n):
    """
Returns a numpy array of all permutations of
states in the state space, row by row. 
"""
    return np.array([i for i in itertools.product(x, repeat = n)])

def selectNRandomStates(stateSpace):
    N = 1000
    idx = np.random.randint(len(stateSpace), size = N)
    return stateSpace[idx, :]

class QBuilderModelBased(object):
    """ Class to build state-action value function using max likelihood model-based approach"""
    def __init__(self):
        self.Q = {}
        self.T = {}
        self.R = {}
    
    def getQValue(self, s, a):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])), int(round(s[3])))
        a = int(round(a))		
        if (s, a) not in self.Q:
           return 0
        else:
            return self.Q[(s, a)]
        pass
    
    def getMaxOverAQValue(self, s):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])), int(round(s[3])))
        maxQOverA = -100000
        for a in range(1,11):
            if (s, a) in self.Q:
                value = self.Q[(s,a)]
                if value > maxQOverA:
                    maxQOverA = value
                pass
            pass
        return maxQOverA 

    def learnFromData(self, data, num_iters = 10, gamma = 0.7):
        N = {}
        rho = {}
        for k in range(num_iters):
            for i in range(data.shape[0] - 1):
                s_t0, s_t1, s_t2, s_t3, a_t, r_t, s_p0, s_p1, s_p2, s_p3 = data.values[i]
                stkey = (toKey(s_t0), toKey(s_t1),  toKey(s_t2), toKey(s_t3))
                atkey = toKey(a_t)
                st1key = (toKey(s_p0), toKey(s_p1),  toKey(s_p2), toKey(s_p3))
                thisNKey = (stkey, atkey, st1key)
                if thisNKey in N.keys():
                    N[thisNKey] += 1
                else:
                    N[thisNKey] = 1
                thisRhoKey = (stkey, atkey)
                if thisRhoKey in rho.keys():
                    rho[thisRhoKey] += r_t
                else:
                    rho[thisRhoKey] = r_t
                Nsa = 1 # initialize with one for laplace smoothing
                stateSpace = np.array(space([1,2,3,4,5,6,7,8,9,10],4), dtype = int)
                for state in stateSpace:
                    state = tuple(state)
                    if (stkey, atkey, state) in N.keys():
                        Nsa += N[(stkey, atkey, state)]
                    pass
                self.T[thisNKey] = N[thisNKey]/Nsa
                self.R[thisRhoKey] = rho[thisRhoKey]/Nsa
                sumTerm = 0
                # Randomized updates
                nRandomStates = selectNRandomStates(stateSpace)
                for state in nRandomStates:
                    state = tuple(state)
                    Tkey = (stkey, atkey, state)
                    if Tkey in self.T.keys():
                        sumTerm += self.T[Tkey]*self.getMaxOverAQValue(state)
                    pass
                self.Q[thisRhoKey] = r_t + gamma*sumTerm
            pass
        pass
        
    def getQ(self):
        return self.Q
                

class QBuilder(object):
    """ Represents a class to build a dictionary Q with the key as (s,a)"""

    def __init__(self):
        self.Q = {}

    def getQValue(self, s, a):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])), int(round(s[3])))
        a = int(round(a))		
        if (s, a) not in self.Q:
           return 0
        else:
            return self.Q[(s, a)]
        pass

    def learnFromDataSARSA(self, data, num_iters = 100, alpha=0.5, gamma=0.7):
        s_t = np.zeros(4, dtype=int)
        s_p = np.zeros(4, dtype=int)
        for k in range(num_iters):
            for i in range(data.shape[0]-1):
                s_t[0], s_t[1], s_t[2], s_t[3], a_t, r_t, s_p[0], s_p[1], s_p[2], s_p[3] = data.values[i]
                _, _,_,_, a_t1, _,_, _,_, _ = data.values[i+1]
                s_t_key = (int(round(s_t[0])), int(round(s_t[1])), int(round(s_t[2])), int(round(s_t[3])))
                a_t_key = int(round(a_t))
                self.Q[(s_t_key, a_t_key)] = self.getQValue(s_t, a_t) + alpha * \
                        (r_t + gamma*self.getQValue(s_p, a_t1) - self.getQValue(s_t, a_t))
            pass
        pass
    
    def getMaxOverAQValue(self, s):
        s = (int(round(s[0])), int(round(s[1])), int(round(s[2])), int(round(s[3])))
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
        s_t = np.zeros(4, dtype=int)
        s_p = np.zeros(4, dtype=int)
        for k in range(num_iters):
            for i in range(data.shape[0]):
                s_t[0], s_t[1], s_t[2], s_t[3], a_t, r_t, s_p[0], s_p[1], s_p[2], s_p[3] = data.values[i]
                s_t_key = (int(round(s_t[0])), int(round(s_t[1])), int(round(s_t[2])), int(round(s_t[3])))
                a_t_key = int(round(a_t))
                self.Q[(s_t_key, a_t_key)] = self.getQValue(s_t, a_t) + alpha * \
                        (r_t + gamma*self.getMaxOverAQValue(s_p) - self.getQValue(s_t, a_t))
            pass
        pass

    def getQ(self):
        return self.Q
