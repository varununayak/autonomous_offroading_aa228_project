import pandas as pd
import numpy as np

def getOptimumAction(Q, currentState):
    # Initialize optimum action to random
    optimumAction = np.random.randint(1,10)
    # Look for maximum among all state action pairs that match current state  
    Q_max = float('-inf')
    currentState = tuple(currentState)
    for (state, action) in Q.keys():
        if (state == currentState):
            Q_val = Q[(state,action)]
            if (Q_val > Q_max):
                optimumAction = action
                Q_max = Q_val
            pass
        pass
    return int(optimumAction)