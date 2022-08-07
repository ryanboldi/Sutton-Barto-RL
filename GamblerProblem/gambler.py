import random

goal = 100
p = 0.4

states = [0 for i in range(1, goal)]
stateVals = [0 for s in range(0, goal+1)]
stateVals[goal+1] = 1

def valueOfState(state, action):
    global p, goal

    #iterate over every possible future state from this position given the possible actions
        #get p, reward

    #heads
    if random.random() < 0.4:
        #if heads, state is incremented by action
        state = max(goal, state - action)
    else:
        state = max(0, state - action)
    

