import numpy as np
import matplotlib.pyplot as plt

goal = 100
p = 0.55
gamma = 1

stateVals = [0 for s in range(goal+1)]
reward = [0 for _ in range(goal+1)]
reward[goal] = 1
policy = [0 for _ in range(goal+1)]

def valueOfState(state):
    #iterate over every possible future state from this position given the possible actions
    best_val = 0
    
    for a in range(0, min(state, goal-state) + 1):
        score = 0   
        #states if win or loss
        sIfWin = state + a
        sIfLoss = state - a

        score += p * (reward[sIfWin] + (gamma * stateVals[sIfWin]))
        score += (1-p) * (reward[sIfLoss] + (gamma * stateVals[sIfLoss]))

        if score > best_val:
            best_val = score 

    return best_val

def get_best_policy(state):
    #iterate over every possible future state from this position given the possible actions
    best_val = 0
    
    for a in range(0, min(state, goal-state) + 1):
        score = 0   
        #states if win or loss
        sIfWin = state + a
        sIfLoss = state - a

        score += p * (reward[sIfWin] + (gamma * stateVals[sIfWin]))
        score += (1-p) * (reward[sIfLoss] + (gamma * stateVals[sIfLoss]))

        if score > best_val:
            best_val = score 
            best_act = a

    return best_act

while(True): 
    delta = 0
    #not S+, just S
    for s in range(1, goal):
        v = stateVals[s]
        actual = valueOfState(s) 
        stateVals[s] = actual
        delta = max(delta, abs(v - actual))
    if delta < 0.00000001:
        break

for i in range(1, goal):
    policy[i] = get_best_policy(i)

print(policy)
#print(stateVals)

plt.scatter([i for i in range(goal+1)], policy)
plt.xlabel("Amount of Money ($)")
plt.ylabel("Stake (policy)")
plt.title("Gambler's Problem, p=" + str(p))
plt.show()