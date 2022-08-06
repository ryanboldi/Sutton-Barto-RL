from math import exp, factorial

facts = [factorial(n) for n in range(0, 300)]

class poisson:
    def __init__(self, lmda, ep=0.001):
        self.lmda = lmda
        self.ep = ep #values with prob less than this don't happen and are normalized out
        self.values = {}

        #using markov bound, we know that P(X >= a) <= E[X]/a
        # need to set ep >= P(X >= a) >= E[X]/a ====> ep = E[X]/a =====> a = E[X] / ep = lambda/ep
        #good upper bound

        for i in range(0, int(self.lmda / self.ep)):
            if(self.prob(i) >= self.ep):
                self.values[i] = self.prob(i)
            else:
                break #if one is smaller than ep, all the rest will be
        

        #normalize
        tot = sum(self.values.values())
        for key in self.values.keys():
            self.values[key] = self.values[key]/tot

    def prob(self, n):
        return ((self.lmda**n)/(facts[n]))*exp(-self.lmda)

class rentalPolicy:
    def __init__(self):
        self.carPark1 = 20
        self.carPark2 = 20
        
        self.actions = {}
        for i in range(0, self.carPark1 + 1):
            for j in range(0, self.carPark2 + 1):
                self.actions[(i, j)] = 0


class rentalEnv1:
    def __init__(self):
        self.cars1 = 20
        self.cars2 = 20
        self.gamma = 0.9

        self.car1RequestPoisson = poisson(3)
        self.car1ReturnPoisson = poisson(3)

        self.car2RequestPoisson = poisson(4)
        self.car2ReturnPoisson = poisson(2)

        self.stateValues = {}
        for i in range(0, self.cars1 + 1):
            for j in range(0, self.cars2 + 1):
                self.stateValues[(i, j)] = 0

    def valueOfState(self, carsLot1, carsLot2, action):
        rew = 0

        rew -= 2*action

        #don't add cars beyond the limit
        carsLot1 = min(self.cars1, carsLot1 - action)
        carsLot2 = min(self.cars2, carsLot1 + action)

        #iterate over all 4 random variables
        for rq1, prq1 in self.car1RequestPoisson.values.items():
            for rt1, prt1 in self.car1ReturnPoisson.values.items():
                for rq2, prq2 in self.car2RequestPoisson.values.items():
                    for rt2, prt2 in self.car2ReturnPoisson.values.items():
                        event_rew = 0
                        zeta = prq1 * prt1 * prq2 * prt2 #chance of this event happening

                        #calculate the reward for this specific combo
                        rented_cars1 = min(carsLot1, rq1)
                        rented_cars2 = min(carsLot2, rq2)

                        event_rew += rented_cars1*10
                        event_rew += rented_cars2*10

                        #new state for value of next state
                        new_cars1 = max(min(self.cars1, carsLot1 + rt1 - rented_cars1), 0)
                        new_cars2 = max(min(self.cars2, carsLot2 + rt2 - rented_cars2), 0)

                        rew += zeta * (event_rew + self.gamma * self.stateValues[(new_cars1, new_cars2)])
        return rew

    def policy_eval(self, policy):
        theta = 50
        while(True):
            delta = 0
            for one in range(0, self.cars1 + 1):
                for two in range(0, self.cars2 + 1):
                    val = self.stateValues[(one, two)]
                    act = self.valueOfState(one, two, policy.actions[(one, two)])
                    self.stateValues[(one, two)] = act
                    #print("val: " + str(val) + ", act: " + str(act))
                    delta = max(delta, abs(val - act))
            print("d" + str(delta))
            #print(self.stateValues)
            if delta < theta:
                break

    def policy_improvement(self, policy):
        policy_copy = policy.actions.copy()
        policy_stable = True
        for one in range(0, self.cars1 + 1):
            for two in range(0, self.cars2 + 1):
                b = policy_copy[(one, two)]
                
                best_val = 0
                best_act = None
                #find best action from all possible actions
                for i in range(-min(two, 5), min(one, 5)):
                    v = self.valueOfState(one, two, i)
                    if v > best_val:
                        v = best_val
                        best_act = i
                policy_copy[(one, two)] = best_act

                if b != policy_copy[(one, two)]:
                    policy_stable = False
        return policy_stable, policy_copy
                

policy = rentalPolicy()
env = rentalEnv1()

while(True):
    print(1)
    env.policy_eval(policy)
    np, stop = env.policy_improvement(policy)
    policy.actions = np
    if stop:
        break

#plotting
policyPlot = [[0 for i in range(0, 21)] for j in range(0, 21)]
for i in range(0, 21):
    for j in range(0, 21):
        policyPlot[i][j] = policy.actions[(i, j)]

import matplotlib.pyplot as plt
import numpy as np

a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()