import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def softmax(x, temp):
    x= np.array(x)
    return np.exp(x/temp) / np.sum(np.exp(x / temp))

class NArmedBandit:
    def __init__(self, n, mean, sd, sample_noise):
        self.n = n
        self.sample_noise = sample_noise
        self.true_values = np.random.normal(mean, sd, n)
        #print(self.true_values)
    
    #generates an observation of the true value (noisy)
    def sample_val(self, i):
        return np.random.normal(self.true_values[i], self.sample_noise)


class MovingNArmedBandit(NArmedBandit):
    def __init__(self, n, mean, sd, sample_noise, move_sd):
        super().__init__(n, mean, sd, sample_noise)
        self.move_sd = move_sd
    
    def sample_val(self, i):
        #print(self.true_values[i])
        return np.random.normal(self.true_values[i], self.sample_noise)
        

    def random_walk(self):
        for i in range(len(self.true_values)):
            self.true_values[i] += np.random.normal(0, self.move_sd)

class Solver:
    def __init__(self, n, bandit):
        self.n = n
        self.vals = [[] for _ in range(n)] #stores recorded reward from every play of every state
        self.bandit = bandit

        self._total_reward = 0
        self._rewards_collected = []

    #returns the sample average value of the previously seen predicitons
    def get_sample_avg(self, i):
        if (len(self.vals[i]) == 0):
            return 0 #default
        else:
            return sum(self.vals[i])/len(self.vals[i])

    def add_observation(self, i, val):
        self.vals[i].append(val)

class incrementalSolver(Solver):
    def __init__(self, n, bandit):
        super().__init__(n, bandit)
        self.vals = [{'k': 0, 'Qk': 0} for _ in range(n)]
    
     #returns the sample average value of the previously seen predicitons
    def get_sample_avg(self, i):
        return self.vals[i]['Qk']

    def add_observation(self, i, val):
        qk = self.vals[i]['Qk']
        k = self.vals[i]['k']

        self.vals[i]['Qk'] = qk + (1/(k+1))*(val - qk)
        self.vals[i]['k'] += 1

class EpGreedy(Solver):
    def __init__(self, n, Ep, bandit):
        super().__init__(n, bandit)
        self.Ep = Ep
    
    def make_move(self):
        #decide if explore or exploit
        if (np.random.random() > self.Ep):
            #exploit - Take the move with the highest sample_avg_score
            move_to_make = np.argmax(list(map(self.get_sample_avg, range(0, self.n))), 0)
        else:
            #explore
            move_to_make = np.random.randint(0, self.n)
        
        #make the move
        #sample the chosen index, add to observations
        sampled_val = self.bandit.sample_val(move_to_make)
        self._total_reward += sampled_val
        self._rewards_collected.append(sampled_val)
        #print(move_to_make)
        self.add_observation(move_to_make, sampled_val)

class softMax(Solver):
    def __init__(self, n, temp, bandit):
        super().__init__(n, bandit)
        self.temp = temp
    
    def make_move(self):
        #pick the move to make
        sm = softmax(list(map(self.get_sample_avg, range(0, self.n))), temp=self.temp)
        move_to_make = np.random.choice(list(range(0, self.n)), p=sm)
        
        #make the move
        #sample the chosen index, add to observations
        sampled_val = self.bandit.sample_val(move_to_make)
        self._total_reward += sampled_val
        self._rewards_collected.append(sampled_val)
        #print(move_to_make)
        self.add_observation(move_to_make, sampled_val)

class incrementalEpGreedySampleAvg(incrementalSolver):
    def __init__(self, n, Ep, bandit):
        super().__init__(n, bandit)
        self.Ep = Ep

    def make_move(self):
        #decide if explore or exploit
        if (np.random.random() > self.Ep):
            #exploit - Take the move with the highest sample_avg_score
            move_to_make = np.argmax(list(map(self.get_sample_avg, range(0, self.n))), 0)
        else:
            #explore
            move_to_make = np.random.randint(0, self.n)
        
        #make the move
        #sample the chosen index, add to observations
        sampled_val = self.bandit.sample_val(move_to_make)
        self._total_reward += sampled_val
        self._rewards_collected.append(sampled_val)
        #print(move_to_make)
        self.add_observation(move_to_make, sampled_val)

class incrementalEpGreedyConstantWeight(incrementalSolver):
    def __init__(self, n, Ep, weight, bandit):
        super().__init__(n, bandit)
        self.Ep = Ep        
        self.weight = weight

    def make_move(self):
        #decide if explore or exploit
        if (np.random.random() > self.Ep):
            #exploit - Take the move with the highest sample_avg_score
            move_to_make = np.argmax(list(map(self.get_sample_avg, range(0, self.n))), 0)
        else:
            #explore
            move_to_make = np.random.randint(0, self.n)
        
        #make the move
        #sample the chosen index, add to observations
        sampled_val = self.bandit.sample_val(move_to_make)
        self._total_reward += sampled_val
        self._rewards_collected.append(sampled_val)
        #print(move_to_make)
        self.add_observation(move_to_make, sampled_val)
    
    def add_observation(self, i, val):
        qk = self.vals[i]['Qk']
        k = self.vals[i]['k']

        self.vals[i]['Qk'] = qk + self.weight*(val - qk)
        self.vals[i]['k'] += 1


fig, ax = plt.subplots()  # Create a figure containing a single axes.

es01_rewards = []
ea01_rewards = []
for j in range(0, 200):
    if (j%100 == 0): print(j)
    #b = NArmedBandit(10, 0, 1, 1)
    mb = MovingNArmedBandit(10, 0, 1, 1, 1)
    es01 = incrementalEpGreedySampleAvg(10, 0.1, mb)
    ea01 = incrementalEpGreedyConstantWeight(10, 0.1, 0.1, mb)
    for i in range(0, 5000):
        es01.make_move()
        ea01.make_move()
        mb.random_walk()
    es01_rewards.append(es01._rewards_collected)
    ea01_rewards.append(ea01._rewards_collected)
    

print(np.array(es01_rewards).shape)

plt.title("n-armed-bandit, n=10, Q*(a)~N(0, 1), Q_t(a)~N(Q*(a), 1), moving+~N(0, 1)")
plt.plot(np.mean(es01_rewards, 0), label="epsilon=0.1, sample average")
plt.plot(np.mean(ea01_rewards, 0), label="epsilon=0.1, alpha=0.1")
plt.legend()
plt.show()