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


fig, ax = plt.subplots()  # Create a figure containing a single axes.

s1_rewards = []
s01_rewards = []
s001_rewards = []
for j in range(0, 2000):
    b = NArmedBandit(10, 0, 1, 1)
    s1 = softMax(10, 1, b)
    s01 = softMax(10, 0.1, b)
    s001 = softMax(10, 0.01, b)
    print(j)
    for i in range(0, 1000):
        s1.make_move()
        s01.make_move()
        s001.make_move()
    s1_rewards.append(s1._rewards_collected)
    s01_rewards.append(s01._rewards_collected)
    s001_rewards.append(s001._rewards_collected)

plt.title("n-armed-bandit, n=10, Q*(a)~N(0, 1), Q_t(a)~N(Q*(a), 1)")
plt.plot(np.mean(s1_rewards, 0), label="Softmax (T = 1)")
plt.plot(np.mean(s01_rewards, 0), label="Softmax T=0.1")
plt.plot(np.mean(s001_rewards, 0), label="Softmax T=0.01")
plt.legend()
plt.show()

#print(e._rewards_collected)