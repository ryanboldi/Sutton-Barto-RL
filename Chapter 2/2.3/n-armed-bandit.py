import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class NArmedBandit:
    def __init__(self, n, mean, sd, sample_noise):
        self.n = n
        self.sample_noise = sample_noise
        self.true_values = np.random.normal(mean, sd, n)
        #print(self.true_values)
    
    #generates an observation of the true value (noisy)
    def sample_val(self, i):
        return np.random.normal(self.true_values[i], self.sample_noise)


class EpGreedy:
    def __init__(self, n, Ep, bandit):
        self.n = n
        self.Ep = Ep
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

fig, ax = plt.subplots()  # Create a figure containing a single axes.

ep01_rewards = []
ep001_rewards = []
ep1_rewards = []
for j in range(0, 2000):
    b = NArmedBandit(10, 0, 1, 1)
    e01 = EpGreedy(10, 0.1, b)
    e001 = EpGreedy(10, 0.01, b)
    e1 = EpGreedy(10, 0, b)
    for i in range(0, 1000):
        e01.make_move()
        e001.make_move()
        e1.make_move()
    ep01_rewards.append(e01._rewards_collected)
    ep001_rewards.append(e001._rewards_collected)
    ep1_rewards.append(e1._rewards_collected)


plt.title("n-armed-bandit, n=10, Q*(a)~N(0, 1), Q_t(a)~N(Q*(a), 1)")
plt.plot(np.mean(ep01_rewards, 0), label="Epsilon = 0.1")
plt.plot(np.mean(ep001_rewards, 0), label="Epsilon = 0.01")
plt.plot(np.mean(ep1_rewards, 0), label="Epsilon = 0 (greedy)")
plt.legend()
plt.show()

#print(e._rewards_collected)