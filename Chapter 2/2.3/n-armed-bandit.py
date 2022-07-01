import numpy as np
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
