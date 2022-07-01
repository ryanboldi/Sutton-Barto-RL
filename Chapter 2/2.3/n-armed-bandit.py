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
