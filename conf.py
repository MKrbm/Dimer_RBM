import numpy as np

n_samples = int(1e6)
length = [4, 4]
alpha = 2
n_chains = 10
a = 0 # direction of dimer when calculate DimerCorrelation
t_list = np.linspace(0, 50, 1001)
n_iter = int(2e3)
n_jobs = -1