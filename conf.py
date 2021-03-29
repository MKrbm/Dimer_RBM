import numpy as np


n_samples = int(1e6)
n_samples_RBM = 2 * 10 ** 3 # number of sample used for montecarlo
length = [4, 4]
alpha = 1
n_chains = 1
n_discard = 600
a = 0 # direction of dimer when calculate DimerCorrelation
t_list = np.linspace(0, 30, 201)
n_iter = int(300)
n_jobs = -1
sweep_size = 400

# n_discard = 72