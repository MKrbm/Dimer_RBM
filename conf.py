import numpy as np


n_samples = int(1e6)
n_samples_RBM = 3 * 10 ** 4 # number of sample used for montecarlo
length = [12 ,12]
alpha = 4
n_chains = 1
n_discard = 1200
a = 0 # direction of dimer when calculate DimerCorrelation
t_list = np.linspace(0, 30, 201)
n_iter = int(600)
n_jobs = -1
sweep_size = 20
n_discard = 300

# n_discard = 72