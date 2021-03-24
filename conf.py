import numpy as np


n_samples = int(1e6)
n_samples_RBM = 2000 # number of sample using with montecarlo
length = [10, 10]
alpha = 1
n_chains = 1
n_discard = 300
a = 0 # direction of dimer when calculate DimerCorrelation
t_list = np.linspace(0, 30, 201)
n_iter = int(600)
n_jobs = -1
sweep_size = 400

# n_discard = 72