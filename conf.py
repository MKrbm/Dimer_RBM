import numpy as np


n_samples = 3 * 10 **5 # number of samples in dynamics
n_max = 10 # number of partitions
n_samples_corr = int(n_samples/n_max)
n_samples_RBM = 10 ** 3 # number of sample used for montecarlo
length = [20, 20]
alpha = 2
n_chains = 1
# n_discard = 1200
a = 0 # direction of dimer when calculate DimerCorrelation
N_bin = 200
# t_list = np.linspace(0, 20, N_bin)
# t_list = np.arange(0,20,0.5)
t_list = np.arange(0,10,0.1)
n_iter = int(1000)
n_jobs = -1
sweep_size = np.prod(length) * 2
# n_discard = 50
# NUM = 300  # number of transition in dynamics. 
n_discard = 50
n_discard_d = np.prod(length) 