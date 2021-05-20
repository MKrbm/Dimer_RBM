import numpy as np


n_samples = int(1e6) # number of samples in dynamics
n_max = 10 # number of partitions
n_samples_corr = int(n_samples/n_max)
n_samples_RBM = 1 * 10 ** 4 # number of sample used for montecarlo
length = [4, 4]
alpha = 2
n_chains = 1
# n_discard = 1200
a = 0 # direction of dimer when calculate DimerCorrelation
N_bin = 200
# t_list = np.linspace(0, 20, N_bin)
t_list = np.arange(0,20,0.1)
n_iter = int(600)
n_jobs = -1
sweep_size = 40
n_discard = 50
NUM = 300  # number of transition in dynamics. 
# n_discard = 72