import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import scripts.dynamics as Dynamics
import time

def Dimer_Dynamics(h, V, length,alpha,  t_list, n_jobs = -1, n_chains = 10, n_samples = 100):

    name = 'h={}V={}l={}'.format(h, V, length)

    # create save folder.
    if not os.path.exists(parentdir + '/save/dynamics/'+name):
        os.makedirs(parentdir + '/save/dynamics/'+name)

    S = time.time()

    if n_jobs == -1:
        try:
            n_jobs = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            n_jobs = os.cpu_count()
        print('n_jobs',n_jobs)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    op = f.dimer_hamiltonian(h, V,np.array(length))
    op_transition = f.dimer_flip1(length = np.array(length))
    hex_ = nk.machine.new_hex(np.array(length))


    ma = nk.machine.RbmDimer(hi, hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float, reverse=True, half=True)
    
    ma.load(parentdir + '/save/ma/'+name)
    print('loaded machine',time.time()-S)

    sweep_size = 400
    sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition, length = length, n_chains=1, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs)
    sa_mul.reset()
    sa_mul.generate_samples(1000) # discard the begginings of metropolis sampling.
    print('discard samples',time.time()-S)


    '''

    Split samples into 10 block since memory problem.

    '''
    n_max = 10
    n_samples_ = int(n_samples/n_max)
    for n in range(n_max):

        samples_state = sa_mul.generate_samples(int(n_samples_ / n_chains))
        samples_state = samples_state.reshape(-1, ma.hilbert.size)

        print('prepared initial samples', time.time()-S)

        d = Dynamics.new_dynamics(op, ma)

        P = d.multiprocess(samples_state, 500, n_jobs) 
        print('prepaired montecarlo sampling', time.time()-S)
        # print(P.shape)
        np.save(parentdir + '/save/dynamics/'+name + '/P_n={:.1e}_{}.npy'.format(n_samples_, n), P[0])
        np.save(parentdir + '/save/dynamics/'+name + '/T_n={:.1e}_{}.npy'.format(n_samples_, n), P[1])

        print('done {}/{}'.format(n+1,n_max))
