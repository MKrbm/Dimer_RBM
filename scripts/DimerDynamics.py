import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import scripts.dynamics as Dynamics
from scripts.dynamics_4 import new_dynamics_one
import time

def Dimer_Dynamics(h, V, length,alpha,  t_list, n_jobs = -1, n_chains = 10, n_samples = 100, n_max = 10, n_discard = 1000, sweep_size = 400, debug = False):


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
    op_transition , ad2o_o, op_num, label_num = f.dimer_flip1(length = np.array(length), return_info = True)
    op_transition = f.dimer_flip1(length = np.array(length))

    hex_ = nk.machine.new_hex(np.array(length))


    ma = nk.machine.RbmDimer(hi, hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float, reverse=True, half=True)
    
    ad2_bool = np.zeros([ad2o_o.shape[0], ad2o_o.shape[0]], dtype = np.bool)
    for l in range(ad2o_o.shape[0]):
        label = ad2o_o[l]
        for op_ in label:
            ad2_bool[l,op_] = True
            
    ma.hex.ad2o_o = ad2o_o.astype(np.int64)
    ma.hex.ad2_bool = ad2_bool


    if debug:
        ma.init_random_parameters(seed=1234)
        ma._ws[:] = 0
        ma._set_bare_parameters( ma._a, ma._b, ma._w, ma._as, ma._bs, ma._ws, ma._autom, ma._z2 )
    else:
        ma.load(parentdir + '/save/ma/'+name)
        print('loaded machine',time.time()-S)

    # sweep_size = 400
    sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition, length = length, n_chains=1, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs, transition = 1)
    sa_mul.reset()
    sa_mul.generate_samples(1000) # discard the begginings of metropolis sampling.
    print('discard samples',time.time()-S)


    '''

    Split samples into 10 block due to memory problem.

    '''

    n_samples_ = 10**4
    n_max = int(n_samples/n_samples_)
    # n_samples_ = int(n_samples/n_max)
    for n in range(n_max):

        samples_state = sa_mul.generate_samples(int(n_samples_ / n_chains))
        samples_state = samples_state.reshape(-1, ma.hilbert.size)

        print('prepared initial samples', time.time()-S)

        d = new_dynamics_one(op, ma)

        P = d.multiprocess(samples_state, t_list, n_jobs) 

        print('obtained dynamics')
        # f.process_P2(P)
        print('prepaired montecarlo sampling', time.time()-S)
        # print(P.shape)
        np.save(parentdir + '/save/dynamics/'+name + '/P_n={:.1e}_{}.npy'.format(n_samples_, n), P)
        # np.save(parentdir + '/save/dynamics/'+name + '/T_n={:.1e}_{}.npy'.format(n_samples_, n), P[1])

        print('done {}/{}'.format(n+1,n_max))
