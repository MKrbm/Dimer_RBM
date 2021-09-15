import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f


def Dimer_RBM(h, V, length, alpha, n_iter, n_samples, n_chains, n_discard , sweep_size):


    kernel = 1
    # sweep_size = 200
    decay_factor = 1  # or 'sigmoid decay'

    n_jobs = -1
    if n_jobs == -1:
        try:
            n_jobs = int(int(os.environ['SLURM_JOB_CPUS_PER_NODE'])/2)
        except:
            n_jobs = os.cpu_count()
        print('n_jobs',n_jobs)




    name = 'h={}V={}l={}'.format(h, V, length)
    print(name)


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)


    ham = f.dimer_hamiltonian(V = V, h = h ,length=np.array(length))
    op_transition1, ad2o_o, op_num, label_num  = f.dimer_flip1(length = np.array(length), return_info = True)

    ad2_bool = np.zeros([ad2o_o.shape[0], ad2o_o.shape[0]], dtype = np.bool)
    for l in range(ad2o_o.shape[0]):
        label = ad2o_o[l]
        for op_ in label:
            ad2_bool[l,op_] = True
            
    hex_ = nk.machine.new_hex(np.array(length))



    ma = nk.machine.RbmDimer(hi, hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float, reverse=True, half=True)
    ma.init_random_parameters(seed=1234)

    try:
        ma.load(parentdir + '/save/ma/'+name)
        print('load saved params')
    except:
        pass
    
    transition = 2
    ma.hex.ad2o_o = ad2o_o.astype(np.int64)
    ma.hex.ad2_bool = ad2_bool

    sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition1
        , length = length, n_chains=n_chains, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs, transition = transition)
        

    sr = nk.optimizer.SR(ma, diag_shift=5e-3)
    opt = nk.optimizer.Sgd(ma, learning_rate=0.05, decay_factor = decay_factor ,N = n_iter)


    gs = nk.Vmc(
    hamiltonian=ham,
    sampler=sa_mul,
    optimizer=opt,
    n_samples=n_samples,
    sr=sr,
    n_discard=n_discard,
    )


    gs.run(n_iter=n_iter, out=parentdir+'/log/'+name)
    ma.save(parentdir + '/save/ma/'+name)




# slight modification with large sample and large seep_size

    # sweep_size = sweep_size * 3
    # n_samples = n_samples * 3
    # n_iter = 100
    # n_discard = 600

    # sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition1
    #     , length = length, n_chains=n_chains, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs)

    # sr = nk.optimizer.SR(ma, diag_shift=0)
    # opt = nk.optimizer.Sgd(ma, learning_rate=0.01, decay_factor = 1 ,N = n_iter)

    # gs = nk.Vmc(
    # hamiltonian=ham,
    # sampler=sa_mul,
    # optimizer=opt,
    # n_samples=n_samples,
    # sr = sr,
    # n_discard = n_discard,
    # )


    # gs.run(n_iter=n_iter, out=parentdir + '/log/'+name + '2')

    ma.save(parentdir + '/save/ma/'+name)