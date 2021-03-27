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
    decay_factor = 'sigmoid decay'  # or 'sigmoid decay'
    n_jobs = 12
    n_discard = 300


    name = 'h={}V={}l={}'.format(h, V, length)


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)


    ham = f.dimer_hamiltonian(V = V, h = h ,length=np.array(length))
    op_transition1 = f.dimer_flip1(length = np.array(length))


    hex_ = nk.machine.new_hex(np.array(length))

    ma = nk.machine.RbmDimer(hi, hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float, reverse=True)
    ma.init_random_parameters(seed=1234)


    sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition1
        , length = length, n_chains=n_chains, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs)

    sr = nk.optimizer.SR(ma, diag_shift=0)
    opt = nk.optimizer.Sgd(ma, learning_rate=0.05, decay_factor = decay_factor ,N = n_iter)

    gs = nk.Vmc(
    hamiltonian=ham,
    sampler=sa_mul,
    optimizer=opt,
    n_samples=n_samples,
    sr = sr,
    n_discard = n_discard,
    )


    gs.run(n_iter=n_iter, out=parentdir + '/log/'+name)
    ma.save(parentdir + '/save/ma/'+name)


# slight modification with large sample and large seep_size

    sweep_size = sweep_size * 3
    n_samples = n_samples * 3
    n_iter = 100
    n_discard = 600

    sa_mul = nk.sampler.DimerMetropolisLocal_multi(machine=ma, op=op_transition1
        , length = length, n_chains=n_chains, sweep_size = sweep_size, kernel = 1, n_jobs=n_jobs)

    sr = nk.optimizer.SR(ma, diag_shift=0)
    opt = nk.optimizer.Sgd(ma, learning_rate=0.01, decay_factor = 1 ,N = n_iter)

    gs = nk.Vmc(
    hamiltonian=ham,
    sampler=sa_mul,
    optimizer=opt,
    n_samples=n_samples,
    sr = sr,
    n_discard = n_discard,
    )


    gs.run(n_iter=n_iter, out=parentdir + '/log/'+name + '2')

    ma.save(parentdir + '/save/ma/'+name)