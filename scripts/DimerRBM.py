import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
import functions as f


def Dimer_RBM(h, V, length, alpha, n_iter):


    name = 'h={}V={}l={}'.format(h, V, length)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)


    op = f.dimer_hamiltonian(V = V, h = h ,length=np.array(length))
    op_transition = f.dimer_flip(length = np.array(length))


    hex_ = nk.machine.new_hex(np.array(length))

    ma = nk.machine.RbmDimer(hi,hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float)
    ma.init_random_parameters(seed=1234)



    sa = nk.sampler.DimerMetropolisLocal(machine=ma, op=op_transition, length = length)
    sr = nk.optimizer.SR(ma, diag_shift=0)
    opt = nk.optimizer.Sgd(ma, learning_rate=0.01)
    gs = nk.Vmc(
    hamiltonian=op,
    sampler=sa,
    optimizer=opt,
    n_samples=2000,
    sr = sr,
    )
    gs.run(n_iter=n_iter, out=parentdir + '/log/'+name)

    ma.save(parentdir + '/save/ma/'+name)