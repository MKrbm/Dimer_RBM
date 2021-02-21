import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
import functions as f


def Dimer_Dynamics(h, V, length,alpha,  t_list, n_jobs = 1, n_chains = 10, n_samples = 100):

    name = 'h={}V={}l={}'.format(h, V, length)

    if n_jobs == -1:
        try:
            n_jobs = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            n_jobs = os.cpu_count()
        print('n_jobs',n_jobs)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)

    op = f.dimer_hamiltonian(h, V,np.array(length))
    op_transition = f.dimer_flip(length = np.array(length))
    hex_ = nk.machine.new_hex(np.array(length))


    ma = nk.machine.RbmDimer(hi, hex_, alpha = alpha, symmetry = True
                        ,use_hidden_bias = False, use_visible_bias = False, dtype=float)
    
    ma.load(parentdir + '/save/ma/'+name)


    sa = nk.sampler.DimerMetropolisLocal(machine=ma, op=op_transition, length = length, n_chains=n_chains, sweep_size = 72)
    sa.generate_samples(1000) # discard the begginings of metropolis sampling.
    samples_state = sa.generate_samples(int(n_samples / n_chains))
    samples_state = samples_state.reshape(-1, ma.hilbert.size)

    print('prepared initial samples')

    d = f.dynamics2(
            op._local_states,
            op._basis,
            op._constant,
            op._diag_mels,
            op._n_conns,  
            op._mels,
            op._x_prime,
            op._acting_on,
            op._acting_size,
            ma
            )

    P = d.multiprocess(samples_state, t_list, 0, n = n_jobs) 
    print('saving dynamics')
    np.save(parentdir + '/save/dynamics/'+name + 'n={:.1e}.npy'.format(n_samples), P)
