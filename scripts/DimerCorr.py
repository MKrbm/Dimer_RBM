import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f


def Dimer_Corr(h, V, length,t_list, n_samples, a):

    name = 'h={}V={}l={}n={:.1e}'.format(h, V, length, n_samples)

    # if n_jobs == -1:
    #     try:
    #         n_jobs = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    #     except:
    #         n_jobs = os.cpu_count()
    #     print('n_jobs',n_jobs)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))

    P = np.load('save/dynamics/' + name + '.npy')

    l = []
    for i in range(length[0]):
        for j in range(length[1]):
            l.append([i, j])
    l = np.array(l)

    a_ = [a for _ in range(np.prod(length))]
    a = np.array(a_)

    edges, colors = hex_.dimer_corr(l,a)
    operators = f.return_dimer_operator(hi, edges, colors)
    sections = np.arange(P.shape[0])

    _, mels1 = operators[0].get_conn_flattened(P[:,0,:], sections)
    sub1 = operators[0].get_conn_flattened(P[:,0,:], sections)[1].mean().real

    dimer_corr = np.zeros((length[0],length[1],t_list.shape[0]))
    dimer_std = np.zeros((length[0],length[1],t_list.shape[0]))

    for l1 in range(length[0]):
        for l2 in range(length[1]):
            sub2 = operators[1].get_conn_flattened(P[:,0,:], sections)[1].mean().real 
            for i in range(t_list.shape[0]):
                _, mels2 = operators[l1 * length[1] + l2].get_conn_flattened(P[:,i,:], sections)
                dimer_corr[l1,l2,i] = np.real(((mels1 * mels2 ).mean()))
                dimer_std[l1,l2,i] = np.real(((mels1 * mels2 ).std()))
            dimer_corr[l1,l2] -= sub1 * sub2
            dimer_std[l1,l2] /= np.sqrt(P.shape[0])
    
    np.save(parentdir + '/save/corr/corr_'+name + 'a={}.npy'.format(a[0]), dimer_corr)
    np.save(parentdir + '/save/corr/std_'+name + 'a={}.npy'.format(a[0]), dimer_std)

    # return dimer_corr, dimer_std

