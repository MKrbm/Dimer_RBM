import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f


def Dimer_Corr(h, V, length, t_list, n_samples, a, n):

    # n is the number of batch (multiprocessing)

    print('start n = {}'.format(n))
    foler = 'h={}V={}l={}'.format(h, V, length)

    if not os.path.exists(parentdir + '/save/corr/'+foler):
        os.makedirs(parentdir + '/save/corr/'+foler)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))


    l = []
    for i in range(length[0]):
        for j in range(length[1]):
            l.append([i, j])
    l = np.array(l)

    a_ = [a for _ in range(np.prod(length))]
    a = np.array(a_)

    edges, colors = hex_.dimer_corr(l,a)
    operators = f.return_dimer_operator(hi, edges, colors)

    print('start initial process ')


    name_P = '/P_n={:.1e}_{}.npy'.format(n_samples, n)
    name_T = '/T_n={:.1e}_{}.npy'.format(n_samples, n)

    P = np.load('save/dynamics/' + foler + name_P)
    T = np.load('save/dynamics/' + foler + name_T)

    print('load P and T')


    P = f.process_P(P, T, t_list)

    print('processed P ')
    P_ = P.reshape(-1,P.shape[-1])
    num_samples = (P[:,:,0]!=0).sum(axis=1)

    sections1 = np.arange(P.shape[1])
    sections2 = np.zeros(P_.shape[0])

    _, mels1 = operators[0].get_conn_flattened(P[0,:,:], sections1)
    sub1 = mels1.mean().real
    mels1 = mels1.real


    dimer_corr = np.zeros((length[0],length[1],t_list.shape[0]))
    dimer_std = np.zeros((length[0],length[1],t_list.shape[0]))

    for l1 in range(length[0]):
        for l2 in range(length[1]):
            
            print('l1={}/l2={}'.format(l1,l2), end ="  ")
            
            sub2 = operators[l1 * length[1] + l2].get_conn_flattened(P[0,:,:], sections1)[1].mean().real
            
            _, mels2_ = operators[l1 * length[1] + l2].get_conn_flattened(P_, sections2)
            mels2 = mels2_.reshape(P.shape[0], P.shape[1]).real
            dimer_corr[l1,l2] = (mels2 * mels1).sum(axis=1)/num_samples - sub1*sub2
            dimer_std[l1,l2] = (mels2 * mels1).std(axis=1) 
    
    corr_list = [dimer_corr, dimer_std, t_list, num_samples]



    np.save(parentdir + '/save/corr/'+foler + '/n={:.1e}_a={}.npy'.format(n_samples,int(a[0])), np.array(corr_list, dtype=object))

        # print('save {}/{}'.format(n+1, n_max))

    # return dimer_corr, dimer_std

