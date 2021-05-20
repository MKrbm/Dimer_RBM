import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import scipy.fft as fft
import pickle


def Dimer_Corr(h, V, length, t_list, n_samples, a, n):

    # n is the number of batch (multiprocessing)

    print('start n = {}'.format(n))
    folder = 'h={}V={}l={}'.format(h, V, length)

    if not os.path.exists(parentdir + '/save/corr/'+folder):
        os.makedirs(parentdir + '/save/corr/'+folder)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))

    '''
    define some variables.
    '''
    edge_coor = np.array([[ 0.5     ,  0.      ],
                           [ 0.25    , -0.433013],
                           [ 0.75    , -0.433013]])
    a = np.tan((1/6)*np.pi) * 1/2
    edge_coors_prime = np.array([
        hex_.a((1/2)*np.pi) * a,
        hex_.a(np.pi*(2/3)+(1/2)*np.pi) * a,
        hex_.a(np.pi*(4/3)+(1/2)*np.pi) * a,
    ])
    edge_coors_prime = np.round(edge_coors_prime,10)


    K = np.zeros((2*length[0],2*length[1],2))
    for l1 in range(K.shape[0]):
        for l2 in range(K.shape[1]):
            K[l1,l2] = hex_.b1 * 2*np.pi*l1/K.shape[0] + hex_.b2 * 2*np.pi*l2/K.shape[1]  


    uni = np.exp(1j*np.einsum('ijk,lk->lij',K,edge_coors_prime))


    name = '/n={:.1e}_{}'.format(n_samples, n)

    path = 'save/processed/'+ folder + name
    with open(path, "rb") as fp:   
        [P_list, t_list] = pickle.load(fp)


    print('load P and t_list')

    x = np.arange(length[0]*2)
    y = np.arange(length[1]*2)

    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    edge_coor_array = xx[:,None] * hex_.a1[None,:] * 1/2 + yy[:,None]  * hex_.a2[None,:] * 1/2
    fft_dimer_corr_prime = np.zeros((2*length[0],2*length[1],t_list.shape[0]), dtype=np.complex128)

    for i in range(3):

        base_edge = edge_coor[i]

        edges = hex_.edge_coor_to_lattice(hex_.ProcessPeriodic(edge_coor_array + base_edge))
        colors = hex_.get_edge_color(edges)

        operators = f.return_spin_corr(hi, edges, colors)

        num_samples = (P_list[:,:,0]!=0).sum(axis=1)
        P_list_ = P_list.reshape(-1,P_list.shape[-1])
        sections1 = np.arange(P_list.shape[1])
        sections2 = np.zeros(P_list_.shape[0])

        _, mels1 = operators[0].get_conn_flattened(P_list[0,:,:], sections1)
        sub1 = mels1.mean().real
        mels1 = mels1.real


        dimer_corr = np.zeros((2*length[0],2*length[1],t_list.shape[0]))
        dimer_std = np.zeros((2*length[0],2*length[1],t_list.shape[0]))

        for l2 in range(2*length[1]):
            for l1 in range(2*length[0]):
        #     l1 = 1
        #     l2 = 0
                print('alpha={}, l1={}, l2={}'.format(i,l1,l2))
                if operators[l1 + l2 * 2*length[0]]:
                    sub2 = operators[l1 + l2 * 2*length[0]].get_conn_flattened(P_list[0,:,:], sections1)[1].mean().real

                    _, mels2_ = operators[l1 + l2 * 2*length[0]].get_conn_flattened(P_list_, sections2)
                    mels2 = mels2_.reshape(P_list.shape[0], P_list.shape[1]).real
                    dimer_corr[l1,l2] = (mels2 * mels1).sum(axis=1)/num_samples 
                    dimer_std[l1,l2] = (mels2 * mels1).std(axis=1)
                else:
                    dimer_corr[l1,l2] = 0
                    dimer_std[l1,l2] = 0


        fft_dimer_corr = np.empty_like(dimer_corr)
        for t in range(t_list.shape[0]):
            fft_dimer_corr[:,:,t] = np.real(fft.fft2(dimer_corr[:,:,t]))


        fft_dimer_corr_prime += fft_dimer_corr * uni[i][:,:,None]


    corr_list = [fft_dimer_corr_prime, t_list, num_samples]


    with open(parentdir + '/save/corr/'+folder + '/n={:.1e}'.format(n_samples),'wb') as fp:
        pickle.dump(corr_list, fp)

    # np.save(, np.array(corr_list, dtype=object))

        # print('save {}/{}'.format(n+1, n_max))

    # return dimer_corr, dimer_std

