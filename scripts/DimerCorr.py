import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import scipy.fft as fft
import pickle


def Dimer_Corr(h, V, length, t_list, n_samples, a):


    name = 'h={}V={}l={}'.format(h, V, length)

    # n is the number of batch (multiprocessing)

    # print('start n = {}'.format(n))
    folder = 'h={}V={}l={}'.format(h, V, length)

    if not os.path.exists(parentdir + '/save/corr/'+folder):
        os.makedirs(parentdir + '/save/corr/'+folder)
    



    # np.save(parentdir + '/save/corr/'+folder + "/dimer_momentum.npy", fft_dimer_corr_prime)

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))

    '''
    define some variables.
    '''



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
            K[l1,l2] = hex_.b1*4*np.pi*l1/K.shape[0] + hex_.b2*4*np.pi*l2/K.shape[1]  


    uni = np.exp(1j*np.einsum('ijk,lk->lij',K,edge_coors_prime))'''


    edge_corr1 = np.array([ 0.5     ,  0.      ])
    edge_corr2 = np.array([ 0.25    , -0.433013])
    edge_corr3 = np.array([ 0.75    , -0.433013])

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))


    x = np.arange(length[0])
    y = np.arange(length[1])

    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    operators_list = []

    for base_edge in [edge_corr1, edge_corr2, edge_corr3]:
        edge_coor_array = hex_.ProcessPeriodic(xx[:,None] * hex_.a1[None,:]+ yy[:,None]  * hex_.a2[None,:] + base_edge)
        edges = hex_.edge_coor_to_lattice(edge_coor_array)
        colors = hex_.get_edge_color(edges)

        operators_list.append(f.return_spin_corr(hi, edges, colors))
    
    T, color = hex_.autom(reverse=True, half=True)








    # name = '/n={:.1e}_{}'.format(n_samples, n)

    path = 'save/processed/'+ folder + name
    # with open(path, "rb") as fp:   
    #     [P_list, t_list] = pickle.load(fp)

    n_samples_= 10**4 
    NUM  = int(n_samples / n_samples_)
    print("NUM = ",NUM)

    # print('load P and t_list')
    '''
    x = np.arange(length[0]*2)
    y = np.arange(length[1]*2)

    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    edge_coor_array = xx[:,None] * hex_.a1[None,:] * 1/2 + yy[:,None]  * hex_.a2[None,:] * 1/2

    Dimer = []
    Dimer_std = []
    total_num_samples = []
    fft_dimer_corr_prime = np.zeros((2*length[0],2*length[1],t_list.shape[0]), dtype=np.complex128)
    fft_dimer_std_prime = np.zeros((2*length[0],2*length[1],t_list.shape[0]), dtype=np.complex128)

    for i in range(3):
        Dimer.append(np.zeros((2*length[0],2*length[1],t_list.shape[0])))
        Dimer_std.append(np.zeros((2*length[0],2*length[1],t_list.shape[0])))
        total_num_samples.append(np.zeros(t_list.shape[0]))'''

    Dimer = []
    Dimer_std = []
    total_num_samples = []


    # for i in range(3):
    #     Dimer.append(np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64))
    #     Dimer_std.append(np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64))
    #     total_num_samples.append(np.zeros(t_list.shape[0]))
    Dimer = np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64)
    Dimer_std = np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64)
    total_num_samples = np.zeros(t_list.shape[0])

    for j in range(NUM):
        
        P_list = np.load(parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy')
        # new_P_list = np.zeros((P_list.shape[0], P_list.shape[1] * T.shape[0] * T.shape[1], P_list.shape[-1]))

        # for t in range(len(new_P_list)):
        #     new_P_list[t] = f.permute(P_list[t], T, color)
        
        print(f"load : {parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy'}")

        
        for i in range(3):
            '''
            base_edge = edge_coor[i]

            edges = hex_.edge_coor_to_lattice(hex_.ProcessPeriodic(edge_coor_array + base_edge))
            colors = hex_.get_edge_color(edges)

            operators = f.return_spin_corr(hi, edges, colors)


                # P_list = np.load(parentdir+f'/save/dynamics/h=1.0V=1.0l=[4, 4]/P_n=1.0e+04_{i}.npy')
            dimer_corr, dimer_std, num_samples = f.cal_dimer_corr(operators, P_list, hex_, t_list)'''

        dimer_corr = np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64)
        dimer_std = np.zeros((3,3,length[0],length[0],len(t_list)), dtype = np.float64)


        for a in range(3):
            for b in range(3):
                dimer_corr[b, a], dimer_std, num_samples = f.cal_dimer_corr(operators_list[a][0], operators_list[b], P_list, hex_, t_list)

        for a in range(3):
            for b in range(a+1):
                for i in range(length[0]):
                    for j in range(length[1]):
                        
                        tmp = (dimer_corr[b,a][i,j] + dimer_corr[a,b][-i,-j])/2
                        dimer_corr[b,a][i,j] = tmp
                        dimer_corr[a,b][-i,-j] = tmp


        total_num_samples += num_samples
        Dimer += dimer_corr * num_samples
        Dimer_std += dimer_std * num_samples
            

            
    # for i in range(3):
    #     Dimer[i] /= total_num_samples[i]
    #     Dimer_std[i] /= total_num_samples[i]
    #     Dimer_std[i] /= np.sqrt(total_num_samples[i])

    Dimer /= total_num_samples
    Dimer_std /= total_num_samples
    Dimer_std /= np.sqrt(total_num_samples)

    dimer_momentum_ = np.zeros((3,3,2 * length[0], 2*length[1],len(t_list)),dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            dimer_momentum_[a,b] = f.dimer_fourier2D(dimer_corr[a,b])
    
    dimer_momentum = f.dimer_fourier2D_sub(dimer_momentum_,t_list, hex_)

    """
    for i in range(3):    
        dimer_corr = Dimer[i]
        dimer_std = Dimer_std[i]
        fft_dimer_corr = np.empty_like(dimer_corr,dtype=np.complex128)
        fft_dimer_std=np.empty_like(dimer_std,dtype=np.complex128)
        for t in range(t_list.shape[0]):
            fft_dimer_corr[:,:,t] = fft.fft2(dimer_corr[:,:,t])
            fft_dimer_std[:,:,t] = fft.fft2(dimer_std[:,:,t])


        fft_dimer_corr_prime += fft_dimer_corr * uni[i][:,:,None]
        fft_dimer_std_prime += fft_dimer_std * uni[i][:,:,None]"""

    

    np.save(parentdir + '/save/corr/'+folder + "/dimer_momentum_mean.npy", dimer_momentum)
    # np.save(parentdir + '/save/corr/'+folder + "/dimer_momentum._std.npy", fft_dimer_std_prime)
    np.save(parentdir + '/save/corr/'+folder + "/dimer_real_mean.npy", np.array(Dimer))
    np.save(parentdir + '/save/corr/'+folder + "/dimer_real_std.npy", np.array(Dimer_std))
    np.save(parentdir + '/save/corr/'+folder + "/total_num.npy", total_num_samples)
    print('done')