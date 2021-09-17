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
    


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))


    edge_corr1 = np.array([ 0.5     ,  0.      ])
    edge_corr2 = np.array([ 0.25    , -0.433013])
    edge_corr3 = np.array([ 0.75    , -0.433013])

    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))


    x = np.arange(2*length[0])
    y = np.arange(2*length[1])

    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    operators_list = []

    # for base_edge in [edge_corr1, edge_corr2, edge_corr3]:
    edge_coor_array = hex_.ProcessPeriodic(xx[:,None] * hex_.a1[None,:]/2 + yy[:,None]  * hex_.a2[None,:]/2 + edge_corr1)
    edges = hex_.edge_coor_to_lattice(edge_coor_array)
    colors = hex_.get_edge_color(edges)

    operators_list = f.return_spin_corr(hi, edges, colors)
    
    # T, color = hex_.autom(reverse=True, half=True)

    n_samples_= 10**4 
    NUM  = int(n_samples / n_samples_)
    print("NUM = ",NUM)



    Dimer = []
    Dimer_std = []
    total_num_samples = []


    Dimer = np.zeros((2*length[0],2*length[0],2*length[0],2*length[0],len(t_list)), dtype = np.float64)
    Dimer_std = np.zeros((2*length[0],2*length[0],2*length[0],2*length[0],len(t_list)), dtype = np.float64)
    total_num_samples = np.zeros(t_list.shape[0])

    for j in range(NUM):
        
        P_list = np.load(parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy')
        print(f'P_list.shape = {P_list.shape}')
        print(f'P_list.shape = {t_list.shape}')
        
        print(f"load : {parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy'}")

        
        # for i in range(3):


        dimer_corr, dimer_std, num_samples = f.cal_dimer_corr_2(operators_list, P_list, hex_, t_list, sub_limit = True)



        total_num_samples[:] += num_samples
        Dimer[:] += dimer_corr * num_samples
        Dimer_std[:] += dimer_std * num_samples

    Dimer /= total_num_samples
    Dimer_std /= total_num_samples
    Dimer_std /= np.sqrt(total_num_samples)
            
    
    Dimer = f.process_symm_dimer(Dimer)
    dimer_momentum = f.dimer_fourier_simple(Dimer, hex_, edge_coor_array)

    

    np.save(parentdir + '/save/corr/'+folder + "/dimer_momentum_mean_{:.1e}.npy".format(n_samples), dimer_momentum)
    # np.save(parentdir + '/save/corr/'+folder + "/dimer_momentum._std.npy", fft_dimer_std_prime)
    np.save(parentdir + '/save/corr/'+folder + "/dimer_real_mean_{:.1e}.npy".format(n_samples), np.array(Dimer))
    np.save(parentdir + '/save/corr/'+folder + "/dimer_real_std_{:.1e}.npy".format(n_samples), np.array(Dimer_std))
    np.save(parentdir + '/save/corr/'+folder + "/total_num_{:.1e}.npy".format(n_samples), total_num_samples)
    print('done')