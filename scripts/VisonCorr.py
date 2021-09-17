import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import scipy.fft as fft
import pickle


def Vison_Corr(h, V, length, t_list, n_samples, a):


    name = 'h={}V={}l={}'.format(h, V, length)

    # n is the number of batch (multiprocessing)

    # print('start n = {}'.format(n))
    folder = 'h={}V={}l={}'.format(h, V, length)

    if not os.path.exists(parentdir + '/save/corr/'+folder):
        os.makedirs(parentdir + '/save/corr/'+folder)
    


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    hex_ = nk.machine.new_hex(np.array(length))


    v1 = np.array([0.5, -np.tan(np.pi / 6) * 1/2])
    v2 = hex_.lattice_coor[2] - hex_.lattice_coor[1]

    x = np.arange(3*length[0])
    y = np.arange(length[1])

    xx, yy = np.meshgrid(x,y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    edge_coor_array = (hex_.ProcessPeriodic(xx[:,None] * v1[None,:] + yy[:,None]  * v2[None,:] + hex_.lattice_coor[0]))

    corrs = hex_.coor_to_lattice_num(edge_coor_array)
    operators_list = f.return_vison(hi, corrs)
    
    n_samples_= 10**4 
    NUM  = int(n_samples / n_samples_)
    print("NUM = ",NUM)



    Vison = []
    Vison_std = []
    total_num_samples = []


    Vison = np.zeros((3*length[0],length[0],3*length[0],length[0],len(t_list)), dtype = np.float64)
    Vison_std = np.zeros((3*length[0],length[0],3*length[0],length[0],len(t_list)), dtype = np.float64)
    total_num_samples = np.zeros(t_list.shape[0])

    for j in range(NUM):
        
        P_list = np.load(parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy')
        print(f'P_list.shape = {P_list.shape}')
        print(f'P_list.shape = {t_list.shape}')
        
        print(f"load : {parentdir+f'/save/dynamics/{name}/P_n=1.0e+04_{j}.npy'}")

        
        # for i in range(3):


        vison_corr, vison_std, num_samples = f.cal_vison_corr(operators_list, P_list, hex_, t_list, True)



        total_num_samples[:] += num_samples
        Vison[:] += vison_corr * num_samples
        Vison_std[:] += vison_std * num_samples

    Vison /= total_num_samples
    Vison_std /= total_num_samples
    Vison_std /= np.sqrt(total_num_samples)
            
    
    Vison = f.process_symm_vison(Vison, corrs, hex_)
    Vison_momentum = f.vison_fourier_simple(Vison, hex_, edge_coor_array)

    

    np.save(parentdir + '/save/corr/'+folder + "/vison_momentum_mean_{:.1e}.npy".format(n_samples), Vison_momentum)
    # np.save(parentdir + '/save/corr/'+folder + "/Vison_momentum._std.npy", fft_Vison_std_prime)
    np.save(parentdir + '/save/corr/'+folder + "/vison_real_mean_{:.1e}.npy".format(n_samples), np.array(Vison))
    np.save(parentdir + '/save/corr/'+folder + "/vison_real_std_{:.1e}.npy".format(n_samples), np.array(Vison_std))
    np.save(parentdir + '/save/corr/'+folder + "/total_num_{:.1e}.npy".format(n_samples), total_num_samples)
    print('done')