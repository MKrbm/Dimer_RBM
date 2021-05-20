import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy as np
import netket as nk
from scripts import functions as f
import pickle



def Dimer_Process(h, V, length, t_list, n_samples, a, n):

    # n is the number of batch (multiprocessing)

    print('start n = {}'.format(n))
    folder = 'h={}V={}l={}'.format(h, V, length)
    print(folder)

    if not os.path.exists(parentdir + '/save/processed/'+folder):
        os.makedirs(parentdir + '/save/processed/'+folder)


    name_P = '/P_n={:.1e}_{}.npy'.format(n_samples, n)
    name_T = '/T_n={:.1e}_{}.npy'.format(n_samples, n)

    P = np.load('save/dynamics/' + folder + name_P)
    T = np.load('save/dynamics/' + folder + name_T)

    print('load P and T')


    P = f.process_P(P, T, t_list)
    name = '/n={:.1e}_{}'.format(n_samples, n)
    print('start save')
    path = 'save/processed/'+ folder + name
    with open(path, "wb") as fp:   #Pickling
        pickle.dump([P,t_list], fp)

