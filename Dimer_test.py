import sys
import importlib
importlib.reload(sys)
# sys.path.insert(0,'../')
import numpy as np
import netket as nk
from scripts import functions as f
from scripts import new_dynamics 
import os
from conf import *

currentpath = os.getcwd()


length = [4, 4]
g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])
hi = nk.hilbert.Spin(s=0.5, graph=g)
hex_ = nk.machine.new_hex(np.array(length))


l = []
for i in range(length[0]):
    for j in range(length[1]):
        l.append([i, j])
l = np.array(l)
a = 0
a_ = [a for _ in range(np.prod(length))]
a = np.array(a_)



edges, colors = hex_.dimer_corr(l,a)
operators = f.return_spin_corr(hi, edges, colors)





size = len(operators)
Dimer = np.zeros((size,t_list.shape[0]))
total_num_samples = np.zeros(t_list.shape[0])

for i in range(10):

    P_list = np.load(currentpath+f'/save/dynamics/h=1.0V=1.0l=[4, 4]/P_n=1.0e+04_{i}.npy')

    dimer_corr = np.zeros((size,t_list.shape[0]))
    dimer_std = np.zeros((size,t_list.shape[0]))


    num_samples = (P_list[:,:,0]!=0).sum(axis=1)
    P_list_ = P_list.reshape(-1,P_list.shape[-1])
    sections1 = np.arange(P_list.shape[1])
    sections2 = np.zeros(P_list_.shape[0])


    _, mels1 = operators[0].get_conn_flattened(P_list_, sections2)
    mels1 = mels1.reshape(P_list.shape[0], P_list.shape[1]).real
    sub1 = mels1[0].mean().real






    for s in range(size):
        print(s)
        if operators[s]:

            _, mels2 = operators[s].get_conn_flattened(P_list[0,:,:], sections1)
            mels2 = mels2.real
            sub2 = mels2.mean().real
    #         mels2 = mels2_.reshape(P_list.shape[0], P_list.shape[1]).real
            print(mels1.shape, mels2.shape)
            dimer_corr[s] = (mels2 * mels1).mean(axis=1)
            dimer_std[s] = (mels2 * mels1).std(axis=1)
        else:
            dimer_corr[s] = 0
            dimer_std[s] = 0

    total_num_samples += num_samples
    Dimer += dimer_corr* num_samples


Dimer /= total_num_samples
file_name = 'test_dimer_corr.dat'
f.save_corr(file_name, Dimer, t_list)





