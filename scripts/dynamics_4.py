
import numpy as np
from numba import njit, prange, jit, objmode
import netket as nk
import multiprocessing as mp
import time
import random
from.dynamics import new_dynamics


get_conn2 = nk.operator.DimerLocalOperator2._get_conn_flattened_kernel
log_val_kernel = nk.machine.rbm.RbmSpin._log_val_kernel

#N = 200 give the best performance.

import numba as nb


class new_dynamics_one(new_dynamics):
    
    
    def dynamics(self, X, t_list):
        
        # assert X.dtype == np.int8
        
        assert isinstance(t_list, np.ndarray)
        assert t_list.ndim == 1
        

        P = self._dynamics(
                X, 
                t_list,
                self.basis,
                self.constant,
                self.diag_mels,
                self.n_conns,
                self.mels,
                self.x_prime,
                self.acting_on,
                self._w,
                self.c_r,
                self.s_r
        )

        return np.swapaxes(P, 0, 1)

    
    
    @staticmethod
    @njit
    def _dynamics(
            X,
            t_list,
            _basis,
            _constant,
            _diag_mels,
            _n_conns,
            _mels,
            _x_prime,
            _acting_on,
            _w,
            c_r,
            s_r,
            ):
        
        # basis is float64[:]
        

        batch_size = X.shape[0]
        t_num = len(t_list)
        L = X.shape[1]
        p_array = np.zeros((batch_size, t_num, L),dtype= np.int8) 
        p_array[:,0,:] = X[:,:]
        X = X.astype(np.int8)
        _x_prime = _x_prime.astype(np.int8)
        
        

        
#         T = np.zeros(batch_size)
        t_delta = t_list[1] - t_list[0]

        l = 0
        for b in range(batch_size):
            T = 0
            ti_old = np.int64(0)
            x_b = X[b]
            P = p_array[b]
            sections = 0
            
            while True:


                r = (x_b).astype(np.float64).dot(_w)
                tan = np.tanh(r)


                x_prime, sites, mels  = get_transition_one(
                                    x_b,
                                    c_r,
                                    s_r,
                                    tan,
                                    sections,
                                    _basis,
                                    _constant,
                                    _diag_mels,
                                    _n_conns,
                                    _mels,
                                    _x_prime,
                                    _acting_on)


                N_conn = mels.shape[0]
#                 print(x_b)
#                 print(x_prime, sites, mels)
#                 for n in range(batch_size):
#                     a_0[n] = (-1)* mels[sections[n]: sections[n+1]].sum()
                a_0 = (-1) * mels.sum()

                r_1 = random.random()
                r_2 = random.random()

                tau = np.log(1/r_1)/a_0
                T += tau

#                 print(T,ti_new)
                if T > (t_list[-1] + t_delta):
                    P[ti_old+1:] = x_b
                    break
                    
                ti_new = np.int64(T // t_delta)

                if ti_new > ti_old:
                    P[ti_old+1:ti_new+1] = x_b
#                     P[ti_new] = x_b
                    ti_old = ti_new
                s = 0
                for i in range(N_conn):
                    s -= mels[i]
                    if s >= r_2 * a_0:
                        x_b[sites[i]] = x_prime[i]
                        break
                l += 1
            p_array[b,:,:] = P[:,:]

        return p_array
            
        
    def run(self, X, t_list, qout):
        
        out = self.dynamics(X, t_list)
        
        qout.put(out)
    
        
    def multiprocess(self, X, t_list,  n = 1):
        queue = []
        process = []
        N = X.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(X[index[i]:index[i+1]], t_list ,queue[i]))
            p.start()
            process.append(p)
        
        out1 = []
        for q in queue:
            out = q.get()
            out1.append(out)
        
        return np.concatenate(out1, axis=1)


@njit
def get_transition_one(
        x,
        c_r,
        s_r,
        tan,
        sections,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on
    ):


    n_sites = x.shape[0]


    n_operators = n_conns.shape[0]
    xs_n = np.empty(n_operators, dtype=np.intp)
    xs_n_prime = np.int64(0)
    max_conn = 0
    batch_size = 1

    tot_conn = 0

    sections = 1

    for i in range(n_operators):
        n_conns_i = n_conns[i]
        x_i = (x[acting_on[i]] + 1) / 2
        s = 0
        for j in range(4):
            s += x_i[j] * basis[j]
        xs_n[i] = s
        sections += n_conns_i[xs_n[i]]
    sections -= 1
    tot_conn=sections 



    x_prime = np.empty((tot_conn, 2), dtype=np.int8) # x.shpae[0] is number of connected elements of hamiltonian from batch of states. 
    t_mels = np.empty(tot_conn, dtype=np.float64)
    sites_ = np.empty((tot_conn, 2), dtype=np.int16)
    
    
    acting_on = acting_on[:,1:3].copy()
    
    basis_r = np.array([3,1])
    c = 0
    
    
    x_batch = x
    xs_n_b = xs_n
    tan_b = tan


    for i in range(n_operators):

        s_r_i = s_r[i]
        c_r_i = c_r[i]
        # Diagonal part
        n_conn_i = n_conns[i, xs_n_b[i]]

        if n_conn_i > 0:
            sites = acting_on[i]

            for cc in range(n_conn_i):

                x_prime_cc = x_prime[c + cc]
                x_prime_cc[:] = all_x_prime[i, xs_n_b[i], cc]
                sites_[c + cc] = sites

                num = ((np.sum((x_prime_cc - x_batch[sites]) * basis_r) + 8)/2)

                sin_prime = s_r_i[np.int(num)]
                cos_prime = c_r_i[np.int(num)]


                t_mels[c + cc] = all_mels[i, xs_n_b[i], cc] *np.prod(tan_b*sin_prime + cos_prime)
            c += n_conn_i


    return x_prime, sites_, t_mels