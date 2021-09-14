
import numpy as np
from numba import njit, prange, jit, objmode, boolean
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

    def dynamics2(self, X, t_list):
        
        # assert X.dtype == np.int8
        
        assert isinstance(t_list, np.ndarray)
        assert t_list.ndim == 1
        

        P = self._dynamics2(
                X, 
                t_list,
                self.basis,
                self.n_conns,
                self.mels,
                self.x_prime,
                self.acting_on,
                self._w,
                self.c_r,
                self.s_r,
                self.ad2o_o,
                self.ad2_bool

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
            
    @staticmethod
    @njit
    def _dynamics2(
            X,
            t_list,
            _basis,
            _n_conns,
            _mels,
            _x_prime,
            _acting_on,
            _w,
            c_r,
            s_r,
            _ad2o_o,
            _ad2_bool    
            ):



        batch_size = X.shape[0]
        sys_size = X.shape[1]
        t_num = len(t_list)
        L = X.shape[1]
        p_array = np.zeros((batch_size, t_num, L),dtype= np.int8) 
        p_array[:,0,:] = X[:,:]
        X = X.astype(np.int8)
        _x_prime = _x_prime[:,:,0,:].astype(np.int8)
        _mels = _mels[:,:,0]
        t_mels = np.zeros(int(sys_size/4), dtype=np.float64)
        conn_op_prime = np.zeros(int(sys_size/4), dtype=np.int32)
        conn_op_prime_1 = np.zeros(int(sys_size/4), dtype=np.int32)

        acting_on_prime = np.zeros((int(sys_size/4), 2), dtype=np.int32)

        x_prime = np.zeros((int(sys_size/4), 2), dtype=np.int32)
        op_labels_bool = np.zeros(len(_acting_on), dtype = boolean)
        
        t_delta = t_list[1] - t_list[0]

        l = 0
        for b in range(batch_size):
            T = 0
            ti_old = np.int64(0)
            x_b = X[b]
            P = p_array[b]
            sections = np.zeros(1, dtype=np.int64)
            op_labels = np.arange(len(_acting_on))
            n_conn = 0


            r = (x_b).astype(np.float64).dot(_w)
            
            while True:

                r = (x_b).astype(np.float64).dot(_w)
                
                tan = np.tanh(r)
                get_transition_one_2( x_b, c_r, s_r, tan, sections, _basis,_n_conns,  
                                    _mels, _x_prime, _acting_on, op_labels, op_labels_bool ,x_prime, 
                                    acting_on_prime, conn_op_prime, conn_op_prime_1, n_conn, t_mels)
                n_conn = sections[0]


                
    #             a_0 = (-1) * t_mels[:n_conn].sum()
                a_0 = 0
                # print(n_conn)
                for i in range(n_conn):
                    a_0 -= t_mels[i]
    #             print(sections)

                r_1 = random.random()
                r_2 = random.random()

                tau = np.log(1/r_1)/a_0
                T += tau

                if T > (t_list[-1] + t_delta):
                    P[ti_old+1:] = x_b
                    break

                ti_new = np.int64(T // t_delta)

                if ti_new > ti_old:
                    P[ti_old+1:ti_new+1] = x_b
                    ti_old = ti_new
                s = 0
            

                    
                for i in range(n_conn):
                    s -= t_mels[i]
                    if s >= r_2 * a_0:

                        acting_on_prime_i = acting_on_prime[i]
                        x_b[acting_on_prime_i] = x_prime[i]
                        conn_op_prime[:] = conn_op_prime_1
                        op_label = conn_op_prime[i]
                        op_labels = _ad2o_o[op_label]
                        op_labels_bool[:] = _ad2_bool[op_label]
                        r[:] = r + 2 * (x_prime[i][0] * _w[acting_on_prime_i[0]] + x_prime[i][1] * _w[acting_on_prime_i[1]])

                        break
                l += 1
            p_array[b,:,:] = P[:,:]
        return p_array
        

    def run(self, X, t_list, qout):
        
        out = self.dynamics(X, t_list)
        # print('2')
        # out = self.dynamics2(X, t_list)

        
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


@njit
def get_transition_one_2(
        x,
        c_r,
        s_r,
        tan,
        sections,
        basis,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        op_labels,
        op_labels_bool,
        x_prime, 
        acting_on_prime,
        conn_on_prime,
        conn_op_prime_1,
        n_conn_prime,
        t_mels
    ):

    '''
    S : sections
    all_mels : all non-zero elements in all_mels are fixed to -1.

    '''
    n_sites = x.shape[0]

    basis_r = np.array([3,1])

    S = 0
    for i in op_labels:
        acting_on_i = acting_on[i]
        x_i = x[acting_on[i]]
        a = 15
        for k in range(4):
            a += (
                x_i[k]
                * basis[k]
            )
        a = int(a/2)
        if n_conns[i, a]:

            # X_temp[S] = all_x_prime[i, a][1:3]
            acting_on_prime[S] = acting_on_i[1:3]
            conn_op_prime_1[S] = i
            s_r_i = s_r[i]
            c_r_i = c_r[i]
            x_prime_S = all_x_prime[i, a]
            x_prime[S] = x_prime_S

            # num = ((np.sum((x_prime[S] - x[sites]) * basis_r) + 8)/2)
            num = (x_prime_S[0] - x[acting_on_i[1]]) * basis_r[0] + (x_prime_S[1] - x[acting_on_i[2]]) * basis_r[1] + 8
            num /= 2
            sin_prime = s_r_i[np.int(num)]
            cos_prime = c_r_i[np.int(num)]

            # print(all_mels[i, a])
            t_mels[S] = -1 *np.prod(tan*sin_prime + cos_prime)


            S += 1

    for i in conn_on_prime[:n_conn_prime]:
        if not op_labels_bool[i]:
            conn_op_prime_1[S] = i
            acting_on_i = acting_on[i][1:3]
            acting_on_prime[S] = acting_on_i
            x_prime_S = -x[acting_on_i]
            x_prime[S] = x_prime_S
            s_r_i = s_r[i]
            c_r_i = c_r[i]

            # num = ((np.sum((x_prime[S] - x[sites]) * basis_r) + 8)/2)
            num = (x_prime_S[0] - x[acting_on_i[0]]) * basis_r[0] + (x_prime_S[1] - x[acting_on_i[1]]) * basis_r[1] + 8
            num /= 2
            sin_prime = s_r_i[np.int(num)]
            cos_prime = c_r_i[np.int(num)]

            t_mels[S] = -1 *np.prod(tan*sin_prime + cos_prime)

            S += 1

    sections[0] = S

