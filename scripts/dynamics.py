
import numpy as np
from numba import njit, prange, jit, objmode
import netket as nk
import multiprocessing as mp
import time

get_conn2 = nk.operator.DimerLocalOperator2._get_conn_flattened_kernel
log_val_kernel = nk.machine.rbm.RbmSpin._log_val_kernel

#N = 200 give the best performance.

import numba as nb



results = {}

class new_dynamics:
    def __init__(self,op,ma):
        
        self.local_states = np.sort(op._local_states)
        self.basis = op._basis[::-1].copy()
        self.constant = op._constant
        self.diag_mels = op._diag_mels
        self.n_conns = op._n_conns
        self.mels = np.real(op._mels)
        self.x_prime = op._x_prime[:,:,:,1:3].copy()
        self.acting_on = op._acting_on
        self.ma = ma


        self._w = ma._w
        self._a = ma._a
        self._b = ma._b
        self._r = ma._r

        acting_list = op._acting_on[:,1:3]
        w_ = self._w[acting_list]
        x = np.zeros((9,2))

        n = 0
        for i in range(3):
            for j in range(3):
                
                x[n,1] = 2*(j-1)
                x[n,0] = 2*(i-1)
                n += 1

        r_primes = np.einsum('ij,kjl->kil', x, w_)
        self.c_r = np.cosh(r_primes)
        self.s_r = np.sinh(r_primes)

        
    
    
    def dynamics(self, X, num):
        
        # assert X.dtype == np.int8
        
        
        return self._dynamics(
            X, 
            num,
            self.local_states,
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

    
    
    @staticmethod
    @njit
    def _dynamics(
            X,
            num,
            _local_states,
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
        L = X.shape[1]
        p_array = np.zeros((num, batch_size, L),dtype= np.int8) 
        X_float = X
        X = X.astype(np.int8)
        T = np.zeros(batch_size)
        t_array = np.zeros((num, batch_size))
        _x_prime = _x_prime.astype(np.int8)
        
        

        sections = np.zeros(batch_size + 1, dtype = np.int64)
        a_0 = np.zeros(batch_size)
        
        for _ in range(num):


            r = (X).astype(np.float64).dot(_w)
            tan = np.tanh(r)

           
            x_prime, sites, mels  = get_transition(
                                X,
                                c_r,
                                s_r,
                                tan,
                                sections[1:],
                                _basis,
                                _constant,
                                _diag_mels,
                                _n_conns,
                                _mels,
                                _x_prime,
                                _acting_on)
                                



            # with objmode(start='float64'):
            #     start = time.time()

            # # log_val_prime = np.real(log_val_kernel(x_prime.astype(np.float64), None, _w, _a, _b, _r))
            # R = np.empty(x_prime.shape[0], dtype=np.float64)
            # for n in range(batch_size):
            #     R[sections[n]:sections[n+1]] = rate_psi(X_float[n], x_prime[sections[n]:sections[n+1]].astype(np.float64), sites[sections[n]:sections[n+1]], _w)

            # with objmode():
            #     end = time.time()   
            #     print(end-start,'cal log')




            # for n in range(batch_size):
            #     log_val_prime[sections[n] : sections[n+1]] -= log_val_prime[sections[n]]
            

            # mels = np.real(mels) * np.exp(log_val_prime)
            # mels = np.real(mels) * R

            N_conn = sections[1:] - sections[:-1]
            for n in range(batch_size):
                a_0[n] = (-1)* mels[sections[n]: sections[n+1]].sum()
#             print(a_0[0], N_conn[0], mels[sections[0] + 1: sections[1]])
                
#             print((a_0 - mels[sections[:-1]]).mean(), mels[sections[:-1]].mean())

                
            r_1 = np.random.rand(batch_size)
            r_2 = np.random.rand(batch_size)
            
            tau = np.log(1/r_1)/a_0
            t_array[_, :] = T
            T += tau

            
            p_array[_,:,:] = X
            for n in range(batch_size):
                s = 0
                for i in range(N_conn[n]):
                    s -= mels[sections[n] + i]
                    if s >= r_2[n] * a_0[n]:
                        X[n][sites[sections[n] +  i]] = x_prime[sections[n] +  i]
                        break

        return p_array, t_array
            
        
    def run(self, X, num, qout):
        
        out = self.dynamics(X, num)
        
        qout.put(out)
    
        
    def multiprocess(self, X, num,  n = 1):
        queue = []
        process = []
        N = X.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(X[index[i]:index[i+1]], num ,queue[i]))
            p.start()
            process.append(p)
        
        out1 = []
        out2 = []
        for q in queue:
            out = q.get()
            out1.append(out[0])
            out2.append(out[1])
        
        return np.concatenate(out1, axis=1), np.concatenate(out2, axis=1)



@njit
def rate_psi(X, X_prime_local, sites, W):
    
    n_conn = X_prime_local.shape[0]
    
    r = X.dot(W)
    tan = np.tanh(r)
#     r_prime = np.empty((n_conn, W.shape[1]), dtype=np.float64)
    out = np.empty(n_conn, dtype=np.float64)
    
    
    for n in range(n_conn):
        W_prime = W[sites[n]]
        r_prime = X_prime_local[n].dot(W_prime)
        for j in range(W.shape[1]):
            out[n] = np.prod(np.cosh(r_prime) + tan*np.sinh(r_prime))
    return out




@jit(nopython=True)
def get_transition(
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


    batch_size = x.shape[0]
    n_sites = x.shape[1]

    assert sections.shape[0] == batch_size

    n_operators = n_conns.shape[0]
    xs_n = np.empty((batch_size, n_operators), dtype=np.intp)
    xs_n_prime = np.empty(batch_size, dtype=np.intp)
    max_conn = 0

#     acting_size = np.int8(acting_size)
#     acting_size = 4
    tot_conn = 0

    sections[:] = 1

    for i in range(n_operators):
        n_conns_i = n_conns[i]
        x_i = (x[:, acting_on[i]] + 1) / 2
        s = np.zeros(batch_size)
        for j in range(4):
            s += x_i[:, j] * basis[j]
        xs_n[:, i] = s
        sections += n_conns_i[xs_n[:, i]]
    sections -= 1
    tot_conn=sections.sum()

    s = 0 
#     sec = sections.copy()
    for b in range(batch_size):
        s += sections[b]
        sections[b] = s



    x_prime = np.empty((tot_conn, 2), dtype=np.int8) # x.shpae[0] is number of connected elements of hamiltonian from batch of states. 
    t_mels = np.empty(tot_conn, dtype=np.float64)
    sites_ = np.empty((tot_conn, 2), dtype=np.int8)
    
    
    acting_on = acting_on[:,1:3].copy()
    
    basis_r = np.array([3,1])
    c = 0
    
    for b in range(batch_size):
        x_batch = x[b]
        xs_n_b = xs_n[b]
#         r_b = r[b]
#         psi_b = np.prod(np.cosh(r_b))
        tan_b = tan[b]
#         psi_b = 1
#         print('psi_b',psi_b)
        
        
        for i in range(n_operators):
            
#             r_prime_i = r_primes[i]
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
#                     print(x_prime_cc - x_batch[sites])
#                     print(num)
                    sin_prime = s_r_i[np.int(num)]
                    cos_prime = c_r_i[np.int(num)]
                    
#                     print((r_prime+r_b)[10])
#                     temp = np.exp(r_prime+r_b)
#                     t_mels[c + cc] = all_mels[i, xs_n_b[i], cc] * np.prod((temp + 1/temp)/2)/psi_b# transition matrix
#                     log_val[c + cc] = np.prod(np.cosh(r_prime+r_b))/psi_b
#                     log_val[c + cc] = np.prod(tan_b*sin_prime + cos_prime)
                    t_mels[c + cc] = all_mels[i, xs_n_b[i], cc] *np.prod(tan_b*sin_prime + cos_prime)
                c += n_conn_i


    return x_prime, sites_, t_mels