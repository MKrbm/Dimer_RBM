
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
        self.mels = op._mels
        self.x_prime = op._x_prime
        self.acting_on = op._acting_on
        self.acting_size = np.int64(op._acting_size[0])
        self.ma = ma


        self._w = ma._w
        self._a = ma._a
        self._b = ma._b
        self._r = ma._r
        
    
    
    def dynamics(self, X, num):
        
        
        
        
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
            self.acting_size,
            self._w,
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
            _acting_size,
            _w,
            ):
        
        # basis is float64[:]
        
        
        batch_size = X.shape[0]
        L = X.shape[1]
        p_array = np.zeros((num, batch_size, L),dtype= np.int8) 
        X_float = X
        X = X.astype(np.int8)
        T = np.zeros(batch_size)
        t_array = np.zeros((num, batch_size))
        
        

        sections = np.zeros(batch_size + 1, dtype = np.int64)
        a_0 = np.zeros(batch_size)
        
        for _ in range(num):

           
            x_prime, sites, mels  = get_conn_local(
                                X,
                                sections[1:],
                                _basis,
                                _constant,
                                _diag_mels,
                                _n_conns,
                                _mels,
                                _x_prime,
                                _acting_on,
                                _acting_size)
                                



            with objmode(start='float64'):
                start = time.time()

            # log_val_prime = np.real(log_val_kernel(x_prime.astype(np.float64), None, _w, _a, _b, _r))
            R = np.empty(x_prime.shape[0], dtype=np.float64)
            for n in range(batch_size):
                R[sections[n]:sections[n+1]] = rate_psi(X_float[n], x_prime[sections[n]:sections[n+1]].astype(np.float64), sites[sections[n]:sections[n+1]], _w)

            with objmode():
                end = time.time()   
                print(end-start,'cal log')




            # for n in range(batch_size):
            #     log_val_prime[sections[n] : sections[n+1]] -= log_val_prime[sections[n]]
            

            # mels = np.real(mels) * np.exp(log_val_prime)
            mels = np.real(mels) * R

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
            
        
    def run(self, basis, t_list, E_0, qout):
        
        out = self.dynamics(basis, t_list, E_0)
        
        qout.put(out)
    
        
    def multiprocess(self, basis, t_list, E0 , n = 1):
        queue = []
        process = []
        N = basis.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(basis[index[i]:index[i+1]], t_list, E0 ,queue[i]))
            p.start()
            process.append(p)


        out = []
        for q in queue:
            out.append(q.get())

        
        return np.vstack(out)



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
def get_conn_local(
        x,
        sections,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size
    ):


    batch_size = x.shape[0]
    n_sites = x.shape[1]

    assert sections.shape[0] == batch_size

    n_operators = n_conns.shape[0]
    xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

    max_conn = 0

#     acting_size = np.int8(acting_size)
#     acting_size = 4
    tot_conn = 0

    sections[:] = 1

    for i in range(n_operators):
        n_conns_i = n_conns[i]
        x_i = (x[:, acting_on[i]] + 1) / 2
        s = np.zeros(batch_size)
        for j in range(acting_size):
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



    x_prime = np.empty((tot_conn, acting_size), dtype=np.int8) # x.shpae[0] is number of connected elements of hamiltonian from batch of states. 
    mels = np.empty(tot_conn, dtype=np.complex128)
    sites_ = np.empty((tot_conn, acting_size), dtype=np.int8)


    c = 0
    for b in range(batch_size):
        x_batch = x[b]
        xs_n_b = xs_n[b]
        for i in range(n_operators):

            # Diagonal part
            n_conn_i = n_conns[i, xs_n_b[i]]

            if n_conn_i > 0:
                sites = acting_on[i]

                for cc in range(n_conn_i):
                    mels[c + cc] = all_mels[i, xs_n_b[i], cc]
                    x_prime_cc = x_prime[c + cc]
                    x_prime_cc[:] = all_x_prime[i, xs_n_b[i], cc] - x_batch[sites]
                    sites_[c + cc] = sites
                c += n_conn_i


    return x_prime, sites_ , mels