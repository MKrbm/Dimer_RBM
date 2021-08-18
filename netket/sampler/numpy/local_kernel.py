import numpy as _np
from netket.operator import _local_operator
from netket import random as _random
from numba import jit, int64, float64, complex128, int8
from ..._jitclass import jitclass
import math


@jitclass([("local_states", float64[:]), ("size", int64), ("n_states", int64)])
class _LocalKernel:
    def __init__(self, local_states, size):
        self.local_states = _np.sort(_np.asarray(local_states, dtype=_np.float64))
        self.size = size
        self.n_states = self.local_states.size

    def transition(self, state, state_1, log_prob_corr):

        for i in range(state.shape[0]):
            state_1[i] = state[i]

            si = _random.randint(0, self.size)

            rs = _random.randint(0, self.n_states - 1)

            state_1[i, si] = self.local_states[
                rs + (self.local_states[rs] >= state[i, si])
            ]

        log_prob_corr.fill(0.0)

    def random_state(self, state):

        for i in range(state.shape[0]):
            for si in range(state.shape[1]):
                rs = _random.randint(0, self.n_states)
                state[i, si] = self.local_states[rs]



spec = [
    ("local_states", int8[:]),
    ("size", int64), 
    ("n_states", int64),
    ("basis", int64[:]),
    ("constant", int64),
    ("diag_mels", complex128[:,:]),
    ("n_conns", int64[:,:]),
    ("mels", complex128[:,:,:]),
    ("x_prime", int8[:,:,:,:]),
    ("acting_on", int64[:,:]),
    # ("acting_size", int64[:]),
    ("acting_size", int64),
    
]

get_conn1 = _local_operator.LocalOperator._get_conn_flattened_kernel
get_conn2 = _local_operator.DimerLocalOperator2._get_conn_flattened_kernel


@jitclass(spec)
class _DimerLocalKernel_2:
    def __init__(self, 
                local_states, 
                size,
                basis,
                constant,
                diag_mels,
                n_conns,
                mels,
                x_prime,
                acting_on,
                acting_size,
                ):

        self.local_states = _np.sort(local_states)
        self.size = size
        self.n_states = self.local_states.size
        self.basis = basis[::-1].copy()
        # self.basis = basis
        self.constant = constant
        self.diag_mels = diag_mels
        self.n_conns = n_conns
        self.mels = mels
        self.x_prime = x_prime
        self.acting_on = acting_on
        self.acting_size = int(acting_size[0])
        # self.acting_size = acting_size

    def transition(self, state, state_1, log_prob_corr, w, r, sweep_size):

        accepted = 0
        log_values = _log_val_kernel(state.astype(_np.float64), w, r)
        batch_size = state.shape[0]
        sections = _np.zeros(batch_size + 1, dtype=_np.int64)
        sections_1 = _np.zeros(batch_size + 1, dtype=_np.int64)
        state_1 = state

        state_prime, mels = self.get_conn(state, sections[1:])
        n_conn = -sections[:-1] + sections[1:] - 1

        for _ in range(sweep_size):


            # state_prime_, mels_ = self.get_conn(state, sections[1:])
            # n_conn_ = - sections[:-1] + sections[1:] - 1

            # print((state_prime_ == state_prime).any())
            # print((n_conn_ + 1).sum())

            rs = (_np.random.rand(batch_size) * n_conn).astype(_np.int64)

            state_1 = state_prime[sections[:-1] + rs + 1]

            state_1_prime, mels_1 = self.get_conn(state_1, sections_1[1:])
            n_conn_1 = - sections_1[:-1] + sections_1[1:] - 1

            log_prob_corr = _np.log(n_conn/n_conn_1) * (1/2)

            
            log_values_1 = _log_val_kernel(state_1.astype(_np.float64), w, r)

            plus, state_prime, mels, sections = acceptance_kernel(
            state,
            state_1,
            log_values,
            log_values_1,
            log_prob_corr,
            n_conn,
            n_conn_1,
            state_prime,
            state_1_prime,
            mels,
            mels_1,
            sections,
            sections_1
            )

            # sections = sections_1

            accepted += plus
        return accepted


    def random_state(self, state):

        for i in range(state.shape[0]):
#         for si in range(state.shape[1]):
            rs = _random.randint(0, self.n_states)
            state[i] = _np.zeros_like(state[i]) + self.local_states[rs]
    






    def get_conn(self, x, sections):

        # return get_conn1(
        #     x,
        #     sections,
        #     self.local_states,
        #     self.basis,
        #     self.constant,
        #     self.diag_mels,
        #     self.n_conns,
        #     self.mels,
        #     self.x_prime,
        #     self.acting_on,
        #     self.acting_size,
        # )

        return get_conn2(
            x,
            sections,
            self.basis,
            self.constant,
            self.diag_mels,
            self.n_conns,
            self.mels,
            self.x_prime,
            self.acting_on,
            self.acting_size,
        )
spec = [
    ("local_states", int8[:]),
    ("size", int64), 
    ("n_states", int64),
    ("basis", int64[:]),
    ("constant", int64),
    ("diag_mels", complex128[:,:]),
    ("n_conns", int64[:,:]),
    ("mels", complex128[:,:,:]),
    ("x_prime", int8[:,:,:]),
    ("acting_on", int64[:,:]),
    ("acting_size", int64[:]),
#     ("acting_size", int64),
    
]

get_conn1 = _local_operator.LocalOperator._get_conn_flattened_kernel
get_conn2 = _local_operator.DimerLocalOperator2._get_conn_flattened_kernel


@jitclass(spec)
class _DimerLocalKernel_1:
    def __init__(self, 
                local_states, 
                size,
                basis,
                constant,
                diag_mels,
                n_conns,
                mels,
                x_prime,
                acting_on,
                acting_size,
                ):

        self.local_states = _np.sort(local_states)
        self.size = size
        self.n_states = self.local_states.size
        self.basis = basis[::-1].copy()
#         self.basis = basis
        self.constant = constant
        self.diag_mels = diag_mels
        self.n_conns = n_conns
        self.mels = mels
        self.x_prime = x_prime[:,:,0,:].copy()
        self.acting_on = acting_on
#         self.acting_size = int(acting_size[0])
        self.acting_size = acting_size

        
    
    def transition(self, state, state_1, w, r, sweep_size):
        '''
        This transition is exclusively for batch_size = 1
        '''

        accepted = 0
        batch_size = state.shape[0]
        
        assert batch_size == 1, 'batch_size must be 1'
        
        sections = _np.zeros(1, dtype=_np.int64)
        sections_1 = _np.zeros(1, dtype=_np.int64)
        state_1 = state

        log_values = _log_val_kernel(state.astype(_np.float64), w, r)[0]
#         print(log_values)
        
        state_prime = self.get_conn(state,sections)
        n_conn = sections[0]-1
#         print(log_values_prime)
        
        N = 0
        for _ in range(sweep_size * 2):
        # while True:
            
#             state_prime_, mels_ = self.get_conn(state, sections)
#             n_conn_ = sections[0]-1

            # print((state_prime_ == state_prime).any())
            # print((n_conn_ + 1).sum())
#             print(log_values)
            
            rs = (_np.random.rand(1) * (n_conn)).astype(_np.int64)

            state_1 = state_prime[rs+1].reshape(1,-1)

            state_1_prime = self.get_conn(state_1, sections_1)
            n_conn_1 = sections_1[0]-1

            prob_corr = n_conn/n_conn_1
            
            
            log_values_1 = _log_val_kernel(state_1.astype(_np.float64), w, r)[0]

            prob = _np.exp(
                2 * (log_values_1 - log_values)
            ) * prob_corr
            
            if prob > _np.random.rand(1):
                state[:] = state_1
                state_prime = state_1_prime
                log_values = log_values_1
                n_conn = n_conn_1
                accepted += 1
                

            # sections = sections_1

            N += 1
            if accepted >= sweep_size:
                break
                
        return accepted/N


    def random_state(self, state):

        for i in range(state.shape[0]):
#         for si in range(state.shape[1]):
            rs = _random.randint(0, self.n_states)
            state[i] = _np.zeros_like(state[i]) + self.local_states[rs]

    def get_conn(self, x, sections):

        return get_conn_one(
            x,
            sections,
            self.basis,
            self.n_conns,
            self.x_prime,
            self.acting_on,
        )

@jit(fastmath=True)
def _log_cosh_sum(x, out, add_factor=None):
    x = x * _np.sign(x.real)
    if add_factor is None:
        for i in range(x.shape[0]):
            out[i] = _np.sum(x[i] - _np.log(2.0) + _np.log(1.0 + _np.exp(-2.0 * x[i])))
    else:
        for i in range(x.shape[0]):
            out[i] += add_factor * (
                _np.sum(x[i] - _np.log(2.0) + _np.log(1.0 + _np.exp(-2.0 * x[i])))
            )
    return out

@jit(nopython=True)
def _log_val_kernel(x, W, r):

    if x.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    # if out is None:
    out = _np.empty(x.shape[0], dtype=_np.float64)
    r = x.dot(W)
    _log_cosh_sum(r, out)

    return out

@jit(nopython=True)
def get_conn_one(
    x,
    sections,
    basis,
    n_conns,
    all_x_prime,
    acting_on,
):
    batch_size = x.shape[0]
    n_sites = x.shape[1]

    assert batch_size == 1, 'batch_size must be 1'

    n_operators = n_conns.shape[0]
    temp_size = int(n_sites/4)+1
    
    X_temp = _np.zeros((temp_size, n_sites), dtype=x.dtype)
    


#     X_temp[0] = x
    S = 1
    for b in range(batch_size):
        x_b = x[b]
        # diagonal element
        # counting the off-diagonal elements
        for i in range(n_operators):
            a = 15
            
            acting_on_i = acting_on[i]
            x_i = x_b[acting_on_i]

            for k in range(4):
                a += (
                    x_i[k]
                    * basis[k]
                )
            a = int(a/2)
            
            if n_conns[i, a]:
                X_temp[S] = x_b
                X_temp[S][acting_on_i] = all_x_prime[i, a]
                S += 1
                

#         tot_conn += conn_b
        sections[b] = S
    



    return X_temp[:S]


@jit(nopython=True)
def get_conn_one_2(
    x,
    sections,
    basis,
    n_conns,
    all_x_prime,
    acting_on,
):
    batch_size = x.shape[0]
    n_sites = x.shape[1]

    assert batch_size == 1, 'batch_size must be 1'

    n_operators = n_conns.shape[0]
    temp_size = int(n_sites/4)+1
    
    X_temp = _np.zeros((temp_size, n_sites), dtype=x.dtype)
    xs_n = _np.zeros((batch_size, n_operators), dtype=_np.intp) + 15


#     X_temp[0] = x
    S = 1
    for b in range(batch_size):
        x_b = x[b]
        # diagonal element
        # counting the off-diagonal elements
        for i in range(n_operators):
            # xs_n_b_i = xs_n[b,i]
            
            acting_on_i = acting_on[i]
            x_i = x_b[acting_on_i]

            for k in range(4):
                xs_n[b,i] += (
                    x_i[k]
                    * basis[k]
                )
            xs_n[b,i] = int(xs_n[b,i]/2)
            
            if n_conns[i, xs_n[b,i]]:
                # X_temp[S] = x_b
                # X_temp[S][acting_on_i] = all_x_prime[i, xs_n_b_i]
                S += 1
        sections[b] = S
    X_temp = _np.zeros((S, n_sites), dtype=x.dtype)

    S = 1
    for b in range(batch_size):
        for i in range(n_operators):
            if n_conns[i, xs_n[b,i]]:
                X_temp[S] = x[b]
                X_temp[S][acting_on[i]] = all_x_prime[i, xs_n[b,i]]
                S += 1


#         tot_conn += conn_b
    



    return X_temp