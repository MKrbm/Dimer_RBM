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
    ("x_prime", int8[:,:,:,:]),
    ("acting_on", int64[:,:]),
    ("acting_size", int64[:]),
    # ("acting_size", int64),
    
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
        # self.basis = basis[::-1].copy()
        self.basis = basis
        self.constant = constant
        self.diag_mels = diag_mels
        self.n_conns = n_conns
        self.mels = mels
        self.x_prime = x_prime
        self.acting_on = acting_on
        # self.acting_size = int(acting_size[0])
        self.acting_size = acting_size

    def transition(self, state, state_1, w, r, sweep_size):

        accepted = 0
        log_values = _log_val_kernel(state.astype(_np.float64), w, r)
        batch_size = state.shape[0]
        sections = _np.zeros(batch_size + 1, dtype=_np.int64)
        sections_1 = _np.zeros(batch_size + 1, dtype=_np.int64)
        state_1 = state

        state_prime, mels = self.get_conn(state, sections[1:])
        n_conn = -sections[:-1] + sections[1:] - 1

        # for _ in range(2*sweep_size):
        while True:


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

        return get_conn1(
            x,
            sections,
            self.local_states,
            self.basis,
            self.constant,
            self.diag_mels,
            self.n_conns,
            self.mels,
            self.x_prime,
            self.acting_on,
            self.acting_size,
        )

        # return get_conn2(
        #     x,
        #     sections,
        #     self.basis,
        #     self.constant,
        #     self.diag_mels,
        #     self.n_conns,
        #     self.mels,
        #     self.x_prime,
        #     self.acting_on,
        #     self.acting_size,
        # )

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
def acceptance_kernel(
    state, state1, log_values, log_values_1, log_prob_corr, n_conn, n_conn_1,
    state_prime, state_1_prime, mels, mels_1 ,sections, sections_1,
):
    accepted = 0

    batch_size = state.shape[0]

    # for i in range(state.shape[0]):
    #     prob = _np.exp(
    #         2 * (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
    #     )
    #     assert not math.isnan(prob)
    #     if prob > _random.uniform(0, 1):
    #         log_values[i] = log_values_1[i]
    #         state[i] = state1[i]
    #         accepted += 1
    prob = _np.exp(
        2 * (log_values_1 - log_values + log_prob_corr).real
    )

    index = prob > _np.random.rand(batch_size)
    n_conn[index] = n_conn_1[index]
    log_values[index] = log_values_1[index]
    state[index] = state1[index]
    accepted += index.sum()

    i = 0
    j = 0
    state_prime_ = _np.zeros(((n_conn + 1).sum(), state.shape[1]), dtype=_np.int8)
    mels_ = _np.zeros((n_conn + 1).sum(), dtype = _np.complex128)
    sections_ = _np.zeros_like(sections)
    for n in n_conn:
        if index[j]:
            assert (sections_1[j+1] - sections_1[j] ) == n + 1
            temp = state_1_prime[sections_1[j]: sections_1[j+1]]
            temp_mel = mels_1[sections_1[j]:sections_1[j+1]]
        else:
            assert (sections[j+1] - sections[j] ) == n + 1
            temp = state_prime[sections[j]: sections[j+1]]
            temp_mel = mels[sections[j]:sections[j+1]]
        
        sections_[j+1] = n + sections_[j] + 1
        
        state_prime_[i:i+n+1] = temp
        mels_[i:i+n+1] = temp_mel
        i += n+1
        j += 1
    
    # sections = sections_1
    # print('done')

    # state_prime = state_prime_
    # mels = mels_



    return accepted , state_prime_, mels_, sections_

@jit(nopython=True)
def _log_val_kernel(x, W, r):

    if x.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    # if out is None:
    out = _np.empty(x.shape[0], dtype=_np.complex128)
    r = x.dot(W)
    _log_cosh_sum(r, out)

    return out

# @jit(nopython=True)
# def acceptance_kernel(
#     state, state1, log_values, log_values_1, log_prob_corr
# ):
#     accepted = 0

#     for i in range(state.shape[0]):
#         prob = _np.exp(
#             2 * (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
#         )
#         assert not math.isnan(prob)
#         if prob > _random.uniform(0, 1):
#             log_values[i] = log_values_1[i]
#             state[i] = state1[i]
#             accepted += 1

#     return accepted