import numpy as _np
from netket.operator import _local_operator
from netket import random as _random
from numba import jit, int64, float64, complex128, int8
from ..._jitclass import jitclass


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
    ("acting_size", int64[:]),
    
]

get_conn = _local_operator.LocalOperator._get_conn_flattened_kernel


@jitclass(spec)
class _DimerLocalKernel:
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
        self.basis = basis
        self.constant = constant
        self.diag_mels = diag_mels
        self.n_conns = n_conns
        self.mels = mels
        self.x_prime = x_prime
        self.acting_on = acting_on
        self.acting_size = acting_size

    def transition(self, state, state_1,log_prob_corr):

        for n in range(state.shape[0]):
            
            state_1[n] = state[n]

            state_ = state[n]

            state_prime, mels = self.get_conn(state_)

            n_conn = state_prime.shape[0] - 1

            rs = _random.randint(0, n_conn)

            state_1[n] = state_prime[rs + 1]

            n_conn_prime = self.get_conn(state_1[n])[0].shape[0] - 1

            log_prob_corr[n] = _np.log(n_conn/n_conn_prime) * (1/2)


    def random_state(self, state):

        for i in range(state.shape[0]):
#         for si in range(state.shape[1]):
            rs = _random.randint(0, self.n_states)
            state[i] = _np.zeros_like(state[i]) + self.local_states[rs]
    
    @staticmethod
    def acceptance_kernel(
        state, state1, log_values, log_values_1, log_prob_corr, machine_pow
    ):
        accepted = 0

        for i in range(state.shape[0]):
            prob = _np.exp(
                machine_pow * (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
            )
            assert not math.isnan(prob)
            if prob > _random.uniform(0, 1):
                log_values[i] = log_values_1[i]
                state[i] = state1[i]
                accepted += 1

        return accepted



    def get_conn(self, x):

        return get_conn(
            x.reshape((1, -1)),
            _np.ones(1),
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
