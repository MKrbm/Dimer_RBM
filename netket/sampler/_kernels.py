from functools import singledispatch
import numpy as _np
from . import numpy

# Register numpy kernels here
@singledispatch
def _LocalKernel(machine):
    return numpy._LocalKernel(
        _np.asarray(machine.hilbert.local_states), machine.input_size
    )

def _DimerLocalKernel(machine, op, kernel):


    if kernel == 1:
        return numpy._DimerLocalKernel_1(
            _np.asarray(machine.hilbert.local_states, dtype=_np.int8),
            machine.input_size,
            op._basis,
            op._constant,
            op._diag_mels,
            op._n_conns,
            op._mels,
            op._x_prime,
            op._acting_on,
            op._acting_size
        )

        
    elif kernel == 2:
        return numpy._DimerLocalKernel_2(
            _np.asarray(machine.hilbert.local_states, dtype=_np.int8),
            machine.input_size,
            op._basis,
            op._constant,
            op._diag_mels,
            op._n_conns,
            op._mels,
            op._x_prime,
            op._acting_on,
            op._acting_size
        )




@singledispatch
def _ExchangeKernel(machine, d_max):
    return numpy._ExchangeKernel(machine.hilbert, d_max)


@singledispatch
def _CustomKernel(machine, move_operators, move_weights=None):
    return numpy._CustomKernel(move_operators, move_weights)


@singledispatch
def _HamiltonianKernel(machine, hamiltonian):
    return numpy._HamiltonianKernel(hamiltonian)
