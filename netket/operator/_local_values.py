import numpy as _np
from numba import jit
import netket as _nk

from ._local_liouvillian import LocalLiouvillian as _LocalLiouvillian
from netket.machine.density_matrix.abstract_density_matrix import (
    AbstractDensityMatrix as DensityMatrix,
)


'''

calculate .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle
for all input x : v_primes

'''

hexagon = _nk.machine.new_hex(_np.array([4,4]))


@jit(nopython=True)
def _local_values_kernel(log_vals, log_val_primes, mels, sections, out):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s] * _np.exp(log_val_primes[low_range:s] - log_vals[i])
        ).sum()
        low_range = s


def _local_values_impl(op, machine, v, log_vals, out):


    S = time.time()
    V = _np.asarray(v)
    sections = _np.empty(V.shape[0], dtype=_np.int32)
    # print(confirm)
    s = time.time()
    acting_list = op._acting_on[:,1:3]
    w_ = machine._w[acting_list]
    x = _np.zeros((9,2))

    n = 0
    for i in range(3):
        for j in range(3):

            x[n,1] = 2*(j-1)
            x[n,0] = 2*(i-1)
            n += 1
    r_primes = _np.einsum('ij,kjl->kil', x, w_)
    e = _np.exp(r_primes)
    c_r = (e + 1/e)/2
    s_r = (e - 1/e)/2

    r = V.dot(machine._w)
    tan = _np.tanh(r)

    # print('     prepare getting local values', time.time()-s)


    out[:] = get_local_value(
        V,
        c_r,
        s_r,
        tan,
        sections,
        op._basis[::-1].copy(),
        op._constant,
        _np.real(op._diag_mels),
        op._n_conns,
        _np.real(op._mels),
        op._x_prime[:,:,:,1:3].copy(),
        op._acting_on
    )

    # print('     obtained local values',time.time()-s)

    # v_primes, mels = op.get_conn_flattened(_np.asarray(v), sections)

    # print('     get connected ',time.time()-s)

    # log_val_primes = machine.log_val(v_primes)

    # print('     get log_val_prime ',time.time()-s)

    # print(out-l_mels)



@jit(nopython=True)
def _op_op_unpack_kernel(v, sections, vold):

    low_range = 0
    for i, s in enumerate(sections):
        vold[low_range:s] = v[i]
        low_range = s

    return vold


def _local_values_op_op_impl(op, machine, v, log_vals, out):

    acting_list = op._acting_on[:,1:3]
    w_ = ma._w[acting_list]
    x = _np.zeros((9,2))

    n = 0
    for i in range(3):
        for j in range(3):

            x[n,1] = 2*(j-1)
            x[n,0] = 2*(i-1)
            n += 1
    r_primes = _np.einsum('ij,kjl->kil', x, w_)
    e = _np.exp(r_primes)
    c_r = (e + 1/e)/2
    s_r = (e - 1/e)/2

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_np = _np.asarray(v)


    v_primes, mels = op.get_conn_flattened(v_np, sections)

    vold = _np.empty((sections[-1], v.shape[1]))
    _op_op_unpack_kernel(v_np, sections, vold)

    log_val_primes = machine.log_val(v_primes, vold)

    _local_values_kernel(
        _np.asarray(log_vals), _np.asarray(log_val_primes), mels, sections, out
    )


import inspect
def whoami():
    return inspect.stack()[1][3]
def whosdaddy():
    return inspect.stack()[2][3]


import time


def local_values(op, machine, v, log_vals=None, out=None):

    # print(whoami(),whosdaddy())
    # print(hexagon.is_dimer_basis2(v).all())
    r"""
    Computes local values of the operator `op` for all `samples`.

    The local value is defined as
    .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle


            Args:
                op: Hermitian operator.
                v: A numpy array or matrix containing either a batch of visible
                    configurations :math:`V = v_1,\dots v_M`.
                    Each row of the matrix corresponds to a visible configuration.
                machine: Wavefunction :math:`\Psi`.
                log_vals: A scalar/numpy array containing the value(s) :math:`\Psi(V)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                out: A scalar or a numpy array of local values of the operator.
                    If not given, it is allocated from scratch and then returned.
                    Defaults to None.

            Returns:
                If samples is given in batches, a numpy array of local values
                of the operator, otherwise a scalar.
    """


    s = time.time()
    # True when this is the local_value of a densitymatrix times an operator (observable)
    is_op_times_op = isinstance(machine, DensityMatrix) and not isinstance(
        op, _LocalLiouvillian
    )
    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    if log_vals is None:
        if not is_op_times_op:
            log_vals = machine.log_val(v)
        else:
            log_vals = machine.log_val(v, v)

    if not is_op_times_op:
        _impl = _local_values_impl
    else:
        _impl = _local_values_op_op_impl

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)

    if out is None:
        out = _np.empty(v.shape[0], dtype=_np.complex128)

    _impl(op, machine, v, log_vals, out)

    return out


@jit(nopython=True)
def get_local_value(
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
    xs_n = _np.empty((batch_size, n_operators), dtype=_np.intp)
    xs_n_prime = _np.empty(batch_size, dtype=_np.intp)
    max_conn = 0

#     acting_size = _np.int8(acting_size)
#     acting_size = 4
    tot_conn = 0

    sections[:] = 1

    for i in range(n_operators):
        n_conns_i = n_conns[i]
        x_i = (x[:, acting_on[i]] + 1) / 2
        s = _np.zeros(batch_size)
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



    x_prime = _np.empty((tot_conn, 2), dtype=_np.int8) # x.shpae[0] is number of connected elements of hamiltonian from batch of states. 
    l_mels = _np.zeros(batch_size, dtype=_np.float64)
    sites_ = _np.empty((tot_conn, 2), dtype=_np.int8)
    
    
    acting_on = acting_on[:,1:3].copy()
    
    basis_r = _np.array([3,1])
    c = 0
    
    for b in range(batch_size):

        c_diag = c
        x_batch = x[b]
        xs_n_b = xs_n[b]
#         r_b = r[b]
#         psi_b = _np.prod(_np.cosh(r_b))
        tan_b = tan[b]
#         psi_b = 1
#         print('psi_b',psi_b)
        l_mels[b] += constant
        
        
        for i in range(n_operators):
            

#             r_prime_i = r_primes[i]
            l_mels[b] += diag_mels[i, xs_n_b[i]]
            s_r_i = s_r[i]
            c_r_i = c_r[i]
            # Diagonal part
            n_conn_i = n_conns[i, xs_n_b[i]]

            if n_conn_i > 0:
                sites = acting_on[i]

                for cc in range(n_conn_i):
                    
                    x_prime_cc = x_prime[c + cc]
                    x_prime_cc[:] = all_x_prime[i, xs_n_b[i], cc]
                    
                    num = ((_np.sum((x_prime_cc - x_batch[sites]) * basis_r) + 8)/2)
#                     print(x_prime_cc - x_batch[sites])
#                     print(num)
                    sin_prime = s_r_i[_np.int(num)]
                    cos_prime = c_r_i[_np.int(num)]
                    
                    # print((r_prime+r_b)[10])
#                     temp = _np.exp(r_prime+r_b)
#                     t_mels[c + cc] = all_mels[i, xs_n_b[i], cc] * _np.prod((temp + 1/temp)/2)/psi_b# transition matrix
#                     log_val[c + cc] = _np.prod(_np.cosh(r_prime+r_b))/psi_b
#                     log_val[c + cc] = _np.prod(tan_b*sin_prime + cos_prime)
                    l_mels[b] += all_mels[i, xs_n_b[i], cc] *_np.prod(tan_b*sin_prime + cos_prime)
                c += n_conn_i

    return l_mels