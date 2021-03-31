import math

import netket as _nk
import numpy as _np

from .operator import local_values as _local_values
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.vmc_common import info, tree_map
from netket.abstract_variational_driver import AbstractVariationalDriver
import time
import multiprocessing as mp
import copy




class Vmc(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, hamiltonian, sampler, optimizer, n_samples, n_discard=None, sr=None
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian (AbstractOperator): The Hamiltonian of the system.
            sampler: The Monte Carlo sampler.
            optimizer (AbstractOptimizer): Determines how optimization steps are performed given the
                bare energy gradient.
            n_samples (int): Number of Markov Chain Monte Carlo sweeps to be
                performed at each step of the optimization.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of the number of samples allocated to each MPI node.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.

            >>> import netket as nk
            >>> SEED = 3141592
            >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            >>> ha = nk.operator.Ising(hi, h=1.0)
            >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            >>> vmc = nk.Vmc(ha, sa, op, 200)

        """
        super(Vmc, self).__init__(
            sampler.machine, optimizer, minimized_quantity_name="Energy"
        )

        self._ham = hamiltonian
        self._sampler = sampler
        self.sr = sr

        self._npar = self._machine.n_par

        self._batch_size = sampler.sample_shape[0]

        # Check how many parallel nodes we are running on
        self.n_nodes = _nk.utils.n_nodes

        self.n_samples = n_samples
        self.n_discard = n_discard

        self._dp = None

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr):
        self._sr = sr
        if self._sr is not None:
            self._sr.setup(self.machine)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )

        n_samples_chain = int(math.ceil((n_samples / self._batch_size)))
        self._n_samples_node = int(math.ceil(n_samples_chain / self.n_nodes))

        self._n_samples = int(self._n_samples_node * self._batch_size * self.n_nodes)

        self._samples = None

        self._grads = None
        self._jac = None

    @property
    def n_discard(self):
        return self._n_discard

    @n_discard.setter
    def n_discard(self, n_discard):
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(n_discard)
            )
        self._n_discard = (
            int(n_discard)
            if n_discard != None
            else self._n_samples_node * self._batch_size // 10
        )

        print('discard = ',self._n_discard)

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self._sampler.reset()

        # Burnout phase


        print('n_discard = ',self._n_discard)
        s = time.time()
        self._sampler.discard(self._n_discard)
        print('discard samples', time.time()-s)

        # Generate samples and store them
        print('n_samples_node = ',self._n_samples_node)

        s = time.time()
        self._samples = self._sampler.generate_samples(
            self._n_samples_node, samples=self._samples
        )
        print('generate_samples : ',time.time()-s)


        # print(hexagon.is_dimer_basis2(self._samples[0]))

        # print(self._samples.shape)

        # Compute the local energy estimator and average Energy
        # print(whoami(), whosdaddy())

        s = time.time()
        eloc, self._loss_stats = self._get_mc_stats(self._ham)
        print('get_mc_stats', time.time()-s)

        # Center the local energy
        eloc -= _mean(eloc)

        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        eloc_r = eloc.reshape(-1, 1)

        # Perform update

        if self._sr:
            s = time.time()
            # When using the SR (Natural gradient) we need to have the full jacobian


            # self._grads, self._jac = self._machine.vector_jacobian_prod(
            #     samples_r.astype(_np.int8), eloc_r / self._n_samples, self._grads, return_jacobian=True
            # )
            # 
            # s = time.time()


            self._grads, self._jac = self.SR_process_mul(samples_r.astype(_np.int8), eloc_r)

            # grads, jac = self.SR_process(samples_r.astype(_np.int8), eloc_r)

            # print(self._grads - grads)
            # print(self._jac - jac )
            print('--->> cal O and jac ', time.time()-s)

            self._grads = tree_map(_sum_inplace, self._grads)

            self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)
            print('compute_update ', time.time()-s)

        else:
            # Computing updates using the simple gradient
            self._grads = self._machine.vector_jacobian_prod(
                samples_r, eloc_r / self._n_samples, self._grads
            )

            self._grads = tree_map(_sum_inplace, self._grads)

            #  if Real pars but complex gradient, take only real part
            # not necessary for SR because sr already does it.
            if not self._machine.has_complex_parameters:
                self._dp = tree_map(lambda x: x.real, self._grads)
            else:
                self._dp = self._grads

        return self._dp

    @property
    def energy(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def _estimate_stats(self, obs):
        return self._get_mc_stats(obs)[1]

    def reset(self):
        self._sampler.reset()
        super().reset()

    def _get_mc_stats(self, op):


        samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
        # print(whoami(), whosdaddy(), hexagon.is_dimer_basis2(samples_r).all(), samples_r.shape)

        loc = _local_values(op, self._machine, samples_r).reshape(
            self._samples.shape[0:2]
        )

        # notice that loc.T is passed to statistics, since that function assumes
        # that the first index is the batch index.
        return loc, _statistics(loc.T)

    def __repr__(self):
        return "Vmc(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("Machine     ", self._machine),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self._sr),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)

    def SR_process(self, samples, eloc):
        n_samples = eloc.shape[0]

        grads, jac =  self._machine.vector_jacobian_prod(
                samples.astype(_np.int8), eloc/n_samples, return_jacobian=True
            )

        print(grads, jac)

        return grads, jac

    
    @staticmethod
    def run_(ma, samples, eloc, qout):
        try:
            s = time.time()
            n_samples = eloc.shape[0]

            out =  ma.vector_jacobian_prod(
                    samples.astype(_np.int8), eloc/n_samples, return_jacobian=True
                )
            # print('cal jacobian', time.time()-s)
            
            # out = vmc.SR_process(samples, eloc)
            qout.put(out)
        
        except KeyboardInterrupt:
            print('Received keyboardinterrupt\n')
            qout.put(None)

    def SR_process_mul(self, samples, eloc):
        queue = []
        process = []
        n_samples = samples.shape[0]
        n_each = int(n_samples/self._sampler.n_jobs)
        # print(eloc)
        for i in range(self._sampler.n_jobs):
            queue.append(mp.Queue())
            sample_r = samples[i*n_each : (i+1) * n_each].copy()
            eloc_r = eloc[i*n_each : (i+1) * n_each].copy()
            p = mp.Process(target=self.run_, args=(self._machine, sample_r, eloc_r ,queue[i]))
            # out = self.run_(copy.copy(self._machine), sample_r, eloc_r , None)
            p.start()
            process.append(p)
        
        
        out1 = []
        out2 = []

        for i in range(self._sampler.n_jobs):
            temp_out = queue[i].get()

            if temp_out is not None:
                out1.append(temp_out[0])
                out2.append(temp_out[1])
            else:
                raise NameError('keyboardinterrupt')
        
        # print('# of accepted samples',self.sa_list[0]._accepted_samples)

        
        return _np.stack(out1).mean(axis=0), _np.vstack(out2)

import inspect
import os.path
import time

def whoami():
    return inspect.stack()[1][3]
def whosdaddy():
    return inspect.stack()[2][3]

hexagon = _nk.machine.graph_hex(length = [2,4])