import abc
import numpy as _np
import time
import inspect
import os.path
import time

def whoami():
    return inspect.stack()[1][3]
def whosdaddy():
    return inspect.stack()[2][3]

class AbstractSampler(abc.ABC):
    """Abstract class for NetKet samplers"""

    def __init__(self, machine, sample_size=1):
        self.sample_size = sample_size
        self.machine = machine

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def reset(self, init_random=False):
        pass

    @property
    def machine_pow(self):
        return 2.0

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        self._machine = machine
        self._input_size = machine.input_size

        self.sample_shape = (self.sample_size, self._input_size)

    @machine_pow.setter
    def machine_pow(self, m_power):
        raise NotImplementedError

    def samples(self, n_max, init_random=False):

        self.reset(init_random)

        n = 0
        while n < n_max:
            yield self.__next__()
            n += 1

    def generate_samples(self, n_samples, init_random=False, samples=None):
        # print(whoami(),whosdaddy(), type(samples))
        self.reset(init_random)
        self._accepted_samples = 0

        if samples is None:
            samples = _np.empty((n_samples, self.sample_shape[0], self.sample_shape[1]))

        for i in range(n_samples):
            samples[i] = self.__next__()

        # print('# of accepted samples',self._accepted_samples/n_samples) 
        # print(self._w.shape)
        return samples


    def discard(self, n_samples, init_random=False, samples=None):
        # print(whoami(),whosdaddy(), type(samples))
        self.reset(init_random)
        start = time.time()

        if samples is None:
            samples = _np.empty((n_samples, self.sample_shape[0], self.sample_shape[1]))

        for i in range(n_samples):
            samples[i] = self.__next__()

        print('# of accepted samples',self._accepted_samples)
            
        
        print(time.time()-start,'for metropolis')
        return samples
    
