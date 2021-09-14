from .metropolis_local import DimerMetropolisLocal
import multiprocessing as mp
import numpy as np
import mkl


class DimerMetropolisLocal_multi():

    def __init__(self, machine, op, n_chains=16, sweep_size=None, length = [4,2], kernel = 1, n_jobs = 1, transition = 2):
        mkl.set_num_threads(1)
        print('number of core :', n_jobs)
        self.sa_list = []
        self.n_jobs = n_jobs

        for _ in range(n_jobs):
            self.sa_list.append(DimerMetropolisLocal(machine=machine, op=op, length = length, n_chains=n_chains, sweep_size = sweep_size, kernel = kernel, transition = transition))
        
        self.sa_list[0].generate_samples(10)
        self.machine = machine
        self.sample_shape =  self.sa_list[0].sample_shape

    @staticmethod
    def run(sa, N, samples ,qout):
        try:
            out = sa.generate_samples(N)
            qout.put(out)
        
        except KeyboardInterrupt:
            print('Received keyboardinterrupt\n')
            qout.put(None)

    def reset(self):

        for n in range(self.n_jobs):
            self.sa_list[n].reset()

    
    def generate_samples(self, n_samples, samples=None):
        queue = []
        process = []
        n_each = int(n_samples/self.n_jobs)

        for i in range(self.n_jobs):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(self.sa_list[i], n_each, samples ,queue[i]))
            p.start()
            process.append(p)
        
        
        out = []


        for i in range(self.n_jobs):
            temp_out = queue[i].get()

            if temp_out is not None:
                self.sa_list[i]._state = temp_out[-1]
                out.append(temp_out)
            else:
                raise NameError('keyboardinterrupt')
        
        # print('# of accepted samples',self.sa_list[0]._accepted_samples)

        
        return np.concatenate(out, axis=0)
    

    def discard(self, n_discard):

        queue = []
        process = []

        for i in range(self.n_jobs):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(self.sa_list[i], n_discard, None ,queue[i]))
            p.start()
            process.append(p)
        
        for i in range(self.n_jobs):


            temp_out = queue[i].get()

            if temp_out is not None:
                self.sa_list[i]._state = temp_out[-1]



            else:
                raise NameError('keyboardinterrupt')