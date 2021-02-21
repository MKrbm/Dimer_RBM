import netket as nk
import functions
import numpy as np

import multiprocessing as mp


length = np.array([2,4])

op = functions.dimer_hamiltonian(1, 1, length = length)


d = functions.dynamics2(
            op._local_states,
            op._basis,
            op._constant,
            op._diag_mels,
            op._n_conns,  
            op._mels,
            op._x_prime,
            op._acting_on,
            op._acting_size,
            )

t_list = np.linspace(0, 30, 101)
basis = np.resize(np.ones(16), (100000, 16)).astype(np.int8)
sections = np.empty(basis.shape[0], dtype=np.int32)

print('job start')

queue = []
process = []
for i in range(12):
    queue.append(mp.Queue())
    p = mp.Process(target=d.run, args=(basis, t_list, 0,queue[i]))
    p.start()
    process.append(p)

out = []
for q in queue:
    out.append(q.get())


#     q.task_done()

# for p in process:
#     p.join()

print('job done')