import multiprocessing as mp
import numpy as _np
import numpy as np
import time
from numba import njit

def run(ma, x, qout):
    try:
        out = der_log(x, ma._ws.copy(), ma._autom.copy(), ma._z2.copy())
        qout.put(out)
    except KeyboardInterrupt:
        print('Received keyboardinterrupt\n')
        qout.put(None)

def der_log(x, ws, autom, z2):
    batch_size = x.shape[0]
    symm_num = autom.shape[0]
    half_symm_num = int(symm_num/2)
    s = time.time()

    ws = ws.astype(_np.float32)
    ws2 = ma._ws.astype(_np.float32)  * 2

    # T_x_1 = translate_x(x, autom[:half_symm_num])
    # T_x_2 = translate_x(x * z2[1], autom[half_symm_num:])

    T_x_1 = _np.empty((x.shape[0], half_symm_num, x.shape[1]), dtype=_np.float32)
    T_x_2 = _np.empty_like(T_x_1)
    T_x_1[:,:,:] = ma.translate_x(x, ma._autom[:half_symm_num])
    T_x_2[:,:,:] = ma.translate_x(x * ma._z2[1], ma._autom[half_symm_num:])

    # tanh_1 = _np.tanh(T_x_1.astype(_np.float32).dot(ws))
    # tanh_2 = _np.tanh(T_x_2.astype(_np.float32).dot(ws))
    e1 = np.exp(T_x_1.dot(ws2))
    e2 = np.exp(T_x_2.dot(ws2))
    tanh_1 = (e1-1)/(e1+1)
    tanh_2 = (e2-1)/(e2+1)

    # print('     cal tan', time.time()-s)
    out1 = _np.einsum('ijk,ijl->ikl',T_x_1, tanh_1)
    out = (out1 + _np.einsum('ijk,ijl->ikl',T_x_2, tanh_2)).reshape(batch_size,-1)
    print('     der_log', time.time()-s)

    return out

def translate_x(x, autom):
    
    # out = _np.empty((x.shape[0], autom.shape[0], x.shape[1]), dtype=x.dtype)

    x_t = x.T.copy()
    out = x_t[autom]
    out = out.transpose((2,0,1))
    return out


    
def mul(ma, x, n_jobs):
    queue = []
    process = []
    n_samples = x.shape[0]
    n_each = int(n_samples/n_jobs)
    # print(eloc)
    for i in range(n_jobs):
        queue.append(mp.Queue())
        x_r = x[i*n_each : (i+1) * n_each].copy()
        p = mp.Process(target=run, args=(ma, x_r ,queue[i]))
        # out = self.run_(copy.copy(self._machine), sample_r, eloc_r , None)
        p.start()
        process.append(p)
    
        out1 = []

    for i in range(n_jobs):
        temp_out = queue[i].get()

        if temp_out is not None:
            out1.append(temp_out)
        else:
            raise NameError('keyboardinterrupt')
    return out1

def mul_p(ma, x, n_jobs):
    processes = []
    for _ in range(n_jobs):
        p = mp.Process(target=der_log, args = [x, ma._ws.copy(), ma._autom.copy(), ma._z2.copy()])
        p.start()
        processes.append(p)
    for process in processes:
        process.join()

def write2dat(corr, t_list, path):
    file = open(path,"w")
    L = corr.shape[0]
    for m1 in range(L):
        for m2 in range(L):
            file.write(f"#{m1},{m2}\n")
            for t in range(len(t_list)):
                tmp = corr[m1][m2][t]
                line = "{:3.1f}\t{:3.6f}\t{:3.6f}\n".format(t_list[t], tmp.real, tmp.imag)
                file.write(line)
            file.write("\n")
    file.close()