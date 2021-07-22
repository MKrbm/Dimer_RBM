import numpy as np
from numba import njit, prange
import netket as nk
import multiprocessing as mp

sigmaz = np.array([[1, 0], [0, -1]])
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,1],[-1,0]])

mszsz = np.kron(sigmaz, sigmaz)
mszsx = np.kron(sigmax, sigmax)
mszsy = np.kron(sigmay, sigmay)







def return_dimer_operator(hilbert , edges, colors):

    edges_ = edges.reshape(-1,2)
    colors_ = colors.reshape(-1,1)

    # op = nk.operator.DimerLocalOperator(hilbert)

    return_list = []

    for edge, color in zip(edges_, colors_):

        if color:
            l_op = (- color * mszsz + np.identity(4)) / 2
            return_list.append(nk.operator.LocalOperator(hilbert, l_op, edge.tolist()))
        else:
            return_list.append(None)

    return return_list 


def return_spin_corr(hilbert , edges, colors):

    edges_ = edges.reshape(-1,2)
    colors_ = colors.reshape(-1,1)

    # op = nk.operator.DimerLocalOperator(hilbert)

    return_list = []

    for edge, color in zip(edges_, colors_):

        if color:
            l_op = - color * mszsz 
            return_list.append(nk.operator.LocalOperator(hilbert, l_op, edge.tolist()))
        else:
            return_list.append(None)
    return return_list 


def return_vison(hilbert , lattice_num):

    lattice_num_ = lattice_num.reshape(-1)

    # op = nk.operator.DimerLocalOperator(hilbert)

    return_list = []

    for num in lattice_num_:

        if num!=-1:
            l_op = sigmaz 
            return_list.append(nk.operator.LocalOperator(hilbert, l_op, [num]))
        else:
            return_list.append(None)
    return return_list 



@njit
def process_P(P, T, t_list):
    mels_list = np.zeros((t_list.shape[0],)+P.shape[1:] , dtype=P.dtype)
    t_list_ = t_list.reshape(-1,1)

    S = 0
    for n in range(T.shape[0]-1):
        

        index = np.logical_and(T[n] <= t_list_,  t_list_ < T[n+1])

        for i in range(index.shape[1]):
            mels_list[index[:,i],i] = P[n,i]
    return mels_list









from numba import jitclass, int64, float64, complex128, njit, prange, int8



spec = [
    ("local_states", int8[:]),
    ("basis", int64[:]),
    ("constant", float64),
    ("diag_mels", complex128[:,:]),
    ("n_conns", int64[:,:]),
    ("mels", complex128[:,:,:]),
    ("x_prime", int8[:,:,:,:]),
    ("acting_on", int64[:,:]),
    ("acting_size", int64[:]),
    
]

get_conn = nk.operator.LocalOperator._get_conn_flattened_kernel


@jitclass(spec)
class _dynamics: 
    def __init__(self,
                local_states,
                basis,
                constant,
                diag_mels,
                n_conns,
                mels,
                x_prime,
                acting_on,
                acting_size,):
        
        self.local_states = np.sort(local_states)
        self.basis = basis
        self.constant = constant
        self.diag_mels = diag_mels
        self.n_conns = n_conns
        self.mels = mels
        self.x_prime = x_prime
        self.acting_on = acting_on
        self.acting_size = acting_size
        
    def _get_conn(self, x):
        
        return get_conn(
            x.reshape((1, -1)),
            np.ones(1, dtype=np.int64),
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
    
    def dynamics(self,X,time_list, E0):
        
        # basis is float64[:]
        
        
         
        t_d = time_list[1]-time_list[0]
        t_end = np.shape(time_list)[0]
        out = np.zeros((X.shape[0], t_end, X.shape[1]),dtype= np.int8)
        t_s = time_list[0]


        
        for j in prange(X.shape[0]):
            p = out[j]
            x = X[j]
            time = 0
            t_index_b = -1
        
            while True:
                x_prime, mels_= self._get_conn(x)

                mels = np.real(mels_)
                n_conn = mels.shape[0]

                a_0 = mels[0] - E0
                r_1 = np.random.uniform(0,1)
                r_2 = np.random.uniform(0,1)
                tau = np.log(1/r_1)/a_0
                time += tau
                if time > t_s:
                    t_index = int((time-t_s) // t_d)
                    if t_index >= t_end - 1:
                        p[np.arange(t_index_b + 1, t_end)] = x
    #                     for i in range(t_index_b + 1, t_end):
    #                         p[i,:] = x
    #                     return p
                        break
    #                 for i in range(t_index_b + 1, t_index+1):　
    #                     p[i,:] = x
                    p[np.arange(t_index_b + 1, t_index+1)] = x

                    t_index_b = t_index

                s = 0

                for i in range(n_conn-1):
                    s -= mels[i + 1]
                    if s >= r_2 * a_0:
                        x = x_prime[i]
                        break

        return out
            
        
    def run(self, basis, t_list, E0, qout):

        out = self.dynamics(basis, t_list, E0)
        print('get output')
        qout.put(out)


    
    def multiprocess(self, basis, t_list, E0, n = 1):
        queue = []
        process = []
        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(basis, t_list, E0 ,queue[i]))
            p.start()
            process.append(p)


        out = []
        for q in queue:
            out.append(q.get())

        
        return out

            
            
        
def dimer_hamiltonian(V, h,length = [4, 2]):


    sigmaz = np.array([[1, 0], [0, -1]])
    sigmax = np.array([[0,1],[1,0]])
    sigmay = np.array([[0,1],[-1,0]])

    mszsz = np.kron(sigmaz, sigmaz)
    mszsx = np.kron(sigmax, sigmax)
    mszsy = np.kron(sigmay, sigmay)

    potential_anti = (np.identity(4) + mszsz)/2
    potential_ferro = (np.identity(4) - mszsz)/2

    # hexagon = nk.machine.graph_hex(length = length)
    hexagon = new_hex(np.array(length))


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])

    e, ec, ce, cec = hexagon.for_hamiltonian()

    hi = nk.hilbert.Spin(s=0.5, graph=g)
    op = nk.operator.DimerLocalOperator(hi)

    for edge, edge_color, pe, pec in zip(e, ec, ce, cec):

        l_op = -h*mszsx
        
        l_op = np.kron(np.identity(2), l_op)
        l_op = np.kron(l_op, np.identity(2))
        
        mat = []
        edge_ = []
        for p, c in zip(pe, pec):
            mat.append(np.kron(
                potential_ferro if c[0] == 1 else potential_anti,
                potential_ferro if c[1] == 1 else potential_anti,
            ))
            edge_.append(p[0].tolist() + p[1].tolist())
        
        op += nk.operator.DimerLocalOperator(hi, l_op @ mat[0] + V * mat[0], edge_[0])
        op += nk.operator.DimerLocalOperator(hi, l_op @ mat[1] + V * mat[1], edge_[1])
    
    return op


def dimer_flip2(h = 1,length = [4, 2]):


    sigmaz = np.array([[1, 0], [0, -1]])
    sigmax = np.array([[0,1],[1,0]])
    sigmay = np.array([[0,1],[-1,0]])

    mszsz = np.kron(sigmaz, sigmaz)
    mszsx = np.kron(sigmax, sigmax)
    mszsy = np.kron(sigmay, sigmay)

    potential_anti = (np.identity(4) + mszsz)/2
    potential_ferro = (np.identity(4) - mszsz)/2

    # hexagon = nk.machine.graph_hex(length = length)
    hexagon = nk.machine.new_hex(np.array(length))


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])

    e, ec, ce, cec = hexagon.for_hamiltonian()

    hi = nk.hilbert.Spin(s=0.5, graph=g)
    op = nk.operator.DimerLocalOperator2(hi)

    for edge, edge_color, pe, pec in zip(e, ec, ce, cec):

        l_op = -h*mszsx
        
        l_op = np.kron(np.identity(2), l_op)
        l_op = np.kron(l_op, np.identity(2))
        
        mat = []
        edge_ = []
        for p, c in zip(pe, pec):
            mat.append(np.kron(
                potential_ferro if c[0] == 1 else potential_anti,
                potential_ferro if c[1] == 1 else potential_anti,
            ))
            edge_.append(p[0].tolist() + p[1].tolist())
        
        op += nk.operator.DimerLocalOperator2(hi, l_op @ mat[0], edge_[0])
        op += nk.operator.DimerLocalOperator2(hi, l_op @ mat[1], edge_[1])
    
    return op

def dimer_flip1(h = 1,length = [4, 2]):


    sigmaz = np.array([[1, 0], [0, -1]])
    sigmax = np.array([[0,1],[1,0]])
    sigmay = np.array([[0,1],[-1,0]])

    mszsz = np.kron(sigmaz, sigmaz)
    mszsx = np.kron(sigmax, sigmax)
    mszsy = np.kron(sigmay, sigmay)

    potential_anti = (np.identity(4) + mszsz)/2
    potential_ferro = (np.identity(4) - mszsz)/2

    # hexagon = nk.machine.graph_hex(length = length)
    hexagon = nk.machine.new_hex(np.array(length))


    g = nk.graph.Graph(nodes = [i for i in range(length[0] * length[1] * 2)])

    e, ec, ce, cec = hexagon.for_hamiltonian()

    hi = nk.hilbert.Spin(s=0.5, graph=g)
    op = nk.operator.DimerLocalOperator(hi)

    for edge, edge_color, pe, pec in zip(e, ec, ce, cec):

        l_op = -h*mszsx
        
        l_op = np.kron(np.identity(2), l_op)
        l_op = np.kron(l_op, np.identity(2))
        
        mat = []
        edge_ = []
        for p, c in zip(pe, pec):
            mat.append(np.kron(
                potential_ferro if c[0] == 1 else potential_anti,
                potential_ferro if c[1] == 1 else potential_anti,
            ))
            edge_.append(p[0].tolist() + p[1].tolist())
        
        op += nk.operator.DimerLocalOperator(hi, l_op @ mat[0], edge_[0])
        op += nk.operator.DimerLocalOperator(hi, l_op @ mat[1], edge_[1])
    
    return op


from numba import jitclass, int64, float64, complex128, njit, prange





get_conn = nk.operator.LocalOperator._get_conn_flattened_kernel
log_val_kernel = nk.machine.rbm.RbmSpin._log_val_kernel


class dynamics2: 

    def __init__(self,op,ma):
        
        self.local_states = np.sort(op._local_states)
        self.basis = op._basis
        self.constant = op._constant
        self.diag_mels = op._diag_mels
        self.n_conns = op._n_conns
        self.mels = op._mels
        self.x_prime = op._x_prime
        self.acting_on = op._acting_on
        self.acting_size = op._acting_size
        self.ma = ma


        self._w = ma._w
        self._a = ma._a
        self._b = ma._b
        self._r = ma._r
        
        
    
    
    def dynamics(self, X, time_list, E0):
        
        
        
        
        return self._dynamics(
            X, 
            time_list, 
            E0,
            self.local_states,
            self.basis,
            self.constant,
            self.diag_mels,
            self.n_conns,
            self.mels,
            self.x_prime,
            self.acting_on,
            self.acting_size,
            self._w,
            self._a,
            self._b,
            self._r,
                
        )
    
    
    @staticmethod
    @njit
    def _dynamics(
            X,
            time_list,
            E0,
            _local_states,
            _basis,
            _constant,
            _diag_mels,
            _n_conns,
            _mels,
            _x_prime,
            _acting_on,
            _acting_size,
            _w,
            _a,
            _b,
            _r,
            ):
        
        # basis is float64[:]
        
        
         
        t_d = time_list[1]-time_list[0]
        t_end = np.shape(time_list)[0]
        p_array = np.zeros((X.shape[0],t_end,X.shape[1]),dtype= np.int8)
        for j in range(X.shape[0]):
            # print('done ', j/X.shape[0])
            p = p_array[j]
            # x = X[j]
            time = 0
            t_index_b = -1
            num_ = 0
            while True:
                x_prime, mels = get_conn(
                                    X[j].reshape((1, -1)),
                                    np.ones(1),
                                    _local_states,
                                    _basis,
                                    _constant,
                                    _diag_mels,
                                    _n_conns,
                                    _mels,
                                    _x_prime,
                                    _acting_on,
                                    _acting_size)

                # state = x_prime.copy()
                # state = state.astype(np.float64)

                log_val_prime = np.real(log_val_kernel(x_prime.astype(np.float64), None, _w, _a, _b, _r))

                mels = np.real(mels) * np.exp(log_val_prime - log_val_prime[0])
                n_conn = mels.shape[0]

                a_0 = (-1)* mels[1:].sum()
                r_1 = np.random.uniform(0,1)
                r_2 = np.random.uniform(0,1)
                tau = np.log(1/r_1)/a_0
                time += tau
                t_index = int(time // t_d)

                if t_index >= t_end - 1:
                    p[np.arange(t_index_b + 1, t_end)] = X[j]
                    break
                p[np.arange(t_index_b + 1, t_index+1)] = X[j]

                t_index_b = t_index

                s = 0

                for i in range(n_conn-1):
                    s -= mels[i + 1]
                    if s >= r_2 * a_0:
                        X[j] = x_prime[i + 1]
                        break
    #                     print(x_prime[i])
    #                     print(x)      
                num_ += 1
            print(num_)
        return p_array
            
        
    def run(self, basis, t_list, E_0, qout):
        
        out = self.dynamics(basis, t_list, E_0)
        
        qout.put(out)
    
        
    def multiprocess(self, basis, t_list, E0 , n = 1):
        queue = []
        process = []
        N = basis.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(basis[index[i]:index[i+1]], t_list, E0 ,queue[i]))
            p.start()
            process.append(p)


        out = []
        for q in queue:
            out.append(q.get())

        
        return np.vstack(out)




            
        
    def run(self, basis, t_list, E_0, qout):
        
        out = self.dynamics(basis, t_list, E_0)
        
        qout.put(out)
    
        
    def multiprocess(self, basis, t_list, E0 , n = 1):
        queue = []
        process = []
        N = basis.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(basis[index[i]:index[i+1]], t_list, E0 ,queue[i]))
            p.start()
            process.append(p)


        out = []
        for q in queue:
            out.append(q.get())

        
        return np.vstack(out)
            
            

from numba import njit

@njit
def vec_n_to_state(number, local_states, out):
    for n in range(number.shape[0]):
        out[n] = nk.operator._local_operator._number_to_state(number[n] , local_states, out[n])





class new_hex:
    
    def __init__(self, l = np.array([4, 2])):
        
        self.a1 = self.a(np.float32(0))
        self.a2 =  self.a(np.pi*(5/3))

        '''

        definite lattice vector. If one change gage, this vectors will change correspondingly.

        '''
        self.b1 = self.a1 * 2
        self.b2 = self.a2

        # definite list of index of unit cells.
        self.all_unit_cell = np.zeros((int(np.prod(l)/2), 2), dtype=np.int)
        for i in range(int(l[0]/2)):
            for j in range(l[1]):
                self.all_unit_cell[j * int(l[0]/2) + i] = np.array([i,j])


        self.R1 = self.a1 * l[0]
        self.R2 = self.a2 * l[1]
        
        self.epsilon = 1e-5
        
        self.l = l
        
        self.x_array = np.zeros((l[0],l[1],2)).astype(np.float32)
        
        for i in range(l[0]):
            for j in range(l[1]):
                self.x_array[i, j] = self.a1 * i + self.a2 * j
        
        self.x = np.zeros((np.prod(l),2), dtype=np.float32)
        
        for i in range(l[0]):
            for j in range(l[1]):
                self.x[j * l[0] + i] = self.x_array[i, j]
        
        self.lattice_coor_array = np.zeros([l[0], l[1], 6, 2],dtype=np.float32)
        
        
        
        
        for j in range(l[1]):
            for i in range(l[0]):
                for a in range(6):
                    self.lattice_coor_array[i, j, a] = self.x_array[i, j] + self.alpha(a)
                    
        self.lattice_coor_array = self.ProcessPeriodic(self.lattice_coor_array)
                    

                    
        self.lattice_coor = np.zeros((np.prod(l) * 2, 2), dtype=np.float32)
        
        for j in range(l[1]):
            for n,a in enumerate([0,5]):
                for i in range(l[0]):
                    self.lattice_coor[j * 2 * l[0] + l[0] * n + i] = self.lattice_coor_array[i, j, a] 
                    

        self.all_hex_index = []

        for i in range(l[0]):
            for j in range(l[1]):
                self.all_hex_index.append([i,j])
        self.all_hex_index = np.array(self.all_hex_index)

        self.edges_from_hex , self.edges_color_from_hex = self.edges_from_hex_(l = self.all_hex_index, color=True, num =True)
        self.edges = np.sort(self.edges_from_hex[:,np.array([0,5,4]),:].reshape(-1,2),axis=1)
        self.edges_color = self.edges_color_from_hex[:,np.array([0,5,4])].reshape(-1)
        
                    
        
                
                
    def a(self, theta):
        return self.Rotation(theta) @ np.array([1,0])
    
    def alpha(self, i):
        r = np.array([0, (1/(np.sqrt(3)))],dtype=np.float32)
        
        return self.Rotation(np.pi*(1/3) * i) @ r
    
    def LatticeToHexIndex(self, X):
        
        lattice_coor_array = self.ProcessPeriodic(self.lattice_coor_array)
        X_ = self.ProcessPeriodic(X).reshape(-1, 2)
        
        HexIndex = np.zeros((X_.shape[0],3,2), dtype=np.int)
        
        for n, x in enumerate(X_):
            m = 0
            for j in range(self.l[1]):
                for i in range(self.l[0]):
                    lp = lattice_coor_array[i, j]
                    if (np.abs(lp-x).sum(axis=1) < self.epsilon).any():
                        HexIndex[n,m] = np.array([i, j])
                        m += 1
        
        
        return HexIndex.reshape(X.shape[:-1] + (3,) + (2,))
    
    def ProcessPeriodic(self, X):
        
        X_ = X.reshape(-1, 2).copy()
        
        W = self.to_lattice_vec(X_) % self.l
        
        A = np.concatenate((self.a1.reshape(-1,1), self.a2.reshape(-1,1)), axis=1)
        
#         for n in range(X_.shape[0]):

#             w = self.decompose(X_[n].copy(), self.R1, self.R2)

#             X_[n] = ((w[0] % 1)*self.R1 + (w[1] % 1)*self.R2)
            
        return (A @ W.T).T.reshape(X.shape)
    
    
    def edges_from_hex_(self, l, color=False, num = True):
        
        '''
        
        l : coordinate number of hex
        
        '''
        
        l = l.reshape(-1,2)
        edge = np.zeros((l.shape[0],6,2,2),dtype=np.float32)
        
        if color:
            color_ = np.ones((l.shape[0],6),dtype=np.int)
        
        for i in range(6):
            
            edge[:, i, 0] = self.lattice_coor_array[l[:,0], l[:,1], i, :]

            
            edge[:, i, 1] = self.lattice_coor_array[l[:,0], l[:,1], (i + 1) % 6, :]



            if color: # this might change according to gage one will take. 
                if i == 4:
                    color_[l[:,0] % 2 == 0, i] = -1

                if i == 1:
                    color_[l[:,0] % 2 == 1, i] = -1  
        if num:
            edge = self.lpos_to_num(edge)
        
        if color:
            return edge, color_
        else:
            return edge
#         return self.lattice_coor_array[l[:,0], l[:,1]]

        
    def lpos_to_num(self, V):
        
        # convert lattice corrdinate to index of lattice(integer)
        
        V_ = V.reshape(-1, 2)
        V_num = np.zeros(V_.shape[0], dtype=np.int)
        
        for n, v in enumerate(V_):
            V_num[n] = np.where(np.abs(self.lattice_coor - v).sum(axis=1) < self.epsilon)[0][0]
        
        
        return V_num.reshape(V.shape[:-1])
    
    
    def get_edge_color(self, edges):
        
        
        edges_ = np.sort(edges.reshape(-1,2), axis=1)
        edges_color = np.zeros(edges_.shape[0])
        
        
        for i, edge in enumerate(edges_):
            index = np.where((self.edges == edge).all(axis=1))[0][0]
            edges_color[i] = self.edges_color[index]
        
        return edges_color.reshape(edges.shape[:-1])

    
    def is_dimer_basis(self, basis):

        assert basis.shape[-1] == self.l.prod()*2 , 'dimension not confirmed'

        basis = basis.reshape(-1, basis.shape[-1])

        index = ((basis[:, self.edges_from_hex].prod(axis=3)*self.edges_color_from_hex).sum(axis=2) == 4).all(axis=1)

        return index


    def from_edges_to_hex(self, edges, num = False):

        '''

        return hexagon lattice index and alpha that represents dimer direction.

        '''

        edges_ = edges.reshape(-1,2) # = edges[None, :]

        out1 = ((np.resize(self.edges_from_hex, (edges_.shape[0],) + self.edges_from_hex.shape)) == edges_.reshape(-1,1,1,2)).all(axis=-1)

        out2 = ((np.resize(self.edges_from_hex, (edges_.shape[0],) + self.edges_from_hex.shape)) == edges_[:,::-1].reshape(-1,1,1,2)).all(axis=-1)


        out = out1 | out2
        out = np.argwhere(out).reshape(-1, 2, 3)
        if not num:
            
            return self.all_hex_index[out[:,:, 1].reshape(edges.shape[:-1] + (-1,))], out[:,:, 2].reshape(edges.shape[:-1] + (-1,))
        
        else:
            return out[:,:, 1].reshape(edges.shape[:-1] + (-1,)), out[:,:, 2].reshape(edges.shape[:-1] + (-1,))

    def HexIndex_to_num(self, index):

        index_ = index.reshape(-1,2)

        S = index.shape[0]
        
        out = np.argwhere((np.resize(self.all_hex_index,(S, 8, 1, 2)) == index_.reshape(S, 1, 2, 2)).all(axis=-1))[:,1].reshape(index.shape[:-1])

        return out 

    def for_hamiltonian(self):

        e = self.edges
        ec = self.edges_color


        hex_index, alpha_index = self.from_edges_to_hex(e, num = True)


        alpha1 = alpha_index.copy()
        alpha1[:,0] = (alpha1[:,0] + 1) % 6
        alpha1[:,1] = (alpha1[:,1] + 1) % 6

        alpha2 = alpha_index.copy()
        alpha2[:,0] = (alpha2[:,0] - 1) % 6
        alpha2[:,1] = (alpha2[:,1] - 1) % 6
        

        pe1 = self.edges_from_hex[hex_index, alpha1]

        pe2 = self.edges_from_hex[hex_index, alpha2]


        pe = np.concatenate((pe1[:,None,:,:],pe2[:,None,:,::-1]), axis=1)
        temp = pe[:,:,0,:]
        pe[:,:,0,:] = temp[:,:,::-1]

        pec = self.get_edge_color(pe)
    
        return e, ec, pe, pec
    
    def lattice_num_to_coor(self, num):

        num_ = num.reshape(-1, 1)
        
        coor = self.lattice_coor[num_]

        return coor.reshape(num.shape + (2,))

    def coor_to_lattice_num(self, coor):

        coor_ = coor.reshape(-1, 2)
        
        temp = np.argwhere(np.abs(np.expand_dims(self.lattice_coor,axis=0) - np.expand_dims(coor_,axis=1)).sum(-1) < self.epsilon)[:,1]

        return temp.reshape(coor.shape[:-1])

    
    def translation(self,coors, c ):
        '''
        coor : lattice coordinates

        translate coordinate of edges by c[0] * self.b1 + c[1] * self.b2

        '''

        assert c.dtype == np.int, 'dtype of c must be int'

        c_ = c.reshape(-1,2)

        coors_ = coors.reshape(-1,2)



        translate_coors  = np.expand_dims(coors_, axis=0) + (c[:,0][:, None] * self.b1 + c[:,1][:, None] * self.b2)[:, None, :]

        translate_coors = self.ProcessPeriodic(translate_coors)

        return translate_coors.reshape((c_.shape[0],)+coors.shape)
    
    def mirror(self, lattice_num_array, axis = np.array([1,0])):

        '''
        lattice_num : index of lattice to be mirrored.
        0  1  2  3        
        4  5  6  7  
        8  9  10 11
        12 13 14 15
        '''
        lattice_num_array_ = lattice_num_array.reshape(-1, np.prod(self.l) * 2)

        lattice_ = lattice_num_array_.reshape(-1, self.l[1] * 2, self.l[0])

        lattice_ = np.roll(lattice_, (1, 1), axis=(1,2))
        print(lattice_)

        if axis[0] % 2 == 1:
            lattice_ = lattice_[:,:,::-1]
        
        if axis[1] % 2 == 1:
            lattice_ = lattice_[:,::-1, :]

        lattice_ = np.roll(lattice_, (-1, -1), axis=(1,1))
        lattice_num_array_ = lattice_.reshape(lattice_num_array_.shape)

        return lattice_num_array_

        

        
        
        
    
    @staticmethod
    def decompose(v, a1, a2):
        
        b = np.array([np.dot(v,a1), np.dot(v, a2)])
        
        A = np.concatenate((a1.reshape(-1,1), a2.reshape(-1,1)), axis=1)
        B = A.T @ A
        
        return (np.linalg.inv(B) @ b).T
    
    def to_lattice_vec(self, V):
        
        V_ = V.reshape(-1,2)
        
        l_vec = np.zeros((V_.shape[0],2),dtype=np.float32)
        
        for i, v in enumerate(V_):
            w = self.decompose(v, self.a1, self.a2)
            
            l_vec[i] = w
        
        return l_vec.reshape(V.shape)
    
    
    def dimer_corr(self,l, a):
        
        '''
        l : 2 x 2 matrix l[0] = r_i is first index of hexagon, l[1] = r_j is second one
        a : 2 dimensional vector, specify which dimer which direct to a * np.pi*(1/3) will be chosen.
        
        '''
        # assert l.shape[0] == 2, 'l must have exactly two edges'
        # assert a.shape[0] == 2, 'a.shape[0] should be 2'

        assert (0<=l[:,0] ).all() and (l[:,0] <self.l[0]).all(), 'l[:,0] must be interger in interval[0,{}]'.format(self.l[0])
        assert (0<=l[:,1] ).all() and (l[:,1] <self.l[1]).all(), 'l[:,1] must be interger in interval[0,{}]'.format(self.l[1])
        assert (0 <= a).all() and (a <=5).all(), 'a musb be integer in interval[0,5]'
        
        hex_edges, color = self.edges_from_hex_(l = self.all_hex_index, color=True, num =True)
        

        return hex_edges[self.l[1] * l[:, 0] + l[:, 1], a ], color[self.l[1] * l[:, 0] + l[:, 1], a ]
    
    @staticmethod
    def Rotation(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).astype(np.float32)



get_conn2 = nk.operator.DimerLocalOperator2._get_conn_flattened_kernel
log_val_kernel = nk.machine.rbm.RbmSpin._log_val_kernel

#N = 200 give the best performance.

import numba as nb
class dynamics3: 
    def __init__(self,op,ma):
        
        self.local_states = np.sort(op._local_states)
        self.basis = op._basis[::-1].copy()
        self.constant = op._constant
        self.diag_mels = op._diag_mels
        self.n_conns = op._n_conns
        self.mels = op._mels
        self.x_prime = op._x_prime
        self.acting_on = op._acting_on
        self.acting_size = np.int64(op._acting_size[0])
        self.ma = ma


        self._w = ma._w
        self._a = ma._a
        self._b = ma._b
        self._r = ma._r
        
    
    
    def dynamics(self, X, time_list, E0):
        
        
        
        
        return self._dynamics(
            X, 
            time_list, 
            E0,
            self.local_states,
            self.basis,
            self.constant,
            self.diag_mels,
            self.n_conns,
            self.mels,
            self.x_prime,
            self.acting_on,
            self.acting_size,
            self._w,
            self._a,
            self._b,
            self._r,
                
        )
    
    
    @staticmethod
    @njit
    def _dynamics(
            X,
            time_list,
            E0,
            _local_states,
            _basis,
            _constant,
            _diag_mels,
            _n_conns,
            _mels,
            _x_prime,
            _acting_on,
            _acting_size,
            _w,
            _a,
            _b,
            _r,
            ):
        
        # basis is float64[:]
        
        
        batch_size = X.shape[0]
        t_d = time_list[1]-time_list[0]
        t_end = np.shape(time_list)[0]
        p_array = np.zeros((batch_size,t_end,X.shape[1]),dtype= np.int8) + 3
        P = p_array
        t_s = time_list[0]
        X = X.astype(np.int8)
        assert t_s == 0
        
        
#         for j in range(X.shape[0]):
            # print('done ', j/X.shape[0])
#             p = p_array[j]
            # x = X[j]
        time = np.zeros(batch_size)
        t_index_b = np.zeros(batch_size, dtype = np.int64)
        sections = np.zeros(batch_size + 1, dtype = np.int64)
        a_0 = np.zeros(batch_size)
#         continue_index = np.ones(batch_size, dtype=nb.boolean)
        continue_index = np.arange(batch_size)
        ci = continue_index.copy()
        m = 0
        
        while True:
            
                     
            x_prime, mels = get_conn2(
                                X,
                                sections[1:],
                                _basis,
                                _constant,
                                _diag_mels,
                                _n_conns,
                                _mels,
                                _x_prime,
                                _acting_on,
                                _acting_size)

            # state = x_prime.copy()
            # state = state.astype(np.float64)

            log_val_prime = np.real(log_val_kernel(x_prime.astype(np.float64), None, _w, _a, _b, _r))
            
            for n in range(batch_size):
                log_val_prime[sections[n] : sections[n+1]] -= log_val_prime[sections[n]]

            mels = np.real(mels) * np.exp(log_val_prime)
            N_conn = sections[1:] - sections[:-1] - 1
            for n in range(batch_size):
                a_0[n] = (-1)* mels[sections[n] + 1: sections[n+1]].sum()
#             print(a_0[0], N_conn[0], mels[sections[0] + 1: sections[1]])
                
#             print((a_0 - mels[sections[:-1]]).mean(), mels[sections[:-1]].mean())

                
            r_1 = np.random.rand(batch_size)
            r_2 = np.random.rand(batch_size)
            
            tau = np.log(1/r_1)/a_0
            time += tau

            t_index = ((time // t_d) + 1).astype(np.int64)  
            over_index = (t_index >= t_end)
            t_index[over_index] = t_end
            
            m += 1
            
            
           
            
            for n in range(batch_size):
                P[n, t_index_b[n] : t_index[n]] = X[n]
                
            if over_index.all():
                p_array[continue_index] = P
                break
                
            t_index_b = t_index
            

            
            
            for n in range(batch_size):
                s = 0
                for i in range(N_conn[n]):
                    s -= mels[sections[n] + 1 + i]
                    if s >= r_2[n] * a_0[n]:
                        X[n] = x_prime[sections[n] + 1 + i]
                        break


                
            if over_index.any():
                
                p_array[continue_index] = P
                
                
                tci = np.logical_not(over_index).astype(nb.boolean)
                continue_index = continue_index[tci]
                
                P = p_array[continue_index]
                
                batch_size = np.sum(tci)
                P = p_array[continue_index]
                X = X[tci]
                time = time[tci]
                t_index_b = t_index_b[tci]
                sections = np.zeros(batch_size + 1, dtype = np.int64)
                a_0 = np.zeros(batch_size)
        return p_array
            
        
    def run(self, basis, t_list, E_0, qout):
        
        out = self.dynamics(basis, t_list, E_0)
        
        qout.put(out)
    
        
    def multiprocess(self, basis, t_list, E0 , n = 1):
        queue = []
        process = []
        N = basis.shape[0] 
        index = np.round(np.linspace(0, N, n + 1)).astype(np.int)

        for i in range(n):
            queue.append(mp.Queue())
            p = mp.Process(target=self.run, args=(basis[index[i]:index[i+1]], t_list, E0 ,queue[i]))
            p.start()
            process.append(p)


        out = []
        for q in queue:
            out.append(q.get())

        
        return np.vstack(out)