import numpy as np
from numba import njit, prange
import multiprocessing as mp

sigmaz = np.array([[1, 0], [0, -1]])
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,1],[-1,0]])

mszsz = np.kron(sigmaz, sigmaz)
mszsx = np.kron(sigmax, sigmax)
mszsy = np.kron(sigmay, sigmay)





class new_hex:
    
    def __init__(self, l = np.array([4, 2])):
        
        self.a1 = self.a(np.float32(0))
        self.a2 =  self.a(np.pi*(5/3))


        # definite lattice vector. If one change gage, this vectors will change correspondingly.
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
    
    def reverse(self, lattice_coor_array):

        '''
        lattice_num : index of lattice to be mirrored.
        0  1  2  3        
        4  5  6  7  
        8  9  10 11
        12 13 14 15
        '''
        lattice_coor_array_ = lattice_coor_array.reshape(-1, 2)

        cori  = np.argwhere(self.edges_color==-1)[0][0] #center of reverse index

        corc = self.lattice_coor[self.edges[cori]].sum(axis=0)/2 #center of reverse coordinate
        
        lattice_coor_array_prime = -(lattice_coor_array_ - corc) + corc

        lattice_coor_array_prime = self.ProcessPeriodic(lattice_coor_array_prime)

        return lattice_coor_array_prime.reshape(lattice_coor_array.shape)

    # @property
    def autom(self, reverse=False):
        coordinate = self.lattice_coor

        # return_array = self.coor_to_lattice_num(self.translation(coordinate, self.all_unit_cell))
        return_array_coor = self.translation(coordinate, self.all_unit_cell)
        if reverse:
            return_array_coor_ = return_array_coor.copy()
            return_array_coor_ = self.reverse(return_array_coor_)
            return_array_coor = np.concatenate((return_array_coor, return_array_coor_),axis=0)
        
        return self.coor_to_lattice_num(return_array_coor)

        

        
        
        
    
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








