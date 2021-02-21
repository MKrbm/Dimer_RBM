import numpy as np


class graph_hex:
    
    def __init__(self, length = [4, 4], pbc = True):
        
        self.length = length
        
        if length[1] % 2 != 0:
            print("length[1] = {} is not preferable ".format(length[1]))
        
        if pbc:
            self.n_nodes = np.prod(length) * 2
        elif not pbc:
            self.n_nodes = length[1] * 2 + (length[1] + 1) * length[0]
        
        
        edges = self.edges(color = True)
        self.edge_dic = {
            "a" : [],
            "b" : [],
        }
        
        for edge in edges:
            self.edge_dic[edge[1]].append(edge[0])
        
        self.edges_ = np.array(self.edges())
        self.edges_c = self.edges(color=True)


    

    def edges(self, color = False):
        
        edges = []
        
        L = self.length[1]
        
        
        for n in list(range(self.n_nodes+1))[1:]:
            
            if self.where_in_hex(n) == 0:
                
                a = (n + L -1 ) % self.n_nodes + 1
                b = n+2*L-1 if (n % L) == 1 else n-1+L
                
                b = (b - 1) % self.n_nodes + 1
                
        
                if not color:
#                     edges.append(sorted([n, a]))
#                     edges.append(sorted([n, b]))
                    edges.append([n, a])
                    edges.append([n, b])
                else :
#                     lis = sorted([n, a])
                    lis = [n, a]
#                     lis.append("a")
                    edges.append([lis,"a"])
                    
#                     lis = sorted([n, b])
                    lis = [n, b]
#                     lis.append("a")
                    edges.append([lis,"a"])
                
            
            else: 
                a = (n + L - 1) % self.n_nodes + 1
                if not color:
#                     edges.append(sorted([n, a]))
                    edges.append([n, a])
                else:
                    if n%2 == 1:
#                         lis = sorted([n, a])
                        lis = [n, a]
#                         lis.append("b")
                        edges.append([lis,"b"])
                    else:
#                         lis = sorted([n, a])
                        lis = [n, a]
#                         lis.append("a")
                        edges.append([lis,"a"])
        return edges
                
            
#         @property
#         def edges_with_color(self):

    def where_in_hex(self, n):
        # return 0 if n is at the top of a hex and 1 if n is at the bottom of a hex.
        return ((n - 1) // self.length[1]) % 2 
    
    
    def edge_color(self, edge):
        
        if edge in self.edge_dic["a"]:
            return "a"
        else:
            return "b"
    

    def where_edge(self, edge):
        
        if not self.is_in_edges(edge):
            raise ValueError('edge = {} is not in edges (may be because of order)'.format(edge))
        
        L = self.length[1]
        
        if self.where_in_hex(edge[0]):
            return 3
        elif (edge[1] - edge[0]) % self.n_nodes == L:
            return 1
        else:
            return 2
        
    def is_in_edges(self, edge):
        
        if (edge  in self.edges()) or (edge[::-1] in self.edges()):
            return True
        else:
            return False
        
    def ad_edges(self, n, color = False):
        ad_edges = []
        if not color:
            for edge in self.edges():
                if n in edge:
                    ad_edges.append(edge)
            return ad_edges
        
        else:
            for edge in self.edges_c:
                if n in edge[0]:
                    ad_edges.append(edge)
            return ad_edges

        
    
    def second_ad_edges(self, n):
        sec_ad_edges = []
        
        ad_edges = self.ad_edges(n)
        
        for ad_edge in ad_edges:
            ad_edge_copy = ad_edge.copy()
            ad_edge.remove(n)
            ad_site = ad_edge[0]
            can_sec_ad_edge = self.ad_edges(ad_site)
            
            try:
                can_sec_ad_edge.remove(ad_edge_copy)
            except:
                can_sec_ad_edge.remove(ad_edge_copy[::-1])
            
            sec_ad_edges.extend(can_sec_ad_edge)
            
        return sec_ad_edges
    
    
    def potential(self):
        
        potential_edges = []
        
        for edge in self.edges(color = True):
            
            edge_color_1 = edge[1]
            num = self.where_edge(edge[0])
            
            sec_ad_edges = self.second_ad_edges(edge[0][1])
            
            for sec_edge in sec_ad_edges:
#                 print(sec_edge)
                if self.where_edge(sec_edge) == num:
                    edge_color_2 = self.edge_color(sec_edge)
                    
#                     print(edge, sec_edge)
                    potential_edges.append([edge[0] + sec_edge, [edge_color_1, edge_color_2]])
            
        return potential_edges


import collections

class graph_hex_2(graph_hex):
    
    def potential(self):
        
        potentials = super().potential()
        unique_pot = []
        for i, [edge_1, color_1] in enumerate(potentials):
            
            unique = True
            # for  [edge_2, color_2] in potentials[i+1:]:
            #     if self.equal_4_edge(edge_1, edge_2):
            #         # print(edge_1, edge_2)
            #         unique = False
                    
            if unique:
                # edge_1 = sorted(edge_1[:2]), sorted(edge_1[2:])
                edge = [edge_1[:2],edge_1[2:]]
                unique_pot.append(self.sorting(edge_1[:2],edge_1[2:],color_1))

        return sorted(unique_pot)

    @staticmethod
    def equal_4_edge(edge_1, edge_2):
        return collections.Counter(edge_1) == collections.Counter(edge_2) and (collections.Counter(edge_1[:2]) == collections.Counter(edge_2[:2])or collections.Counter(edge_1[:2]) == collections.Counter(edge_2[2:]))
    
    @staticmethod
    def sorting(edge_1,edge_2,color):
        edge = [edge_1, edge_2]
        indices = sorted(range(len(edge)), key = lambda k : edge[k])

        if indices == [0,1]:
            return [edge_1, edge_2, color]
        else:
            color.reverse()
            return [edge_2, edge_1, color]
                    
            

                
            
            
            
            
class graph_hex_3(graph_hex_2):
    
    def finite_conf(self):
        '''
        assume up spin is on lattice 1
        '''
        
    
        N = 2 ** self.n_nodes
        cand_conf = np.zeros([N,self.n_nodes])
        for n in range(N):
            m = n
            for i in range(self.n_nodes):
                cand_conf[n,i] = 2*(m//(2**(self.n_nodes-1-i)) - 1/2)
                m %= 2**(self.n_nodes-1-i)
        self.cand_conf = cand_conf.astype('int')
        
        finite_conf = []
        for conf in self.cand_conf:
            
            is_finite_conf = np.zeros(np.prod(self.length))
            for edge in self.edges_c:
                x = conf[edge[0][0] - 1] 
                y = conf[edge[0][1] - 1]
                color = edge[1]
                
                r = x != y if color=='a' else x==y
#                 time.sleep(3)
#                 print(x,y, edge[0][0],edge[0][1], conf)
                dual_edge=self.where_in_honeycomb([edge[0][0],edge[0][1]])
                is_finite_conf[dual_edge[0]-1] += r
                is_finite_conf[dual_edge[1]-1] += r
#             print(is_finite_conf)
            if np.prod(is_finite_conf) == 1:
                if conf[0] == 1:
                    break
                finite_conf.append(conf)

                print(conf) 
        
        return np.array(finite_conf)
    
    def is_dimer_basis(self,confs):

        assert confs.shape[1] == self.n_nodes, 'the size of candidate configuration is wrong'
        is_dimer_basis = np.zeros(confs.shape[0], dtype=bool)
        for i, conf in enumerate(confs):
            is_finite_conf = np.zeros(np.prod(self.length))
            for edge in self.edges_c:
                x = conf[edge[0][0] - 1] 
                y = conf[edge[0][1] - 1]
                color = edge[1]
                
                r = x != y if color=='a' else x==y
    #                 time.sleep(3)
    #                 print(x,y, edge[0][0],edge[0][1], conf)
                dual_edge=self.where_in_honeycomb([edge[0][0],edge[0][1]])
                is_finite_conf[dual_edge[0]-1] += r
                is_finite_conf[dual_edge[1]-1] += r
    #             print(is_finite_conf)
            if np.prod(is_finite_conf) == 1:
                is_dimer_basis[i] = True
        return is_dimer_basis

    @property
    def hc(self):
        
        hc = {}
        y = self.length[1]
        for i in range(self.length[0]):
            for j in range(self.length[1]):
                
                hc[(i + 1,j + 1)] = []
                hc[(i + 1,j + 1)].append((j + 1) + 2*y*i )
                hc[(i + 1,j + 1)].append(hc[(i + 1,j + 1)][0] + y)
                b = (hc[(i + 1,j + 1)][0]+y-1)
                a = b if b%y != 0 else b+y
                hc[(i + 1,j + 1)].append(a)
                hc[(i + 1,j + 1)].append((hc[(i + 1,j + 1)][1] + y - 1)%(self.n_nodes) + 1)
                hc[(i + 1,j + 1)].append((hc[(i + 1,j + 1)][2] + y - 1)%(self.n_nodes) + 1)
                hc[(i + 1,j + 1)].append(hc[(i + 1,j + 1)][4] + y)
                
#         self.hc = hc
        
        return hc
    
    def where_in_honeycomb(self, edge, num = True):
        
        
        dual_edge = []  
        for key in self.hc.keys():
            

                
            if (edge[0] in self.hc[key]) and (edge[1] in self.hc[key]):
                if num:
                    dual_edge.append((key[0] - 1) * self.length[1] + key[1])
                else:
                    dual_edge.append(key)
        
        return dual_edge


import numpy as np
import time
from numba import njit
class graph_hex_4(graph_hex_3):
    
    def __init__(self, *args, **keyargs):
        super().__init__(*args,**keyargs)
        edges_ = []
        for edge in self.edges_c:
        #     edge_=(edge[0] + [(1 if edge[1] == 'a' else -1)])
            edges_.append((edge[0] + [(-1 if edge[1] == 'a' else 1)]))
            
        self.edges_c = np.array(edges_) # third elements represent edge color -1 == 'a'(ferro) 1 == 'b'(antiferro)
        hc_values = np.array(list(self.hc.values()))
#         print(self.edges_)
        hc_values_array = np.stack([hc_values] * self.edges_.shape[0])
        self.edges_and_honeycomb = self.where_in_honeycomb(hc_values_array, self.edges_.reshape(-1,2,1,1))
        
    
    def is_dimer_basis2(self, cand):
        '''
        return true if candidate configuration is proper for dimer basis.
        '''
        cand = (cand.transpose()).copy()
#         X = (np.prod(cand[self.edges_-1],axis=1) * (self.edges_c[:,-1]) + 1) /2
#         X = X.reshape(-1,1) * self.edges_and_honeycomb
#         return np.prod(np.sum(X,axis=0) == 1)
#         X = np.prod(cand[self.edges_-1],axis=1)
#         return self.cal1(X, self.edges_c[:,-1], self.edges_and_honeycomb)

        X = (np.prod(cand[self.edges_-1],axis=1) * (self.edges_c[:,-1]).reshape(-1,1) + 1)/2
        X2 = (self.edges_and_honeycomb.reshape(-1,8,1) * X.reshape(-1,1,cand.shape[1]))
        X3 = np.sum(X2, axis=0)
#         X3 = self.cal2(X, cand.shape[1], self.edges_and_honeycomb)
        return (X3 == 1).all(axis=0)
    

    def for_transition(self):
        edge_array = np.array(list(zip(*self.edges(color=True)))[0])
        edge_color = np.array(list(zip(*self.edges(color=True)))[1])
        edge_color = np.vectorize(convert_color)(edge_color)
        potential_edge_1 = np.array(list(zip(*self.potential()))[0])
        potential_edge_2 = np.array(list(zip(*self.potential()))[1])
        potential_edge_color = np.array(list(zip(*self.potential()))[2])
        potential_edge_color = np.vectorize(convert_color)(potential_edge_color)
        potential_edges = np.concatenate((potential_edge_1, potential_edge_2), axis = 1)

        corresponding_potential_edges = np.empty((len(edge_array), 2, 4))
        corresponding_potential_color = np.empty((len(edge_array), 2, 2))
        for i, edge in enumerate(edge_array):
            unique = np.zeros(4).astype('int64')
            l = 0
            for j, potential_edge in enumerate(potential_edges):
                is_edge1 = (edge[0] in potential_edge[:2]) and (edge[1] in potential_edge[2:])
                is_edge2 = (edge[0] in potential_edge[2:]) and (edge[1] in potential_edge[:2])
                if is_edge1 or is_edge2:
                    
                    if (unique == np.sort(potential_edge)).all():
                        continue
                    corresponding_potential_edges[i,l,:] = potential_edge.astype('int64')
                    corresponding_potential_color[i,l,:] = potential_edge_color[j]
                    l += 1
                    unique = np.sort(potential_edge)
        corresponding_potential_edges = (corresponding_potential_edges - 1).astype('int64').reshape(3 * np.prod(self.length),2,2,2) 
        edge_array = edge_array - 1
        corresponding_potential_color = corresponding_potential_color.astype('int64')


        for e_, ce_ in zip(edge_array, corresponding_potential_edges):
            for j in range(2):
                index = np.logical_or(ce_[j] == e_, ce_[j] == e_[::-1])
                index0 = np.arange(2) if index[0][1]  else np.arange(2)[::-1]
                index1 = np.arange(2)[::-1] if index[1][1]  else np.arange(2)
                ce_[j][0] = ce_[j][0][index0]
                ce_[j][1] = ce_[j][1][index1]


        return edge_array, edge_color, corresponding_potential_edges, corresponding_potential_color

    @staticmethod
#     @njit
    def cal1(X, edge_color, eah):
        c = (X * edge_color + 1) /2
#         c1 = c.reshape(-1,1) * eah
        return np.prod(np.sum(c.reshape(-1,1) * eah,axis=0) == 1)

    @staticmethod
    @njit
    def cal2(X, shape3, eah):
        X1 = (eah.reshape(-1,8,1) * X.reshape(-1,1,shape3))
        return np.sum(X1, axis=0)
    
    def get_honeycomb_andedge(self):
        
        return self.edges_and_honeycomb, self.edges_c

        
        
    @staticmethod
    @njit('(i8[:,:])(i8[:,:,:], i8[:,:,:,:])')
    def where_in_honeycomb(hc_values, edge):
        '''
        hc_values : this is 3 dimensional tensor that shape[0] == edge.shape[0]
        '''
    #         hc_values_ = np.stack([hc_values] * 10)
        A = np.sum(hc_values - edge[:,0,:] == 0,axis=2)
        B = np.sum(hc_values - edge[:,1,:] == 0,axis=2)
        C = A * B
        return C        

def convert_color(x):
    
    r = x
    if x == 'a':
        r = 1
    elif x == 'b':
        r = -1
    return r