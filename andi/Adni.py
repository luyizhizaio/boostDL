from  scipy.sparse import csr_matrix as csr_matrix
import numpy as np

class Adni(object):
    # construction method
    def __init__(self,
                 c1 = 200, ## hyper parameter
                 c2 = 280, ## hyper parameter
                 c3 = 1800, ## hyper parameter
                 c4 = 140, ## hyper parameter
                 c5 =20, ## hyper parameter
                 c6 = 60, ## hyper parameter
                 phi = 0.8, ## conductance
                 k = 100, ## user num
                 b = 100): ## group size

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.phi = phi
        self.k = k
        self.b = b
        self.V = 0
        self.mu = 0
        self.tlast = 0
        self.epsilon = 0
        self.l = 0

    def initialize(self, y):
        value = np.array([1]* len(y))
        return csr_matrix((value, (y, np.array([0]*len(y)))),shape=(self.V, 1),dtype=np.float)

    def diffusion(self, X, y):
        ### X is CSR_Matrix
        ### y is numpy array (the indices for seed nodes)
        # get the number of vertices

        # floating point operation
        if X.dtype != np.float:
            X = X.astype(np.float)
        self.V, _ = self.nodes(X)
        # degree of graph
        self.mu = self.degree(X)
        # hyper-parameter setting
        self.tlast, self.epsilon, self.l = self.tlast_epsilon_l_cal()
        # use for iterate check stop condition
        d = X.sum(axis = 1)
        # setting M matrix
        row = np.array(range(self.V))
        D = csr_matrix((d.getA1(),(row,row)),shape=(self.V,self.V)) # use for get M
        # identity matrix
        value = np.array([1] * self.V)
        I = csr_matrix((value, (row, row)), shape=(self.V, self.V))
        del row, value
        # Matrix
        M = (X.dot(D.power(-1)) + I) / 2

        d = csr_matrix(d)
        # for iterate over degree
        trunc_value = d * self.epsilon

        del X, I, D
        # initialize the r vector
        r = self.initialize(y)
        t = 1
        while(t <= self.tlast):
            # one step random walk
            q = M.dot(r)
            # truncate value
            mask = q >= trunc_value
            r = q.multiply(mask)
            # permutation function set
            s = q.multiply(d.power(-1))
            # evaluate the length
            if(s.getnnz() < self.k):
                t += 1
                continue
            # evaluate whether contains contains element more than k users
            if((np.nonzero(s.indptr[1:] - s.indptr[:-1])[0] < self.k).sum() < self.k):
                t += 1
                continue
            # order by S set value
            order = np.argsort(s.data)[::-1]
            # nonzero values(sorted from S set)
            values = s.data[order]
            indices = np.nonzero(s.indptr[1:] - s.indptr[:-1])[0][order]
            ordered_degree = d[indices].toarray()
            j = self.k
            sum_of_degree = ordered_degree[:j].sum()
            ## check the condition
            qualify = False
            users = (indices[:j] < self.k).sum()
            while(j < s.getnnz()):
                sum_of_degree += ordered_degree[j]
                #calculate the potential users
                users += indices[j] < self.k
                # condition 1
                condition_1 = users >= self.k
                # condition 2
                condition_2 = (sum_of_degree >= (2 << self.b)) and (sum_of_degree < self.mu * 5.0 / 6)
                # condition 3
                condition_3 = values[j] >= (1 / (self.c4) ) * (self.l +  2) * (2 << self.b)
                qualify = condition_1 and condition_2 and condition_3
                if(qualify):
                    break
                else:
                    j += 1

            if(qualify):
                return indices[:j + 1]
            else:
                t+= 1
        return 'Got A Diffusion Failure'

    def tlast_epsilon_l_cal(self):
        l = np.ceil(np.log2(self.mu / 2.0))
        tlast = (l + 1) * np.ceil(2.0 / np.square(self.phi) * np.log(self.c1 * (l + 2)) * np.sqrt(self.mu / 2.0))
        epsilon = 1.0 / (self.c3 * (l + 2) * tlast * (2 << self.b))
        return tlast, epsilon, l

    def degree(self, X):
        return X.sum()

    def nodes(self, X):
        return X.get_shape()





if __name__ == "__main__":
    # the user indices should be less than k that is from 0 to k - 1
    # initialized a symetric matrix for (the graph)
    row = np.array([0,2,0,3,1,2,1,3])
    col = np.array([2,0,3,0,2,1,3,1])
    data = np.array([2, 2, 3, 3,2, 2, 3, 3])
    X = csr_matrix((data, (row, col)), shape=(4, 4),dtype=float)
    # y is the seed nodes indices
    y = np.array([2,3])
    k = 2 # user node
    b = 2 # group size
    adni = Adni(b=b,k=b)
    result = adni.diffusion(X, np.array([2,3]))
    # result contains the user nodes
    for each in result:
        if each < k:
            print "Diffusion to User %d" % each

