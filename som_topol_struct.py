import numpy as np
from numpy import linalg


class som_topol_struct:
    def __init__(self, dlen=None, data=None, munits=None, msize=None, lattice="", mapshape="", topol=None):
        self.type = "som_topol"
        self.msize = msize  # needs to be a 2x1 array
        self.lattice = lattice
        self.mapshape = mapshape
        
        if data is None:
            D = np.array([])
            dim = 2
        else:
            D = data.data  # elements must be formatted as float, or this won't work right
            dlen = D.shape[0]
            dim = D.shape[1]

        if lattice == "":
            self.lattice = "hexa"

        if mapshape == "":
            self.mapshape = "sheet"

        # if necessary determine the number of map units
        if munits is None:
            if dlen is not None:
                munits = np.ceil(5 * np.sqrt(dlen))
            else:
                munits = 100.0

        # then determine the map size
        if msize is None or (hasattr(msize, 'shape') and msize.shape[0] == 0):
            self.msize = np.array([[0, 0]])
            if dim == 1:  # one dimensional data
                self.msize[0, 0] = 1
                self.msize[0, 1] = np.ceil(munits)
            elif D.shape[0] < 2:  # no data provided, determine msize using munits
                self.msize[0, 0] = np.round(np.sqrt(munits))
                self.msize[0, 1] = np.round(munits / self.msize[0, 0])
            else:  # determine a map size based on eigenvalues.
                # initialize xdim/ydim ratio using principal components of the input space;
                # the ratio is the square root of two largest eigenvalues
                A = np.zeros((dim, dim)) + np.inf  # autocorrelation matrix
                D_centered = D.copy()
                for i in range(0, dim):
                    mean_val = D_centered[:, i].sum(0) / D_centered[:, i].shape[0]
                    D_centered[:, i] = D_centered[:, i] - mean_val

                for i in range(0, dim):
                    for j in range(i, dim):
                        c = np.multiply(D_centered[:, i], D_centered[:, j])
                        A[i, j] = float(np.sum(c)) / c.shape[0]
                        A[j, i] = A[i, j]

                S, V = linalg.eig(A)
                I = np.argsort(S)
                eigval = S[I]

                if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
                    ratio = 1.0
                else:
                    ratio = np.sqrt(eigval[-1] / eigval[-2])  # ratio between xdim and ydim

                if self.lattice == "hexa":
                    self.msize[0, 1] = min(munits, np.round(np.sqrt(munits / ratio * np.sqrt(0.75))))
                else:
                    self.msize[0, 1] = min(munits, np.round(np.sqrt(munits / ratio)))

                self.msize[0, 0] = np.round(munits / self.msize[0, 1])

                if np.min(self.msize) == 1:
                    self.msize[0, 0] = 1
                    self.msize[0, 1] = np.max(self.msize)

                if self.lattice == "hexa" and self.mapshape == "toroid":
                    if self.msize[0, 0] % 2 == 1:
                        self.msize[0, 0] = self.msize[0, 0] + 1
