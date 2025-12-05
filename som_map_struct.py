import sys
import numpy as np
import scipy.linalg
import math
import datetime
import random
from som_topol_struct import som_topol_struct
from som_train_struct import som_train_struct
from som_unit_coords import som_unit_coords

now = datetime.datetime.now()


class som_map_struct:
    def __init__(self, dim, name="", topol=None, msize=None, lattice="", mapshape="", labels=None, 
                 neigh="", mask=None, comp_names=None, comp_norm=None):
        self.type = "som_map"
        self.neigh = neigh
        self.mask = mask
        self.name = name
        self.comp_names = comp_names if comp_names is not None else []
        self.comp_norm = comp_norm if comp_norm is not None else []
        self.labels = labels if labels is not None else []

        if topol is None:
            sTopol = som_topol_struct(lattice=lattice, mapshape=mapshape)
        else:
            sTopol = topol

        if msize is None or (hasattr(msize, 'shape') and msize.shape[1] == 0):
            sTopol.msize = np.array([[0, 0]])
        else:
            sTopol.msize = msize

        codebooklen = int(sTopol.msize[0, 1] * sTopol.msize[0, 0])
        self.codebook = np.zeros((codebooklen, dim))

        for i in range(0, codebooklen):
            for j in range(0, dim):
                self.codebook[i, j] = random.random()

        if mask is None or (hasattr(mask, 'shape') and mask.shape[1] == 0):
            self.mask = np.ones((dim, 1))
        else:
            self.mask = mask

        if self.labels == []:
            self.labels = [''] * int(sTopol.msize[0, 0] * sTopol.msize[0, 1])

        if sTopol.msize[0, 0] == 0 and sTopol.msize[0, 1] == 0:
            sTopol.msize = np.array([[0]])

        self.topol = sTopol

        if neigh == "":
            self.neigh = "gaussian"

        if name == "":
            self.name = 'SOM ' + now.strftime("%Y-%m-%d %H:%M")

        if self.comp_norm == []:
            self.comp_norm = [''] * dim

        if self.comp_names == []:
            self.comp_names = [''] * dim
            for i in range(0, dim):
                self.comp_names[i] = 'variable ' + str(i)

        sTrain = som_train_struct(time=now.strftime("%Y-%m-%d %H:%M"), mask=self.mask)

        sTrain.algorithm = ""
        sTrain.data_name = ""
        sTrain.neigh = "gaussian"
        sTrain.radius_ini = []
        sTrain.radius_fin = []
        sTrain.alpha_ini = []
        sTrain.alpha_type = "inv"
        sTrain.trainlen = []

        self.trainhist = sTrain

    def print_all(self):
        print("      type: " + self.type)
        print("  codebook: " + str(self.codebook.shape[0]) + " " + str(self.codebook.shape[1]))
        print("     topol:")
        print("          type: " + self.topol.type)
        print("         msize: " + str(self.topol.msize))
        print("       lattice: " + self.topol.lattice)
        print("         shape: " + self.topol.mapshape)
        print("    labels: " + str(len(self.labels)))
        print("     neigh: " + self.neigh)
        print("      mask: " + str(self.mask.shape[0]) + " " + str(self.mask.shape[1]))
        print(" trainhist: ")
        print("          type: " + self.trainhist.type)
        print("     algorithm: " + self.trainhist.algorithm)
        print("     data_name: " + self.trainhist.data_name)
        print("         neigh: " + self.trainhist.neigh)
        print("          mask: " + str(self.trainhist.mask.shape[0]) + " " + str(self.trainhist.mask.shape[1]))
        print("    radius_ini: " + str(self.trainhist.radius_ini))
        print("    radius_fin: " + str(self.trainhist.radius_fin))
        print("     alpha_ini: " + str(self.trainhist.alpha_ini))
        print("    alpha_type: " + self.trainhist.alpha_type)
        print("      trainlen: " + str(self.trainhist.trainlen))
        print("          time: " + self.trainhist.time)
        print("comp_names: " + str(len(self.comp_names)))
        print(" comp_norm: " + str(len(self.comp_norm)))

    def som_lininit(self, D):
        """Linear initialization of SOM codebook using PCA"""
        data_name = D.name
        comp_names = D.comp_names
        comp_norm = D.comp_norm
        D_data = D.data.copy()
        dlen = D_data.shape[0]
        dim = D_data.shape[1]

        if dlen < 2:
            sys.exit("Linear map initialization requires at least two samples")

        sTopol = self.topol
        if sTopol.msize.shape[1] == 1:
            msize = sTopol.msize
            sTopol.msize = np.array([[0, 0]])
            sTopol.msize[0, 0] = msize[0, 0]
            sTopol.msize[0, 1] = 1

        if sTopol.msize[0, 0] == 0 or sTopol.msize[0, 1] == 0:
            sys.exit("map needs to be m x n")

        self.topol = sTopol

        munits = self.codebook.shape[0]
        dim2 = self.codebook.shape[1]

        if dim2 != dim:
            sys.exit("Map and data must have the same dimensions")

        self.trainhist.algorithm = "lininit"
        self.trainhist.data_name = data_name

        msize = self.topol.msize
        mdim = msize.shape[1]
        munits = int(msize[0, 0] * msize[0, 1])

        nonzeromapdim = 0
        for i in range(0, mdim):
            if msize[0, i] > 1:
                nonzeromapdim = nonzeromapdim + 1

        if dim > 1 and nonzeromapdim > 1:
            A = np.zeros((dim, dim))
            me = np.zeros((1, dim))
            D_centered = D_data.copy()
            for i in range(0, dim):
                me[0, i] = np.mean(D_data[:, i])
                D_centered[:, i] = D_data[:, i] - me[0, i]

            for i in range(0, dim):
                for j in range(0, dim):
                    c = np.multiply(D_centered[:, i], D_centered[:, j])
                    A[i, j] = np.sum(c) / c.shape[0]
                    A[j, i] = A[i, j]

            S, V = scipy.linalg.eig(A)
            eigval = S
            ind = np.argsort(-1 * eigval)
            eigval = eigval[ind]
            V = V[:, ind]
            V = V[:, 0:mdim]
            eigval = eigval[0:mdim]

            for i in range(0, mdim):
                V[:, i] = V[:, i] / np.sqrt(np.dot(V[:, i], V[:, i])) * np.sqrt(eigval[i])
        else:
            me = np.zeros((1, dim))
            V = np.zeros((1, dim))
            for i in range(0, dim):
                me[0, i] = np.mean(D_data[:, i])
                V[0, i] = np.std(D_data[:, i])

        # Generate unit coordinates
        Coord = som_unit_coords(topol=self.topol)
        coords = Coord.coords

        # Initialize codebook using coordinates and principal components
        if dim > 1 and nonzeromapdim > 1:
            # Scale coordinates to match data variance
            # Normalize coordinates to [-1, 1] range
            coords_norm = coords.copy()
            for j in range(mdim):
                if np.max(coords[:, j]) > np.min(coords[:, j]):
                    coords_norm[:, j] = 2 * (coords[:, j] - np.min(coords[:, j])) / (np.max(coords[:, j]) - np.min(coords[:, j])) - 1
            
            # Initialize codebook using principal components
            for i in range(munits):
                self.codebook[i, :] = me[0, :]
                for j in range(min(mdim, V.shape[1])):
                    self.codebook[i, :] += coords_norm[i, j] * V[:, j].real
        else:
            # Simple initialization using mean and std
            coords_norm = coords.copy()
            if np.max(coords[:, 0]) > np.min(coords[:, 0]):
                coords_norm[:, 0] = (coords[:, 0] - np.mean(coords[:, 0])) / np.std(coords[:, 0])
            
            for i in range(munits):
                for j in range(dim):
                    if V[0, j] > 0:
                        self.codebook[i, j] = me[0, j] + coords_norm[i, 0] * V[0, j]
                    else:
                        self.codebook[i, j] = me[0, j]

        # Update component names and normalization if provided
        if comp_names:
            self.comp_names = comp_names
        if comp_norm:
            self.comp_norm = comp_norm
