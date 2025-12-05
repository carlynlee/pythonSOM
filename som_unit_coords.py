import sys
import numpy as np
from som_topol_struct import som_topol_struct


class som_unit_coords:
    def __init__(self, topol=None, lattice="rect", shape="sheet"):
        if topol is None:
            sTopol = som_topol_struct(lattice="rect", mapshape="sheet")
            sTopol.msize = np.array([[0]])
        else:
            sTopol = topol

        msize = sTopol.msize
        lattice = sTopol.lattice
        shape = sTopol.mapshape

        if msize.shape[1] == 2:
            munits = int(msize[0, 0] * msize[0, 1])
        elif msize.shape[1] == 1:
            msize_temp = msize[0, 0]
            msize = np.array([[0, 0]])
            msize[0, 0] = msize_temp
            msize[0, 1] = 1
            sTopol.msize = msize  # Update topology with corrected msize
            munits = int(msize[0, 0] * msize[0, 1])
        else:
            sys.exit("Map size only supports two dimensions")

        mdim = msize.shape[1]

        Coords = np.zeros((munits, mdim))

        k = np.zeros((1, mdim))
        k[0, 0] = 1
        k[0, 1] = msize[0, mdim - 1]

        # Generate unit indices
        inds = np.arange(munits).reshape(-1, 1)

        # Calculate coordinates based on lattice type
        if lattice == "hexa":
            # Hexagonal lattice coordinates
            for i in range(munits):
                row = i // int(msize[0, 1])
                col = i % int(msize[0, 1])
                Coords[i, 0] = row
                if row % 2 == 0:
                    Coords[i, 1] = col
                else:
                    Coords[i, 1] = col + 0.5
        else:
            # Rectangular lattice coordinates
            for i in range(munits):
                row = i // int(msize[0, 1])
                col = i % int(msize[0, 1])
                Coords[i, 0] = row
                Coords[i, 1] = col

        self.coords = Coords
        self.inds = inds
