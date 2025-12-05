import numpy as np


class som_train_struct:
    def __init__(self, data=None, data_name="", time="", dim=None, dlen=None, msize=None, munits=None, 
                 neigh="", phase="", algorithm="", mask=None, previous=None):
        self.type = "som_train"
        self.algorithm = algorithm
        self.data_name = data_name
        self.neigh = neigh
        self.mask = mask
        self.radius_ini = []
        self.radius_fin = []
        self.alpha_ini = []
        self.alpha_type = "inv"
        self.trainlen = []
        self.time = time

        if previous is not None:
            sTprev = previous
        else:
            sTprev = None

        if data is None:
            D = None
        else:
            D = data
            dlen = D.shape[0]
            dim = D.shape[1]

        # dim
        if sTprev is not None and dim is None:
            dim = sTprev.mask.shape[0]

        # mask
        if mask is None or (hasattr(mask, 'shape') and mask.shape[1] == 0):
            if dim is not None:
                self.mask = np.ones((dim, 1))
            else:
                self.mask = np.array([]).reshape(0, 0)

        # msize, munits
        if msize is None:
            msize = np.zeros(2)
            msize[0] = 10.0
            msize[1] = 10.0
        else:
            if munits is not None:
                s = np.round(np.sqrt(munits))
                msize = np.array([s, np.round(munits / s)])

        if munits is None:
            munits = msize[0] * msize[1]

        # previous training
        prevalg = ""
        if sTprev is not None:
            if sTprev.algorithm == "lininit":
                prevalg = "init"
            else:
                prevalg = sTprev.algorithm

        # determine phase based on previous training
        if phase == "":
            if self.algorithm == "lininit" or self.algorithm == "randinit":
                phase = "init"
            elif self.algorithm == "batch" or self.algorithm == "seq" or self.algorithm == "":
                if sTprev is None:
                    phase = "rough"
                elif prevalg == "init":
                    phase = "rough"
                else:
                    phase = "finetune"
            else:
                phase = "train"

        # determine the algorithm
        if self.algorithm == "":
            if phase == "init":
                self.algorithm = "lininit"
            elif prevalg == "init" or prevalg == "":
                self.algorithm = "batch"
            else:
                self.algorithm = sTprev.algorithm

        # mask
        if self.mask.shape[0] == 0 or (hasattr(self.mask, 'shape') and self.mask.shape[1] == 0):
            if sTprev is not None:
                self.mask = sTprev.mask
            elif dim is not None:
                self.mask = np.ones((dim, 1))

        # neighborhood function
        if self.neigh == "":
            if sTprev is not None and sTprev.neigh != "":
                self.neigh = sTprev.neigh
            else:
                self.neigh = "gaussian"

        if phase == "init":
            self.alpha_ini = []
            self.alpha_type = ""
            self.radius_ini = []
            self.radius_fin = []
            self.trainlen = []
            self.neigh = ""
        else:
            mode = phase + '-' + self.algorithm

            # learning rate
            if self.alpha_ini == []:
                if self.algorithm == "batch":
                    self.alpha_ini = []
                else:
                    if phase == "train" or phase == "rough":
                        self.alpha_ini = 0.5
                    if phase == "finetune":
                        self.alpha_ini = 0.05

            if self.alpha_type == "":
                if sTprev is not None and self.alpha_type != "" and self.algorithm != "batch":
                    self.alpha_type = sTprev.alpha_type
                elif self.algorithm == "seq":
                    self.alpha_type = "inv"

            # radius
            ms = np.max(msize)
            if self.radius_ini == []:
                if sTprev is None or sTprev.algorithm == "randinit":
                    self.radius_ini = max(1.0, np.ceil(ms / 4))
                elif sTprev.algorithm == "lininit" or sTprev.radius_fin == []:
                    self.radius_ini = max(1.0, np.ceil(ms / 8))
                else:
                    self.radius_ini = sTprev.radius_fin

            if self.radius_fin == []:
                if phase == "rough":
                    self.radius_fin = max(1.0, self.radius_ini / 4.0)
                else:
                    self.radius_fin = 1.0

            # trainlen
            if self.trainlen == []:
                if munits is None or dlen is None:
                    mpd = 0.5
                else:
                    mpd = float(munits / dlen)

            if phase == "train":
                self.trainlen = np.ceil(50.0 * mpd)
            elif phase == "rough":
                self.trainlen = np.ceil(10.0 * mpd)
            elif phase == "finetune":
                self.trainlen = np.ceil(40.0 * mpd)

            self.trainlen = max(1.0, self.trainlen)
