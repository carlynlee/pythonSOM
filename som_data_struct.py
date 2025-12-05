import numpy as np


class som_data_struct:
    def __init__(self, D, labels=None, name="", comp_names=None, comp_norm=None, label_names=None):
        self.type = "som_data"
        self.data = D
        self.labels = labels if labels is not None else []
        self.name = name
        self.comp_names = comp_names if comp_names is not None else []
        self.comp_norm = comp_norm if comp_norm is not None else []
        self.label_names = label_names if label_names is not None else []

        dlen = D.shape[0]
        dim = D.shape[1]

        # defaults
        if name == "":
            self.name = "unnamed"

        if self.labels == []:
            self.labels = [''] * dlen

        if self.comp_names == []:
            self.comp_names = [''] * dim
            for i in range(0, dim):
                self.comp_names[i] = 'variable ' + str(i)

        if self.comp_norm == []:
            self.comp_norm = [[] for i in range(dim)]
