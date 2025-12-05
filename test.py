# Carlyn Lee
# test.py
# wrapper to test Extr_PCA_Feature. Projects over randomly selected negative and positive samples from 'imputed-liver.txt' and generates two dimensional data using primary components
# thank you to Dr. Charles Lee for helping me write this script and for the POD

import numpy as np
import random
from Extr_PCA_Features import Extr_PCA_Features

nIndex = list(range(115, 191))
nni = len(nIndex)

primaryTIndexAll = [3]
primaryTIndexAll.extend(list(range(8, 112)))
npta = len(primaryTIndexAll)

infile = open('imputed-liver.txt', 'r')
xraw = []
for line in infile:
    line = line.strip()
    xraw.append(line.split())

infile.close()

for i in range(0, len(xraw)):
    for j in range(0, len(xraw[0])):
        xraw[i][j] = float(xraw[i][j])

xraw = np.matrix(xraw)

# extract positives and negatives from data set
pxraw = xraw[:, primaryTIndexAll]
nxraw = xraw[:, nIndex]
xraw = np.bmat('pxraw nxraw')

# subtract the means of each row to eliminate outliers
mean = xraw.sum(1) / xraw.shape[1]
xraw = xraw - mean

class_num = 2
primaryTIndexAll = list(range(0, npta))
nIndex = list(range(0 + npta, npta + nni))
Ngenes = xraw.shape[0]
Ns = xraw.shape[1]

N_class1 = len(primaryTIndexAll)
divide = random.sample(primaryTIndexAll, npta)

# separate positive and negative sets into train, validation, test sets
N_class1Train = int(round(N_class1 * 0.08))
N_class1Val = int(round(N_class1 * 0.32))
N_class1Test = int(round(N_class1 * 0.60))
TrainIndclass1 = divide[0:N_class1Train]
ValIndclass1 = divide[N_class1Train:N_class1Train + N_class1Val]
TestIndclass1 = divide[N_class1Train + N_class1Val:N_class1Train + N_class1Val + N_class1Test]

N_class2 = len(nIndex)
divide = random.sample(nIndex, nni)

N_class2Train = int(round(N_class2 * 0.08))
N_class2Val = int(round(N_class2 * 0.32))
N_class2Test = int(round(N_class2 * 0.60))
TrainIndclass2 = divide[0:N_class2Train]
ValIndclass2 = divide[N_class2Train:N_class2Train + N_class2Val]
TestIndclass2 = divide[N_class2Train + N_class2Val:N_class2Train + N_class2Val + N_class2Test]

# POD
Cls1projMatA = Extr_PCA_Features(xraw, TrainIndclass1)
Cls2projMatA = Extr_PCA_Features(xraw, TrainIndclass2)

# using only principal components (normalized between 0 and 1)
Nfea = 2
a = Cls1projMatA[:, 1]
a = np.matrix(a)
a_min = float(a.min(1)[0, 0])
a_max = float(a.max(1)[0, 0])
a = (a - a_min) / (a_max - a_min)

b = Cls1projMatA[:, 1]
b = np.matrix(b)
b_min = float(b.min(1)[0, 0])
b_max = float(b.max(1)[0, 0])
b = (b - b_min) / (b_max - b_min)

LiverData = np.bmat('a; b')

targets = list(np.ones(N_class1))
targets.extend(list(np.zeros(N_class2)))
targets = np.array(targets)

# construct train set and targets
extr_TrainInd = TrainIndclass1.copy()
extr_TrainInd.extend(TrainIndclass2)
TrainSet = LiverData[:, extr_TrainInd]
TrainTargets = targets[extr_TrainInd]

# construct validation set and targets
extr_ValInd = ValIndclass1.copy()
extr_ValInd.extend(ValIndclass2)
ValSet = LiverData[:, extr_ValInd]
ValTargets = targets[extr_ValInd]

# construct test set and targets
extr_TestInd = TestIndclass1.copy()
extr_TestInd.extend(TestIndclass2)
TestSet = LiverData[:, extr_TestInd]
TeTargets = targets[extr_TestInd]
