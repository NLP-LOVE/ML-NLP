# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt
from gmm import *

# Debug mode
DEBUG = True

# Load data
Y = np.loadtxt("gmm.data")
matY = np.matrix(Y, copy=True)

# Number of models, i.e., number of clusters
K = 2

# Compute GMM parameters
mu, cov, alpha = GMM_EM(matY, K, 100)

# Cluster samples according to GMM; one model per class
N = Y.shape[0]
# Compute responsibility matrix under current parameters
gamma = getExpectation(matY, mu, cov, alpha)
# For each sample, take argmax of responsibilities as class label
category = gamma.argmax(axis=1).flatten().tolist()[0]
# Assign each sample to its class
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])

# Plot clustering result
plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()
