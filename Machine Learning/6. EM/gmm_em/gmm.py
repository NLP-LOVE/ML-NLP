# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

DEBUG = True

######################################################
# Debug output function
# Controlled by global DEBUG
######################################################
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


######################################################
# Gaussian density for model k
# Row i = probability of sample i under each model
######################################################
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


######################################################
# E-step: compute responsibility of each model for each sample
# Y: sample matrix, one sample per row
# mu: mean array, cov: covariance array, alpha: model weights
######################################################
def getExpectation(Y, mu, cov, alpha):
    # Number of samples
    N = Y.shape[0]
    # Number of models
    K = alpha.shape[0]

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # Responsibility matrix: rows=samples, cols=responsibilities
    gamma = np.mat(np.zeros((N, K)))

    # Probability of each sample under each model
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # Compute responsibility of each model for each sample
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


######################################################
# M-step: update model parameters
# Y: sample matrix, gamma: responsibility matrix
######################################################
def maximize(Y, gamma):
    # Number of samples and features
    N, D = Y.shape
    # Number of models
    K = gamma.shape[1]

    # Initialize parameters
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # Update each model's parameters
    for k in range(K):
        # Sum of responsibilities for model k over all samples
        Nk = np.sum(gamma[:, k])
        # Update mu (mean per feature)
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        # Update cov
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
        # Update alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


######################################################
# Data preprocessing: scale all data to [0, 1]
######################################################
def scale_data(Y):
    # Scale each feature dimension
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


######################################################
# Initialize model parameters
# shape: (num_samples, num_features), K: number of models
######################################################
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


######################################################
# GMM EM algorithm
# Given sample matrix Y, compute model parameters
# K: number of models, times: number of iterations
######################################################
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha
