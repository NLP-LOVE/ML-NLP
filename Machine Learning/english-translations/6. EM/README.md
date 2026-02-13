## Table of Contents
- [1. What is the EM Algorithm](#1-what-is-the-em-algorithm)
- [2. Which Models Use EM?](#2-which-models-use-em)
- [3. Code Implementation](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/6.%20EM/gmm_em)
- [4. References](#4-references)

## 1. What is the EM Algorithm

The **Expectation-Maximization (EM) algorithm** finds maximum likelihood or maximum a posteriori estimates in probabilistic models that depend on unobserved latent variables.

EM alternates between two steps:
- **E-step**: Compute expected values of latent variables given current parameter estimates.
- **M-step**: Maximize the likelihood with respect to parameters using the E-step expectations.

**Maximum likelihood** in one sentence: Given observed results, infer the parameters θ that made them most likely.

### 1.1 Likelihood Function

The **likelihood function** measures how likely the observed data is under different parameter values. Maximum likelihood = most plausible parameter given the data.

### 1.2 EM Algorithm (Coin Example)

With two coins A and B (unknown P(heads)), toss five rounds. If we knew which coin was used each round, we could estimate PA and PB directly. With the coin identity unknown (latent variable), we use EM: initialize PA, PB; E-step: assign each round to the coin that makes it more likely; M-step: re-estimate PA, PB from assignments; repeat until convergence.

## 2. Which Models Use EM?

Models solved with EM include GMM (Gaussian Mixture Model), collaborative filtering, and K-means. EM converges but may reach a local optimum.

## 3. Code Implementation

[GMM EM Algorithm](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/6.%20EM/gmm_em)

## 4. References

[Understanding the EM Algorithm](https://blog.csdn.net/v_july_v/article/details/81708386)

------

> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
