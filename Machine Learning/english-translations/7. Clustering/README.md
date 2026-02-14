## Table of Contents
- [1. Are Clustering Algorithms Unsupervised?](#1-are-clustering-algorithms-unsupervised)
- [2. K-Means](#2-k-means)
- [3. Gaussian Mixture Model (GMM)](#3-gaussian-mixture-model-gmm)
- [4. Clustering Evaluation](#4-clustering-evaluation)
- [5. Code Implementation](#5-code-implementation)

## 1. Are Clustering Algorithms Unsupervised?

Clustering groups data points into clusters such that points in the same cluster are similar. **Clustering is unsupervised.** Common algorithms: K-Means, GMM, SOM.

## 2. K-Means

Iteratively: (1) assign points to nearest centroid; (2) update centroids as cluster means. Use distortion (sum of squared distances) as the objective. Choose K via elbow method or cross-validation.

## 3. Gaussian Mixture Model (GMM)

GMM assumes data is a mixture of Gaussians. Uses EM: E-step assigns soft memberships; M-step updates means, variances, and weights. GMM gives probability of belonging to each cluster.

## 4. Clustering Evaluation

Assess: (1) clustering tendency; (2) number of clusters (e.g., elbow, Gap statistic); (3) cluster quality (e.g., silhouette, Davies–Bouldin).

## 5. Code Implementation

- [GMM](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/7.%20Clustering/GMM.ipynb)
- [K-Means](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/7.%20Clustering/K-Means.ipynb)

------

> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
