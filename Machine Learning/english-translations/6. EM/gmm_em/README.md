# GMM-EM Clustering

EM algorithm implementation for Gaussian Mixture Model (GMM) clustering.

**Notes:**
- Before applying GMM EM, scale all sample values to [0, 1].
- When initializing parameters, ensure no two components have identical parameters, or they will remain identical and collapse.
- K (number of components) must be > 1. K=1 amounts to a single cluster and is meaningless.

**Code:** `main.py` and `gmm.py` in this directory.

## Related Article

[GMM EM Algorithm Implementation in Python](http://www.codebelief.com/article/2017/11/gmm-em-algorithm-implementation-by-python/)
