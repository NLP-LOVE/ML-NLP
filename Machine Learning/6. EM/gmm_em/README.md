# gmm-em-clustering
高斯混合模型（GMM 聚类）的 EM 算法实现。

在给出代码前，先作一些说明。

- 在对样本应用高斯混合模型的 EM 算法前，需要先进行数据预处理，即把所有样本值都缩放到 0 和 1 之间。
- 初始化模型参数时，要确保任意两个模型之间参数没有完全相同，否则迭代到最后，两个模型的参数也将完全相同，相当于一个模型。
- 模型的个数必须大于 1。当 K 等于 1 时相当于将样本聚成一类，没有任何意义。

**代码在本目录下的main.py和gmm.py**

# 相关文章
[高斯混合模型 EM 算法的 Python 实现](http://www.codebelief.com/article/2017/11/gmm-em-algorithm-implementation-by-python/)
