## Table of Contents
- [1. What is Feature Engineering?](#1-what-is-feature-engineering)
  - [1.1 Feature Normalization](#11-feature-normalization)
  - [1.2 Categorical Features](#12-categorical-features)
  - [1.3 High-Dimensional Combination Features](#13-high-dimensional-combination-features)
  - [1.4 Text Representation Models](#14-text-representation-models)
  - [1.5 Other Feature Engineering](#15-other-feature-engineering)
  - [1.6 Feature Engineering Mind Map](#16-feature-engineering-mind-map)
- [2. Machine Learning Optimization Methods](#2-machine-learning-optimization-methods)
  - [2.1 Common Loss Functions](#21-common-loss-functions)
  - [2.2 What is Convex Optimization](#22-what-is-convex-optimization)
  - [2.3 Regularization Terms](#23-regularization-terms)
  - [2.4 Common Optimization Methods](#24-common-optimization-methods)
  - [2.5 Dimensionality Reduction](#25-dimensionality-reduction)
- [3. Machine Learning Evaluation Methods](#3-machine-learning-evaluation-methods)
  - [3.1 Accuracy](#31-accuracy)
  - [3.2 Precision](#32-precision)
  - [3.3 Recall](#33-recall)
  - [3.4 F1 Score (H-mean)](#34-f1-score-h-mean)
  - [3.4 ROC Curve](#34-roc-curve)
  - [3.5 Cosine Distance vs. Euclidean Distance](#35-cosine-distance-vs-euclidean-distance)
  - [3.6 A/B Testing](#36-ab-testing)
  - [3.7 Model Evaluation Methods](#37-model-evaluation-methods)
  - [3.8 Hyperparameter Tuning](#38-hyperparameter-tuning)
  - [3.9 Overfitting and Underfitting](#39-overfitting-and-underfitting)
- [4. Statistical Tests](#4-statistical-tests)
  - [4.1 KS Test](#41-ks-test)
  - [4.2 T Test](#42-t-test)
  - [4.3 F Test](#43-f-test)
  - [4.4 Grubbs Test](#44-grubbs-test)
  - [4.5 Chi-Square Test](#45-chi-square-test)
- [5. References](#5-references)

## 1. What is Feature Engineering?

Feature engineering is the process of transforming raw data into features for algorithms and models. Essentially, it is about representing and presenting data. In practice, **feature engineering aims to remove noise and redundancy** from raw data and design effective features that capture the relationship between the problem and the prediction model.

Two main data types:

1. **Structured data**: Like a relational table; each column has a clear definition (numeric, categorical); each row is a sample.
2. **Unstructured data**: Text, images, audio, video—information not easily represented by simple numbers; no clear category; variable size per sample.

### 1.1 Feature Normalization

To eliminate scale differences across features, we normalize so different metrics are comparable. E.g., for height (1.6–1.8 m) and weight (50–100 kg), raw analysis would overweight weight. Normalization brings all features to a similar scale.

Two common methods:

1. **Min-Max Scaling**: Linear transform mapping to [0, 1].
   ![](https://latex.codecogs.com/gif.latex?X_{norm}=\frac{X-X_{min}}{X_{max}-X_{min}})

2. **Z-Score Normalization**: Maps to mean 0, std 1.
   ![](https://latex.codecogs.com/gif.latex?z=\frac{x-u}{\sigma})

**Advantage**: Normalized data often allows gradient descent to converge faster.

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5cdl44ubjj30gz08i40j.jpg)

Normalization is not universal. Models solved by gradient descent (linear regression, logistic regression, SVM, neural networks) usually benefit. Decision trees typically do not.

### 1.2 Categorical Features

Categorical features (e.g., gender, blood type) take values from a finite set. Input is often string. Aside from tree models, most algorithms (LR, SVM) require converting them to numeric form.

1. **Ordinal encoding**: For ordered categories (e.g., low/mid/high → 1,2,3). Preserves order.
2. **One-hot encoding**: For unordered categories. E.g., blood type A/B/AB/O → (1,0,0,0), (0,1,0,0), etc. Use when the number of categories is moderate.
3. **Binary encoding**: First assign ordinal IDs, then encode IDs in binary.

### 1.3 High-Dimensional Combination Features

To capture complex relationships, first-order discrete features are often combined into higher-order features. E.g., in ad CTR prediction, language and type can form second-order features.

![](http://wx3.sinaimg.cn/mw690/00630Defly1g5cdvbua1aj30n30kf752.jpg)

### 1.4 Text Representation Models

1. **Bag-of-words and N-gram**: Treat documents as bags of words (order ignored). Each document → vector; each dimension = word; weight often from TF-IDF.
2. **Topic models**: Discover topics and topic-word distributions; compute topic distribution per document.
3. **Word embeddings**: Map words to low-dimensional dense vectors (e.g., 50–300 dim). Each dimension can be seen as a latent topic.

### 1.5 Other Feature Engineering

1. **Missing values**: If few, impute (e.g., mean); if many, consider dropping the feature.
2. **Feature selection**: Remove features with low correlation to the target.

### 1.6 Feature Engineering Mind Map

![](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1512980743_407.png)

## 2. Machine Learning Optimization Methods

Optimization is central to ML. ML algorithm = model representation + model evaluation + optimization. The optimizer searches the representation space for the best model under the evaluation metric.

### 2.1 Common Loss Functions

The loss function L(Y, f(x)) measures how much predictions deviate from true values. Smaller loss → better robustness.

1. **Squared loss**: Sum of squared residuals. MSE = (1/n) Σ(y'−y)². Used in linear regression.
2. **Log loss**: Used in logistic regression. Penalizes wrong probabilities; log(0)=∞ gives maximum penalty.
3. **Hinge loss**: Used in SVM for max-margin classification. See [SVM 1.2.3](../4.%20SVM/4.%20SVM.md#13-deep-dive-into-svm-level-2)

### 2.2 What is Convex Optimization

A function L(·) is **convex** iff for any x, y and λ∈[0,1]:
L(λx + (1−λ)y) ≤ λL(x) + (1−λ)L(y)

Interpretation: the line segment between any two points on the surface lies above the surface. Convex optimization: SVM, linear regression. Non-convex: low-rank models, deep neural networks.

### 2.3 Regularization Terms

Add a parameter penalty to the loss: L1, L2, ElasticNet. Benefits: control parameter magnitude, limit search space, address over/underfitting. See [Linear Regression §5](../Liner%20Regression/1.Liner%20Regression.md#5-how-to-address-overfitting-and-underfitting).

### 2.4 Common Optimization Methods

1. **Gradient descent**: Use negative gradient as search direction ("steepest descent"). Simple; global optimum for convex objectives. Drawbacks: slow near minima; zigzagging.
2. **Newton's method**: Uses second-order Taylor expansion; faster convergence. Requires Hessian inverse; expensive for high dimensions; sensitive to saddle points.
3. **Quasi-Newton**: Approximates Hessian inverse with a positive-definite matrix; avoids explicit second derivatives. Often more practical than Newton.
4. **Conjugate gradient**: Between gradient descent and Newton; uses first derivatives; avoids Hessian storage. Good for large problems.

### 2.5 Dimensionality Reduction

#### 2.5.1 LDA (Linear Discriminant Analysis)

Supervised; uses class labels. Project data onto a line (or low-dim subspace) so **within-class variance is minimized and between-class variance is maximized**. Limited to k−1 dimensions for k classes.

**Pros**: Uses class prior; goal is clearer than PCA.
**Cons**: Assumes Gaussian; limited to k−1 dims; can overfit.

#### 2.5.2 PCA (Principal Component Analysis)

Unsupervised. Project high-dim data onto lower dimensions. Key ideas: remove redundancy (linear dependencies), remove noise (small eigenvalues), keep directions with large variance. Uses covariance matrix; diagonalize to find principal components.

**Pros**: Simple; based on variance; orthogonal components.
**Cons**: Less interpretable; may drop useful information in small components.

#### 2.5.3 LDA vs. PCA

**Why reduce dimensions?** Multicollinearity, high-dim sparsity, redundancy, latent structure.

**Goals**: Fewer predictors; independence; interpretability; easier handling; denoising; lower compute.

| | LDA | PCA |
|---|-----|-----|
| Same | Both reduce dims; use matrix decomposition; assume Gaussian |
| | Supervised | Unsupervised |
| | Max k−1 dims | No dim limit |
| | Classification + dim reduction | Dim reduction only |
| | Best classification direction | Max variance direction |

## 3. Machine Learning Evaluation Methods

Confusion matrix (conventional layout; rows = actual, columns = predicted):

| | Positive (pred) | Negative (pred) |
|---|-----------------|-----------------|
| Positive (actual) | TP | FN |
| Negative (actual) | FP | TN |

### 3.1 Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN). Simple but misleading when classes are imbalanced.

### 3.2 Precision

Precision = TP / (TP + FP). Among predicted positives, how many are correct.

### 3.3 Recall

Recall = TP / (TP + FN). Among actual positives, how many were retrieved.

**P-R curve**: Plot precision vs. recall as the threshold varies. Evaluate over the full curve, not a single point.

### 3.4 F1 Score (H-mean)

F1 = 2PR / (P+R) = 2TP / (2TP + FP + FN). Harmonic mean of precision and recall.

### 3.4 ROC Curve

ROC plots TPR (sensitivity) vs. FPR (1−specificity). Ideal: (0, 1). **Closer to (0,1), better.**

**AUC**: Area under ROC. 0.5–1. Larger AUC → better classifier. AUC=1: perfect; 0.5: random; <0.5: worse than random (but invert predictions to fix).

### 3.5 Cosine vs. Euclidean Distance

**Cosine**: cos(A,B) = A·B / (||A|| ||B||). Focuses on angle; [−1, 1]. Good when magnitudes differ but direction is similar (e.g., text similarity). Stable in high dimensions.

**Euclidean**: Straight-line distance. Affected by scale and dimensionality.

### 3.6 A/B Testing

Create two (or more) versions of a UI/flow; randomly assign users; collect metrics; choose the best version.

### 3.7 Model Evaluation Methods

1. **Holdout**: Split into train/validation (e.g., 70/30). Simple but sensitive to split.
2. **K-fold cross-validation**: Split into k folds; each fold serves as validation once; average metrics. Often k=10.
3. **Bootstrap**: Sample n with replacement to form train set; out-of-bag samples as validation. Useful when data is limited.

### 3.8 Hyperparameter Tuning

- **Grid search**: Exhaustive over a grid. Can find global optimum but costly.
- **Random search**: Sample from the search space. Often faster; no guarantee.
- **Bayesian optimization**: Uses prior evaluations to model the objective and guide search.

### 3.9 Overfitting and Underfitting

**Overfitting**: Good on train, poor on test. **Mitigate**: More data; reduce complexity; regularization; ensemble methods.
**Underfitting**: Poor on both. **Mitigate**: Add features; increase complexity; reduce regularization.

## 4. Statistical Tests

### 4.1 KS Test

Kolmogorov-Smirnov: Compares empirical CDF to a theoretical distribution or two empirical distributions. One-sample: fit to a known distribution. Two-sample: compare two samples.

### 4.2 T Test

Student's t-test: For small samples, unknown population variance, normal distribution. Compares means.

### 4.3 F Test

Variance ratio test; tests homogeneity of variance. Used with ANOVA.

### 4.4 Grubbs Test

Detects outliers. Identifies and can remove "suspicious" points that deviate far from the mean.

### 4.5 Chi-Square Test

Measures deviation between observed and expected frequencies. Larger χ² → worse fit. Steps: state H0; partition range into k bins; compute observed frequencies; compute expected under H0; form χ² statistic; compare to χ² distribution with k−1 df.

**KS vs. Chi-square**: Both use observed vs. expected. Chi-square for categorical data; KS for continuous. Chi-square needs binning; KS uses raw data.

## 5. References

[100 Questions of Machine Learning](https://www.lanzous.com/i56i24f)

> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub: [https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! Group ID: 【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
