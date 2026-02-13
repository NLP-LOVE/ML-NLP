## Table of Contents
- [1. What is a Decision Tree](#1-what-is-a-decision-tree)
  - [1.1 Basic Idea of Decision Trees](#11-basic-idea-of-decision-trees)
  - [1.2 The "Tree" Growing Process](#12-the-tree-growing-process)
  - [1.3 How Does the "Tree" Grow](#13-how-does-the-tree-grow)
  - [1.3.1 ID3 Algorithm](#131-id3-algorithm)
  - [1.3.2 C4.5](#132-c45)
  - [1.3.3 CART Algorithm](#133-cart-algorithm)
  - [1.3.4 Three Types of Decision Trees](#134-three-types-of-decision-trees)
- [2. Why Don't Tree Structures Need Normalization?](#2-why-dont-tree-structures-need-normalization)
- [3. Difference Between Classification and Regression Decision Trees](#3-difference-between-classification-and-regression-decision-trees)
- [4. How Does a Decision Tree Prune](#4-how-does-a-decision-tree-prune)
- [5. Code Implementation](DecisionTree.ipynb)

## 1. What is a Decision Tree

### 1.1 Basic Idea of Decision Trees

The following image helps illustrate the fundamental difference between the LR model and the decision tree model: a decision problem—whether to go on a blind date. A girl's mother is introducing potential suitors.

![image](https://wx2.sinaimg.cn/large/00630Defly1g4q286viibj30pk0pfk09.jpg)

The LR model feeds all features into learning at once, while the decision tree works like if-else in programming: conditional branching. That is the core difference.

### 1.2 The "Tree" Growing Process

Decision trees make decisions based on a "tree" structure. Two main questions arise:

- How does the "tree" grow?
- When does the "tree" stop growing?

Understanding these two questions is enough to build the model. The overall flow follows a "divide and conquer" idea: a recursive process from root to leaf, and at each internal node we seek a "splitting" attribute (i.e., a feature).

#### When Does the "Tree" Stop Growing

- All samples at the current node belong to the same class—no further split needed.
- The attribute set is empty, or all samples have the same value for all attributes—no split possible.
- The sample set at the current node is empty—no split possible.

### 1.3 How Does the "Tree" Grow

In life we often follow the majority: where to eat, which gadget to buy, where to travel. Similarly in decision trees, when most samples are of the same class, we have effectively made a decision.

We formalize this as **purity**: higher majority implies higher purity. **Lower entropy implies higher purity.** Entropy measures "information content." If samples are all alike, information is low; if they differ, information is high.

The entropy formula:

![](https://latex.codecogs.com/gif.latex?Ent(D)=-\sum_{k=1}^{|y|}p_klog_2p_k)

Pk denotes: the proportion of class k in the current sample set D.

**Information Gain**

Formula:

![image](https://wx3.sinaimg.cn/large/00630Defly1g4q5h6oby7j30he08tdh5.jpg)

Simply put: entropy before the split minus entropy after the split. It is the "step" toward higher purity.

#### 1.3.1 ID3 Algorithm

At the root, compute entropy. Then split by each attribute and compute entropy at child nodes. Information gain = root entropy − attribute node entropy. Sort by information gain; the top attribute is the first split. Repeat. This gives the decision tree structure.

Note: Information gain favors attributes with many possible values (e.g., "ID"). To address this, C4.5 was introduced.

#### 1.3.2 C4.5

To fix the information gain bias, use **information gain ratio**:

![](https://latex.codecogs.com/gif.latex?Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)})

where:

![](https://latex.codecogs.com/gif.latex?IV(a)=-\sum_{v=1}^{V}\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|})

The more possible values attribute a has (larger V), the larger IV(a) typically is. **Information gain ratio essentially multiplies a penalty factor by information gain. When features have many values, the penalty is small; when few, the penalty is large.**

**Drawback**: Information gain ratio favors features with fewer values.

**Usage**: Do not simply pick the feature with maximum gain ratio. First select features whose information gain is above average, then among those, choose the one with the highest information gain ratio.

#### 1.3.3 CART Algorithm

CART uses the **Gini index** instead of entropy:

![image](https://wx1.sinaimg.cn/large/00630Defly1g4q5dmvyykj30eb01edfs.jpg)

It represents the probability that a randomly chosen sample is misclassified. **Smaller Gini(D) means higher purity of D.**

##### Example

Suppose we have feature "education" with values "undergraduate," "master," "doctor." When splitting D by this feature, we get three possible splits:

1. Split: "undergraduate" → {undergraduate}, {master, doctor}
2. Split: "master" → {master}, {undergraduate, doctor}
3. Split: "doctor" → {doctor}, {undergraduate, master}

For each split we compute the Gini based on splitting D into two subsets:

![](https://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2))

**For a feature with multiple values (>2), compute Gini(D,Ai) for each possible split, then choose the split with the smallest Gini as the best split for that feature.** This yields the full tree.

#### 1.3.4 Three Types of Decision Trees

- **ID3**: Attributes with more values tend to yield purer splits and larger information gain. Often produces a broad, shallow tree—not ideal.
- **C4.5**: Uses information gain ratio instead of information gain.
- **CART**: Uses Gini index instead of entropy; minimizes impurity instead of maximizing information gain.

## 2. Why Don't Tree Structures Need Normalization?

Numerical scaling does not change split points; the structure of the tree remains the same. Splits are based on sorted feature values; the order does not change, so branches and split points stay the same. Tree models do not use gradient descent; they find optimal points by searching for the best split, so the objective is stepwise and not differentiable. Hence normalization is unnecessary.

Why do non-tree models (e.g., AdaBoost, SVM, LR, KNN, KMeans) need normalization?

For linear models, when feature scales differ greatly, gradient descent sees elliptical loss contours and requires many iterations to converge. With normalization, contours become more circular, so SGD converges faster toward the optimum.

## 3. Difference Between Classification and Regression Decision Trees

Classification And Regression Tree (CART) can build either a classification tree or a regression tree. The building process differs slightly.

**Regression tree**:

CART regression trees assume binary splits. At each node, we split on feature j at threshold s: samples with value < s go left, ≥ s go right.

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343854853617715.png)

CART regression trees partition the feature space. Finding the optimal partition is NP-hard, so heuristic methods are used. A typical objective is:

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438551488112806.png)

We search over all features and split points to find the best j and s:

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343855213970444.png)

Reference: [CART Classification Tree, Regression Tree, and Model Tree](https://blog.csdn.net/jiede1/article/details/76034328)

## 4. How Does a Decision Tree Prune

Two main strategies: **pre-pruning** and **post-pruning**.

- **Pre-pruning**: Before each split, use a validation set to check whether splitting improves accuracy. If not, mark the node as a leaf and stop. If yes, continue recursively.
- **Post-pruning**: First build a full tree on the training set, then traverse non-leaf nodes bottom-up. If replacing a subtree with a leaf improves generalization, do the replacement.

Reference: [Decision Tree Generation and Pruning](https://blog.csdn.net/am290333566/article/details/81187562)

## 5. Code Implementation

GitHub: [https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/DecisionTree.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/DecisionTree.ipynb)

------



> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub: [https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! Let's improve this project together! Group ID: 【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
