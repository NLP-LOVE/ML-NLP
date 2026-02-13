## Table of Contents
- [1. News Classification Case](#1-news-classification-case)
  - [1.1 Introduction](#11-introduction)
  - [1.2 Dataset Download](#12-dataset-download)
  - [1.3 libsvm Installation](#13-libsvm-installation)
  - [1.4 Implementation Steps](#14-implementation-steps)
  - [1.5 Code Implementation](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

## 1. News Classification Case

### 1.1 Introduction

This case predicts news categories from a given dataset. It uses the libsvm library. Implementation steps and comments are in the notebook.

### 1.2 Dataset Download

The dataset is too large for GitHub. Download separately and place in the same directory as the code.

- Training data: https://pan.baidu.com/s/1ZkxGIvvGml3vig-9_s1pRw
- Baidu Netdisk accelerator: https://www.baiduwp.com/?m=index

### 1.3 libsvm Installation

LIBSVM is an SVM package developed by Lin Chih-Jen et al. at National Taiwan University. Download: [libsvm-3.23.zip](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip)

**macOS:**
1. Extract libsvm; copy `libsvm.so.2` to Python `site-packages/`
2. Create `libsvm` folder in `site-packages/` with empty `__init__.py`
3. Copy `svm.py`, `svmutil.py`, `commonutil.py` from `libsvm/python/` to `site-packages/libsvm/`

**Windows:** See https://www.cnblogs.com/bbn0111/p/8318629.html

### 1.4 Implementation Steps

1. Tokenize with **jieba**
2. Build vocabulary and assign word IDs
3. Generate word vectors as training data
4. Train with libsvm

### 1.5 Code Implementation

[Click to open](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

------

> Author: [@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> Welcome to join the discussion! <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
