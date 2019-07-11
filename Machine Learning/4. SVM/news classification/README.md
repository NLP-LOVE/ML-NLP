## 目录
- [1.新闻分类案例](#1新闻分类案例)
  - [1.1介绍](#11介绍)
  - [1.2数据集下载](#12数据集下载)
  - [1.3libsvm库安装](#13libsvm库安装)
  - [1.4实现步骤](#14实现步骤)
  - [1.5代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

## 1.新闻分类案例

### 1.1介绍

这是一个预测新闻分类的案例，通过给定的数据集来预测测试集的新闻分类，该案例用到的是libsvm库，实现步骤我已经写到代码里面了，每一步都有注释，相信很多人都能够看得明白。

### 1.2数据集下载

因为数据集比较大，不适合放到github里，所以单独下载吧，放到与代码同级目录即可。

有三个文件，一个是训练数据，一个是测试数据，一个是分类。

训练数据：https://pan.baidu.com/s/1ZkxGIvvGml3vig-9_s1pRw

百度网盘加速下载地址：https://www.baiduwp.com/?m=index

### 1.3libsvm库安装

LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM[模式识别](https://baike.baidu.com/item/%E6%A8%A1%E5%BC%8F%E8%AF%86%E5%88%AB/295301)与回归的软件包。其它的svm库也有，这里以libsvm为例。

libsvm下载地址：[libsvm-3.23.zip](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip)

#### MAC系统

1.下载libsvm后解压，进入目录有个文件：**libsvm.so.2**，把这个文件复制到python安装目录**site-packages/**下。

2.在**site-packages/**下新建libsvm文件夹，并进入**libsvm**目录新建init.py的空文件。

3.进入libsvm解压路径：**libsvm-3.23/python/**，把里面的三个文件：**svm.py、svmutil.py、commonutil.py**，复制到新建的：**site-packages/libsvm/**目录下。之后就可以使用libsvm了。

#### Windows系统

安装教程：https://www.cnblogs.com/bbn0111/p/8318629.html
### 1.4实现步骤

1.先对数据集进行分词，本案例用的是**jieba**分词。

2.对分词的结果进行词频统计，分配词ID。

3.根据词ID生成词向量，这就是最终的训练数据。

4.调用libsvm训练器进行训练。

### 1.5代码实现
GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
