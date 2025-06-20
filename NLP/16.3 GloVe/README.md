## 目录
- [1. 说说GloVe](#1-说说glove)
- [2. GloVe的实现步骤](#2-glove的实现步骤)
  - [2.1 构建共现矩阵](#21-构建共现矩阵)
  - [2.2 词向量和共现矩阵的近似关系](#22-词向量和共现矩阵的近似关系)
  - [2.3 构造损失函数](#23-构造损失函数)
  - [2.4 训练GloVe模型](#24-训练glove模型)
- [3. GloVe与LSA、Word2Vec的比较](#3-glove与lsaword2vec的比较)
- [4. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.3%20GloVe/GloVe.ipynb)
- [5. 参考文献](#5-参考文献)

## 1. 说说GloVe

正如GloVe论文的标题而言，**GloVe的全称叫Global Vectors for Word Representation，它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。** 我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。



## 2. GloVe的实现步骤

### 2.1 构建共现矩阵 

**什么是共现矩阵？**

共现矩阵顾名思义就是共同出现的意思，词文档的共现矩阵主要用于发现主题(topic)，用于主题模型，如LSA。

局域窗中的word-word共现矩阵可以挖掘语法和语义信息，**例如：**

- I like deep learning.	
- I like NLP.	
- I enjoy flying

有以上三句话，设置滑窗为2，可以得到一个词典：**{"I like","like deep","deep learning","like NLP","I enjoy","enjoy flying","I like"}**。

我们可以得到一个共现矩阵(对称矩阵)：

![image](https://wx2.sinaimg.cn/large/00630Defly1g2rwv1op5zj30q70c7wh2.jpg)

中间的每个格子表示的是行和列组成的词组在词典中共同出现的次数，也就体现了**共现**的特性。

**GloVe的共现矩阵**

根据语料库（corpus）构建一个共现矩阵（Co-ocurrence Matrix）X，**矩阵中的每一个元素 Xij 代表单词 i 和上下文单词 j 在特定大小的上下文窗口（context window）内共同出现的次数。**一般而言，这个次数的最小单位是1，但是GloVe不这么认为：它根据两个单词在上下文窗口的距离 d，提出了一个衰减函数（decreasing weighting）：decay=1/d 用于计算权重，也就是说**距离越远的两个单词所占总计数（total count）的权重越小**。



### 2.2 词向量和共现矩阵的近似关系

构建词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的近似关系，论文的作者提出以下的公式可以近似地表达两者之间的关系：

![](https://latex.codecogs.com/gif.latex?w_i^T\tilde{w_j}+b_i+\tilde{b}_j=log(X_{ij}))

其中，![](https://latex.codecogs.com/gif.latex?w_i^T和\tilde{w}_j)是我们最终要求解的词向量；![](https://latex.codecogs.com/gif.latex?b_i和\tilde{b}_j)分别是两个词向量的bias term。当然你对这个公式一定有非常多的疑问，比如它到底是怎么来的，为什么要使用这个公式，为什么要构造两个词向量 ![](https://latex.codecogs.com/gif.latex?w_i^T和\tilde{w}_j)？请参考文末的参考文献。



### 2.3 构造损失函数

有了2.2的公式之后我们就可以构造它的loss function了：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-24_10-11-53.png)

这个loss function的基本形式就是最简单的mean square loss，只不过在此基础上加了一个权重函数![](https://latex.codecogs.com/gif.latex?f(X_{ij}))，那么这个函数起了什么作用，为什么要添加这个函数呢？我们知道在一个语料库中，肯定存在很多单词他们在一起出现的次数是很多的（frequent co-occurrences），那么我们希望：

- 这些单词的权重要大于那些很少在一起出现的单词（rare co-occurrences），所以这个函数要是非递减函数（non-decreasing）；
- 但我们也不希望这个权重过大（overweighted），当到达一定程度之后应该不再增加；
- 如果两个单词没有在一起出现，也就是![](https://latex.codecogs.com/gif.latex?X_{ij}=0)，那么他们应该不参与到 loss function 的计算当中去，也就是f(x) 要满足 f(0)=0。

满足以上三个条件的函数有很多，论文作者采用了如下形式的分段函数：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-23_21-52-27.png)

这个函数图像如下所示：

![](http://www.fanyeong.com/wp-content/uploads/2019/08/zE6t1ig.jpg)



### 2.4 训练GloVe模型

虽然很多人声称GloVe是一种无监督（unsupervised learing）的学习方式（因为它确实不需要人工标注label），但其实它还是有label的，这个label就是以上公式中的 log(Xij)，而公式中的向量 $w和\tilde{w}$ 就是要不断更新/学习的参数，所以本质上它的训练方式跟监督学习的训练方法没什么不一样，都是基于梯度下降的。

具体地，这篇论文里的实验是这么做的：**采用了AdaGrad的梯度下降算法，对矩阵 X 中的所有非零元素进行随机采样，学习曲率（learning rate）设为0.05，在vector size小于300的情况下迭代了50次，其他大小的vectors上迭代了100次，直至收敛。** 最终学习得到的是两个vector是 $w和\tilde{w}$，因为 X 是对称的（symmetric），所以从原理上讲 $w和\tilde{w}$ 是也是对称的，他们唯一的区别是初始化的值不一样，而导致最终的值不一样。

所以这两者其实是等价的，都可以当成最终的结果来使用。**但是为了提高鲁棒性，我们最终会选择两者之和**  ![](https://latex.codecogs.com/gif.latex?w+\tilde{w})**作为最终的vector（两者的初始化不同相当于加了不同的随机噪声，所以能提高鲁棒性）。** 在训练了400亿个token组成的语料后，得到的实验结果如下图所示：

![](http://www.fanyeong.com/wp-content/uploads/2019/08/X6eVUJJ.jpg)

这个图一共采用了三个指标：语义准确度，语法准确度以及总体准确度。那么我们不难发现Vector Dimension在300时能达到最佳，而context Windows size大致在6到10之间。



## 3. GloVe与LSA、Word2Vec的比较

LSA（Latent Semantic Analysis）是一种比较早的count-based的词向量表征工具，它也是基于co-occurance matrix的，只不过采用了基于奇异值分解（SVD）的矩阵分解技术对大矩阵进行降维，而我们知道SVD的复杂度是很高的，所以它的计算代价比较大。还有一点是它对所有单词的统计权重都是一致的。而这些缺点在GloVe中被一一克服了。

而word2vec最大的缺点则是没有充分利用所有的语料，所以GloVe其实是把两者的优点结合了起来。从这篇论文给出的实验结果来看，GloVe的性能是远超LSA和word2vec的，但网上也有人说GloVe和word2vec实际表现其实差不多。



## 4. 代码实现

**生成词向量**

下载GitHub项目：[https://github.com/stanfordnlp/GloVe/archive/master.zip](https://github.com/stanfordnlp/GloVe/archive/master.zip)

解压后，进入目录执行

make

进行编译操作。

然后执行 sh demo.sh 进行训练并生成词向量文件：vectors.txt和vectors.bin

[GloVe代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.3%20GloVe/GloVe.ipynb)



## 5. 参考文献

- [GloVe详解](https://www.fanyeong.com/2018/02/19/glove-in-detail/)
- [NLP从词袋到Word2Vec的文本表示](https://blog.csdn.net/weixin_41510260/article/details/90046989)



------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
