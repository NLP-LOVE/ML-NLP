## 目录
- [1. 什么是词嵌入(Word Embedding)](#1-什么是词嵌入word-embedding)
- [2.离散表示](#2离散表示)
  - [2.1 One-hot表示](#21-one-hot表示)
  - [2.2 词袋模型](#22-词袋模型)
  - [2.3 TF-IDF](#23-tf-idf)
  - [2.4 n-gram模型](#24-n-gram模型)
  - [2.5 离散表示存在的问题](#25-离散表示存在的问题)
- [3. 分布式表示](#3-分布式表示)
  - [3.1 共现矩阵](#31-共现矩阵)
- [4.神经网络表示](#4神经网络表示)
  - [4.1 NNLM](#41-nnlm)
  - [4.2 Word2Vec](#42-word2vec)
  - [4.3 sense2vec](#43-sense2vec)
- [5. 词嵌入为何不采用one-hot向量](#5-词嵌入为何不采用one-hot向量)
- [6. Word2Vec代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.1%20Word%20Embedding/word2vec.ipynb)

## 1. 什么是词嵌入(Word Embedding)

⾃然语⾔是⼀套⽤来表达含义的复杂系统。在这套系统中，词是表义的基本单元。顾名思义，词向量是⽤来表⽰词的向量，也可被认为是词的特征向量或表征。**把词映射为实数域向量的技术也叫词嵌⼊（word embedding）。**近年来，词嵌⼊已逐渐成为⾃然语⾔处理的基础知识。

在NLP(自然语言处理)领域，文本表示是第一步，也是很重要的一步，通俗来说就是把人类的语言符号转化为机器能够进行计算的数字，因为普通的文本语言机器是看不懂的，必须通过转化来表征对应文本。早期是**基于规则**的方法进行转化，而现代的方法是**基于统计机器学习**的方法。

**数据决定了机器学习的上限,而算法只是尽可能逼近这个上限，**在本文中数据指的就是文本表示，所以，弄懂文本表示的发展历程，对于NLP学习者来说是必不可少的。接下来开始我们的发展历程。文本表示分为**离散表示**和**分布式表示**：



## 2.离散表示

### 2.1 One-hot表示

One-hot简称读热向量编码，也是特征工程中最常用的方法。其步骤如下：

1. 构造文本分词后的字典，每个分词是一个比特值，比特值为0或者1。
2. 每个分词的文本表示为该分词的比特位为1，其余位为0的矩阵表示。

例如：**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10}**

每个词典索引对应着比特位。那么利用One-hot表示为：

**John: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]**

**likes: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]** .......等等，以此类推。

One-hot表示文本信息的**缺点**：

- 随着语料库的增加，数据特征的维度会越来越大，产生一个维度很高，又很稀疏的矩阵。
- 这种表示方法的分词顺序和在句子中的顺序是无关的，不能保留词与词之间的关系信息。



### 2.2 词袋模型

词袋模型(Bag-of-words model)，像是句子或是文件这样的文字可以用一个袋子装着这些词的方式表现，这种表现方式不考虑文法以及词的顺序。

**文档的向量表示可以直接将各词的词向量表示加和**。例如：

**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10}**

那么第一句的向量表示为：**[1,2,1,1,1,0,0,0,1,1]**，其中的2表示**likes**在该句中出现了2次，依次类推。

词袋模型同样有一下**缺点**：

- 词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大。
- 词与词之间是没有顺序关系的。



### 2.3 TF-IDF

TF-IDF（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。

**字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。**

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-21_10-7-23.png)

分母之所以加1，是为了避免分母为0。

那么，![](https://latex.codecogs.com/gif.latex?TF-IDF=TF*IDF)，从这个公式可以看出，当w在文档中出现的次数增大时，而TF-IDF的值是减小的，所以也就体现了以上所说的了。

**缺点：** 还是没有把词与词之间的关系顺序表达出来。



### 2.4 n-gram模型

n-gram模型为了保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序。

例如：

**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}**

那么第一句的向量表示为：**[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]**，其中第一个1表示**John likes**在该句中出现了1次，依次类推。

**缺点：** 随着n的大小增加，词表会成指数型膨胀，会越来越大。



### 2.5 离散表示存在的问题

由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

- 无法衡量词向量之间的关系。
- 词表的维度随着语料库的增长而膨胀。
- n-gram词序列随语料库增长呈指数型膨胀，更加快。
- 离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与我们生活中理解的信息是不一样的。



## 3. 分布式表示

科学家们为了提高模型的精度，又发明出了分布式的表示文本信息的方法，这就是这一节需要介绍的。

**用一个词附近的其它词来表示该词，这是现代统计自然语言处理中最有创见的想法之一。** 当初科学家发明这种方法是基于人的语言表达，认为一个词是由这个词的周边词汇一起来构成精确的语义信息。就好比，物以类聚人以群分，如果你想了解一个人，可以通过他周围的人进行了解，因为周围人都有一些共同点才能聚集起来。



### 3.1 共现矩阵

共现矩阵顾名思义就是共同出现的意思，词文档的共现矩阵主要用于发现主题(topic)，用于主题模型，如LSA。

局域窗中的word-word共现矩阵可以挖掘语法和语义信息，**例如：**

- I like deep learning.	
- I like NLP.	
- I enjoy flying

有以上三句话，设置滑窗为2，可以得到一个词典：**{"I like","like deep","deep learning","like NLP","I enjoy","enjoy flying","I like"}**。

我们可以得到一个共现矩阵(对称矩阵)：

![image](https://wx2.sinaimg.cn/large/00630Defly1g2rwv1op5zj30q70c7wh2.jpg)

中间的每个格子表示的是行和列组成的词组在词典中共同出现的次数，也就体现了**共现**的特性。

**存在的问题：**

- 向量维数随着词典大小线性增长。
- 存储整个词典的空间消耗非常大。
- 一些模型如文本分类模型会面临稀疏性问题。
- **模型会欠稳定，每新增一份语料进来，稳定性就会变化。**



## 4.神经网络表示

### 4.1 NNLM

NNLM (Neural Network Language model)，神经网络语言模型是03年提出来的，通过训练得到中间产物--词向量矩阵，这就是我们要得到的文本表示向量矩阵。

NNLM说的是定义一个前向窗口大小，其实和上面提到的窗口是一个意思。把这个窗口中最后一个词当做y，把之前的词当做输入x，通俗来说就是预测这个窗口中最后一个词出现概率的模型。

![image](https://wx3.sinaimg.cn/large/00630Defly1g2vb5thw9rj30eq065dg4.jpg)

以下是NNLM的网络结构图：

![image](https://wx3.sinaimg.cn/large/00630Defly1g2t1f4bqilj30lv0e2adl.jpg)

- input层是一个前向词的输入，是经过one-hot编码的词向量表示形式，具有V*1的矩阵。

- C矩阵是投影矩阵，也就是稠密词向量表示，在神经网络中是**w参数矩阵**，该矩阵的大小为D*V，正好与input层进行全连接(相乘)得到D\*1的矩阵，采用线性映射将one-hot表示投影到稠密D维表示。

  ![image](https://wx3.sinaimg.cn/large/00630Defly1g2t1s20jpnj30f107575i.jpg)

- output层(softmax)自然是前向窗中需要预测的词。

- 通过BP＋SGD得到最优的C投影矩阵，这就是NNLM的中间产物，也是我们所求的文本表示矩阵，**通过NNLM将稀疏矩阵投影到稠密向量矩阵中。**

### 4.2 Word2Vec

谷歌2013年提出的Word2Vec是目前最常用的词嵌入模型之一。Word2Vec实际是一种浅层的神经网络模型，它有两种网络结构，**分别是CBOW（Continues Bag of Words）连续词袋和Skip-gram。**Word2Vec和上面的NNLM很类似，但比NNLM简单。

**CBOW**

CBOW获得中间词两边的的上下文，然后用周围的词去预测中间的词，把中间词当做y，把窗口中的其它词当做x输入，x输入是经过one-hot编码过的，然后通过一个隐层进行求和操作，最后通过激活函数softmax，可以计算出每个单词的生成概率，接下来的任务就是训练神经网络的权重，使得语料库中所有单词的整体生成概率最大化，而求得的权重矩阵就是文本表示词向量的结果。

![image](https://ws2.sinaimg.cn/large/00630Defly1g2u6va5fvyj30gf0h0aby.jpg)

**Skip-gram**：

Skip-gram是通过当前词来预测窗口中上下文词出现的概率模型，把当前词当做x，把窗口中其它词当做y，依然是通过一个隐层接一个Softmax激活函数来预测其它词的概率。如下图所示：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-20_20-34-0.jpg)

**优化方法**：

- **层次Softmax：**至此还没有结束，因为如果单单只是接一个softmax激活函数，计算量还是很大的，有多少词就会有多少维的权重矩阵，所以这里就提出**层次Softmax(Hierarchical Softmax)**，使用Huffman Tree来编码输出层的词典，相当于平铺到各个叶子节点上，**瞬间把维度降低到了树的深度**，可以看如下图所示。这课Tree把出现频率高的词放到靠近根节点的叶子节点处，每一次只要做二分类计算，计算路径上所有非叶子节点词向量的贡献即可。

> **哈夫曼树(Huffman Tree)**：给定N个权值作为N个[叶子结点](https://baike.baidu.com/item/叶子结点/3620239)，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。

![image](https://ws4.sinaimg.cn/large/00630Defly1g2u762c7nwj30jb0fs0wh.jpg)

- **负例采样(Negative Sampling)：** 这种优化方式做的事情是，在正确单词以外的负样本中进行采样，最终目的是为了减少负样本的数量，达到减少计算量效果。将词典中的每一个词对应一条线段，所有词组成了[0，1］间的剖分，如下图所示，然后每次随机生成一个[1, M-1]间的整数，看落在哪个词对应的剖分上就选择哪个词，最后会得到一个负样本集合。

  ![image](https://wx3.sinaimg.cn/large/00630Defly1g2u7vvrgjnj30lu07d75v.jpg)

**Word2Vec存在的问题**

- 对每个local context window单独训练，没有利用包 含在global co-currence矩阵中的统计信息。
- 对多义词无法很好的表示和处理，因为使用了唯一的词向量



### 4.3 sense2vec

word2vec模型的问题在于词语的多义性。比如duck这个单词常见的含义有水禽或者下蹲，但对于 word2vec 模型来说，它倾向于将所有概念做归一化平滑处理，得到一个最终的表现形式。



## 5. 词嵌入为何不采用one-hot向量

虽然one-hot词向量构造起来很容易，但通常并不是⼀个好选择。⼀个主要的原因是，one-hot词向量⽆法准确表达不同词之间的相似度，如我们常常使⽤的余弦相似度。由于任何两个不同词的one-hot向量的余弦相似度都为0，多个不同词之间的相似度难以通过onehot向量准确地体现出来。

word2vec⼯具的提出正是为了解决上⾯这个问题。它将每个词表⽰成⼀个定⻓的向量，并使得这些向量能较好地表达不同词之间的相似和类⽐关系。



## 6. Word2Vec代码实现

**数据下载**

中文维基百科的打包文件地址为链接: https://pan.baidu.com/s/1H-wuIve0d_fvczvy3EOKMQ 提取码: uqua 

百度网盘加速下载地址：https://www.baiduwp.com/?m=index

[Word2Vec训练维基百科文章代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.1%20Word%20Embedding/word2vec.ipynb)

------

> 作者:[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub:[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号:【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
