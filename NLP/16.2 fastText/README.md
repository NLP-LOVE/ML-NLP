## 目录
- [1. 什么是fastText](#1-什么是fasttext)
- [2. n-gram表示单词](#2-n-gram表示单词)
- [3. fastText模型架构](#3-fasttext模型架构)
- [4. fastText核心思想](#4-fasttext核心思想)
- [5. 输出分类的效果](#5-输出分类的效果)
- [6. fastText与Word2Vec的不同](#6-fasttext与word2vec的不同)
- [7. 代码实现](#7-代码实现)
- [8. 参考文献](#8-参考文献)

## 1. 什么是fastText

英语单词通常有其内部结构和形成⽅式。例如，我们可以从“dog”“dogs”和“dogcatcher”的字⾯上推测它们的关系。这些词都有同⼀个词根“dog”，但使⽤不同的后缀来改变词的含义。而且，这个关联可以推⼴⾄其他词汇。

在word2vec中，我们并没有直接利⽤构词学中的信息。⽆论是在跳字模型还是连续词袋模型中，我们都将形态不同的单词⽤不同的向量来表⽰。例如，**“dog”和“dogs”分别⽤两个不同的向量表⽰，而模型中并未直接表达这两个向量之间的关系。鉴于此，fastText提出了⼦词嵌⼊(subword embedding)的⽅法，从而试图将构词信息引⼊word2vec中的CBOW。**

这里有一点需要特别注意，一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物。除非你决定使用预训练的embedding来训练fastText分类模型，这另当别论。



## 2. n-gram表示单词

word2vec把语料库中的每个单词当成原子的，它会为每个单词生成一个向量。这忽略了单词内部的形态特征，比如：“book” 和“books”，“阿里巴巴”和“阿里”，这两个例子中，两个单词都有较多公共字符，即它们的内部形态类似，但是在传统的word2vec中，这种单词内部形态信息因为它们被转换成不同的id丢失了。

**为了克服这个问题，fastText使用了字符级别的n-grams来表示一个单词。**对于单词“book”，假设n的取值为3，则它的trigram有:

**“<bo”,  “boo”,  “ook”, “ok>”**

其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“book”这个单词，进一步，我们可以用这4个trigram的向量叠加来表示“apple”的词向量。

**这带来两点好处**：

1. 对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。
2. 对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。



## 3. fastText模型架构

之前提到过，fastText模型架构和word2vec的CBOW模型架构非常相似。下面是fastText模型架构图：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-21_20-31-22.jpeg)

**注意**：此架构图没有展示词向量的训练过程。可以看到，和CBOW一样，fastText模型也只有三层：输入层、隐含层、输出层（Hierarchical Softmax），输入都是多个经向量表示的单词，输出都是一个特定的target，隐含层都是对多个词向量的叠加平均。

**不同的是，**

- CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；
- CBOW的输入单词被one-hot编码过，fastText的输入特征是被embedding过；
- CBOW的输出是目标词汇，fastText的输出是文档对应的类标。

**值得注意的是，fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。**这两个知识点在前文中已经讲过，这里不再赘述。

fastText相关公式的推导和CBOW非常类似，这里也不展开了。



## 4. fastText核心思想

现在抛开那些不是很讨人喜欢的公式推导，来想一想fastText文本分类的核心思想是什么？

仔细观察模型的后半部分，即从隐含层输出到输出层输出，会发现它就是一个softmax线性多类别分类器，分类器的输入是一个用来表征当前文档的向量；

模型的前半部分，即从输入层输入到隐含层输出部分，主要在做一件事情：生成用来表征文档的向量。那么它是如何做的呢？**叠加构成这篇文档的所有词及n-gram的词向量，然后取平均。**叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。

**于是fastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。**这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类。



## 5. 输出分类的效果

还有个问题，就是为何fastText的分类效果常常不输于传统的非线性分类器？

**假设我们有两段文本：**

肚子 饿了 我 要 吃饭

肚子 饿了 我 要 吃东西

这两段文本意思几乎一模一样，如果要分类，肯定要分到同一个类中去。但在传统的分类器中，用来表征这两段文本的向量可能差距非常大。传统的文本分类中，你需要计算出每个词的权重，比如TF-IDF值， “吃饭”和“吃东西” 算出的TF-IDF值相差可能会比较大，其它词类似，于是，VSM（向量空间模型）中用来表征这两段文本的文本向量差别可能比较大。

**但是fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度**，于是，在fastText模型中，这两段文本的向量应该是非常相似的，于是，它们很大概率会被分到同一个类中。

使用词embedding而非词本身作为特征，这是fastText效果好的一个原因；另一个原因就是字符级n-gram特征的引入对分类效果会有一些提升 。



## 6. fastText与Word2Vec的不同

有意思的是，fastText和Word2Vec的作者是同一个人。

**相同点**：

- 图模型结构很像，都是采用embedding向量的形式，得到word的隐向量表达。
- 都采用很多相似的优化方法，比如使用Hierarchical softmax优化训练和预测中的打分速度。

之前一直不明白fasttext用层次softmax时叶子节点是啥，CBOW很清楚，它的叶子节点是词和词频，后来看了源码才知道，其实fasttext叶子节点里是类标和类标的频数。

|      | Word2Vec                              | fastText                              |
| ---- | ------------------------------------- | ------------------------------------- |
| 输入 | one-hot形式的单词的向量               | embedding过的单词的词向量和n-gram向量 |
| 输出 | 对应的是每一个term,计算某term概率最大 | 对应的是分类的标签。                  |

**本质不同，体现在softmax的使用：**

word2vec的目的是得到词向量，该词向量最终是在输入层得到的，输出层对应的h-softmax也会生成一系列的向量，但是最终都被抛弃，不会使用。

fastText则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label

**fastText优点**：

1. **适合大型数据+高效的训练速度**：能够训练模型“在使用标准多核CPU的情况下10分钟内处理超过10亿个词汇”
2. **支持多语言表达**：利用其语言形态结构，fastText能够被设计用来支持包括英语、德语、西班牙语、法语以及捷克语等多种语言。FastText的性能要比时下流行的word2vec工具明显好上不少，也比其他目前最先进的词态词汇表征要好。
3. **专注于文本分类**，在许多标准问题上实现当下最好的表现（例如文本倾向性分析或标签预测）。



## 7. 代码实现

清华文本分类数据集下载：[https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip](https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip)

[新闻文本分类代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.2%20fastText/fastText.ipynb)



## 8. 参考文献

[fastText原理及实践](http://www.52nlp.cn/fasttext)

------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
