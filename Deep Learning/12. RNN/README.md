## 目录
- [1. 什么是RNN](#1-什么是rnn)
  - [1.1 RNN的应用](#11-rnn的应用)
  - [1.2 为什么有了CNN，还要RNN?](#12-为什么有了cnn还要rnn)
  - [1.3 RNN的网络结构](#13-rnn的网络结构)
  - [1.4 双向RNN](#14-双向rnn)
  - [1.5 BPTT算法](#15-bptt算法)
- [2. 其它类型的RNN](#2-其它类型的rnn)
- [3. CNN与RNN的区别](#3-cnn与rnn的区别)
- [4. 为什么RNN 训练的时候Loss波动很大](#4-为什么rnn-训练的时候loss波动很大)
- [5. 实例代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.%20RNN/RNN.ipynb)

## 1. 什么是RNN

> 循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network）

### 1.1 RNN的应用

- 文本生成(生成序列)
- 机器翻译
- 看图说话
- 文本(情感)分析
- 智能客服
- 聊天机器人
- 语音识别
- 搜索引擎
- 个性化推荐

### 1.2 为什么有了CNN，还要RNN?

- 传统神经网络(包括CNN)，输入和输出都是互相独立的。图像上的猫和狗是分隔开的，但有些任务，后续的输出和之前的内容是相关的。例如：我是中国人，我的母语是____。这是一道填空题，需要依赖于之前的输入。
- 所以，RNN引入“记忆”的概念，也就是输出需要依赖于之前的输入序列，并把关键输入记住。循环2字来源于其每个元素都执行相同的任务。
- 它并⾮刚性地记忆所有固定⻓度的序列，而是通过隐藏状态来存储之前时间步的信息。

### 1.3 RNN的网络结构

首先先上图，然后再解释：

![](http://wx1.sinaimg.cn/mw690/00630Defly1g5x6xyfcadj30zs0g83zh.jpg)

现在我们考虑输⼊数据存在时间相关性的情况。假设 ![](https://latex.codecogs.com/gif.latex?X_t\in_{}\mathbb{R}^{n*d})是序列中时间步t的小批量输⼊， ![](https://latex.codecogs.com/gif.latex?H_t\in_{}\mathbb{R}^{n*h})是该时间步的隐藏变量。那么根据以上结构图当前的隐藏变量的公式如下：

![](https://latex.codecogs.com/gif.latex?H_t=\phi(X_tW_{xh}+H_{t-1}W_{hh}+b_h))

从以上公式我们可以看出，这⾥我们保存上⼀时间步的隐藏变量 ![](https://latex.codecogs.com/gif.latex?H_{t-1})，并引⼊⼀个新的权重参数，该参数⽤来描述在当前时间步如何使⽤上⼀时间步的隐藏变量。具体来说，**时间步 t 的隐藏变量的计算由当前时间步的输⼊和上⼀时间步的隐藏变量共同决定。** ![](https://latex.codecogs.com/gif.latex?\phi)**函数其实就是激活函数。**

我们在这⾥添加了 ![](https://latex.codecogs.com/gif.latex?H_{t-1}W_{hh})⼀项。由上式中相邻时间步的隐藏变量 ![](https://latex.codecogs.com/gif.latex?H_t 和H_{t-1})之间的关系可知，这⾥的隐藏变量能够捕捉截⾄当前时间步的序列的历史信息，就像是神经⽹络当前时间步的状态或记忆⼀样。因此，该隐藏变量也称为隐藏状态。**由于隐藏状态在当前时间步的定义使⽤了上⼀时间步的隐藏状态，上式的计算是循环的。使⽤循环计算的⽹络即循环神经⽹络（recurrent neural network）。**

在时间步t，输出层的输出和多层感知机中的计算类似：

![](https://latex.codecogs.com/gif.latex?O_t=H_tW_{hq}+b_q)

### 1.4 双向RNN

之前介绍的循环神经⽹络模型都是假设当前时间步是由前⾯的较早时间步的序列决定的，因此它
们都将信息通过隐藏状态从前往后传递。有时候，当前时间步也可能由后⾯时间步决定。例如，
当我们写下⼀个句⼦时，可能会根据句⼦后⾯的词来修改句⼦前⾯的⽤词。**双向循环神经⽹络通过增加从后往前传递信息的隐藏层来更灵活地处理这类信息。**下图演⽰了⼀个含单隐藏层的双向循环神经⽹络的架构。

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5x6s399c9j30ju0drq3n.jpg)

在双向循环神经⽹络的架构中，设该时间步正向隐藏状态为 ![](https://latex.codecogs.com/gif.latex?\overrightarrow{H}_t\in_{}\mathbb{R}^{n*h})(正向隐藏单元个数为h)，反向隐藏状态为 ![](https://latex.codecogs.com/gif.latex?\overleftarrow{H}_t\in_{}\mathbb{R}^{n*h})(反向隐藏单元个数为h)。我们可以分别
计算正向隐藏状态和反向隐藏状态：

![](https://latex.codecogs.com/gif.latex?\overrightarrow{H}_t=\phi(X_tW_{xh}^{(f)}+\overrightarrow{H}_{t-1}W_{hh}^{(f)}+b_h^{(f)}))

![](https://latex.codecogs.com/gif.latex?\overleftarrow{H}_t=\phi(X_tW_{xh}^{(b)}+\overleftarrow{H}_{t-1}W_{hh}^{(b)}+b_h^{(b)}))

然后我们连结两个⽅向的隐藏状态 ![](https://latex.codecogs.com/gif.latex?\overrightarrow{H}_t和\overleftarrow{H}_t)来得到隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_t\in_{}\mathbb{R}^{n*2h})，并将其输⼊到输出层。输出层计算输出 ![](https://latex.codecogs.com/gif.latex?O_t\in_{}\mathbb{R}^{n*q})(输出个数为q)：

![](https://latex.codecogs.com/gif.latex?O_t=H_tW_{hq}+b_q)

双向循环神经⽹络在每个时间步的隐藏状态同时取决于该时间步之前和之后的⼦序列（包
括当前时间步的输⼊）。

### 1.5 BPTT算法

![image](https://ws3.sinaimg.cn/large/00630Defly1g2xolucuo8j30go06mq5d.jpg)

在之前你已经见过对于前向传播（上图蓝色箭头所指方向）怎样在神经网络中从左到右地计算这些激活项，直到输出所有地预测结果。而对于反向传播，我想你已经猜到了，反向传播地计算方向（上图红色箭头所指方向）与前向传播基本上是相反的。

我们先定义一个元素**损失函数：**

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-15_14-21-20.png)

整个序列的损失函数：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-15_14-23-38.png)

在这个计算图中，通过 ![](https://latex.codecogs.com/gif.latex?y^{'(1)})可以计算对应的损失函数，于是计算出第一个时间步的损失函数，然后计算出第二个时间步的损失函数，然后是第三个时间步，一直到最后一个时间步，最后为了计算出总体损失函数，我们要把它们都加起来，通过等式计算出最后的𝐿，也就是把每个单独时间步的损失函数都加起来。然后你就可以通过导数相关的参数，用梯度下降法来更新参数。 

在这个反向传播的过程中，最重要的信息传递或者说最重要的递归运算就是这个从右到左的运算，这也就是为什么这个算法有一个很别致的名字，叫做**“通过（穿越）时间反向传播（backpropagation through time）”。**取这个名字的原因是对于前向传播，你需要从左到右进行计算，在这个过程中，时刻𝑡不断增加。而对于反向传播，你需要从右到左进行计算，就像时间倒流。“通过时间反向传播”，就像穿越时光，这种说法听起来就像是你需要一台时光机来实现这个算法一样。

## 2. 其它类型的RNN

- **One to one：** 这个可能没有那么重要，这就是一个小型的标准的神经网络，输入𝑥然后得到输出𝑦。

- **One to many：** 音乐生成，你的目标是使用一个神经网络输出一些音符。对应于一段音乐，输入𝑥 

  可以是一个整数，表示你想要的音乐类型或者是你想要的音乐的第一个音符，并且如果你什么都不想输入，𝑥可以是空的输入，可设为 0 向量。

- **Many to one：** 句子分类问题，输入文档，输出文档的类型。

- **Many to many()：** 命名实体识别。

- **Many to many()：** 机器翻译。

![image](https://wx1.sinaimg.cn/large/00630Defly1g2xq26dpz1j30go09341y.jpg)

## 3. CNN与RNN的区别

| 类别   | 特点描述                                                     |
| ------ | ------------------------------------------------------------ |
| 相同点 | 1、传统神经网络的扩展。<br/>2、前向计算产生结果，反向计算模型更新。<br/>3、每层神经网络横向可以多个神经元共存,纵向可以有多层神经网络连接。 |
| 不同点 | 1、CNN空间扩展，神经元与特征卷积；RNN时间扩展，神经元与多个时间输出计算<br/>2、RNN可以用于描述时间上连续状态的输出，有记忆功能，CNN用于静态输出 |

## 4. 为什么RNN 训练的时候Loss波动很大

由于RNN特有的memory会影响后期其他的RNN的特点，梯度时大时小，learning rate没法个性化的调整，导致RNN在train的过程中，Loss会震荡起伏，为了解决RNN的这个问题，在训练的时候，可以设置临界值，当梯度大于某个临界值，直接截断，用这个临界值作为梯度的大小，防止大幅震荡。

## 5. 实例代码

[TensorFlow实现RNN](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.%20RNN/RNN.ipynb)

------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
