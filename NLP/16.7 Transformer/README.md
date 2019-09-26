## 目录
- [1. 什么是Transformer](#1-什么是transformer)
- [2. Transformer结构](#2-transformer结构)
  - [2.1 总体结构](#21-总体结构)
  - [2.2 Encoder层结构](#22-encoder层结构)
  - [2.3 Decoder层结构](#23-decoder层结构)
  - [2.4 动态流程图](#24-动态流程图)
- [3. Transformer为什么需要进行Multi-head Attention](#3-transformer为什么需要进行multi-head-attention)
- [4. Transformer相比于RNN/LSTM，有什么优势？为什么？](#4-transformer相比于rnnlstm有什么优势为什么)
- [5. 为什么说Transformer可以代替seq2seq？](#5-为什么说transformer可以代替seq2seq)
- [6. 代码实现](#6-代码实现)
- [7. 参考文献](#7-参考文献)

## 1. 什么是Transformer

**[《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762.pdf)是一篇Google提出的将Attention思想发挥到极致的论文。这篇论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN**。目前大热的Bert就是基于Transformer构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。



## 2. Transformer结构

### 2.1 总体结构

Transformer的结构和Attention模型一样，Transformer模型中也采用了 encoer-decoder 架构。但其结构相比于Attention更加复杂，论文中encoder层由6个encoder堆叠在一起，decoder层也一样。

不了解Attention模型的，可以回顾之前的文章：[Attention](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.6%20Attention)

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-25_23-4-24.png)

每一个encoder和decoder的内部结构如下图：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-25_23-25-14.png)

- encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。
- decoder也包含encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。



### 2.2 Encoder层结构

首先，模型需要对输入的数据进行一个embedding操作，也可以理解为类似w2c的操作，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-25_23-8-46.png)



#### 2.2.1 Positional Encoding

transformer模型中缺少一种解释输入序列中单词顺序的方法，它跟序列模型还不不一样。为了处理这个问题，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下：

![](https://latex.codecogs.com/gif.latex?PE(pos,2i)=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}))

![](https://latex.codecogs.com/gif.latex?PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}))

其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**。

最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-25-43.png)



#### 2.2.2 Self-Attention

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路，我们看个例子：

The animal didn't cross the street because it was too tired

这里的 it 到底代表的是 animal 还是 street 呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把 it 和 animal 联系起来，接下来我们看下详细的处理过程。

1. 首先，self-attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64。

   ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_8-59-3.png)

2. 计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点成，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2。

   ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-1-36.png)

3. 接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大。

   ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-3-3.png)

4. 下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。

   ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-4-8.png)

在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵 Q 与 K 相乘，乘以一个常数，做softmax操作，最后乘上 V 矩阵。

**这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention。**

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-6-6.png)

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-6-41.png)



#### 2.2.3 Multi-Headed Attention

这篇论文更牛逼的地方是给self-attention加入了另外一个机制，被称为“multi-headed” attention，该机制理解起来很简单，**就是说不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组**，所以最后得到的结果是8个矩阵。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-13-0.png)

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-13-50.png)



#### 2.2.4 Layer normalization

在transformer中，每一个子层（self-attetion，Feed Forward Neural Network）之后都会接一个残缺模块，并且有一个Layer normalization。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-30-31.png)

Normalization有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为0方差为1的数据。我们在把数据送入激活函数之前进行normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。

**Batch Normalization**

BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。**BN的具体做法就是对每一小批数据，在批这个方向上做归一化。**

**Layer normalization**

它也是归一化数据的一种方式，不过**LN 是在每一个样本上计算均值和方差**，而不是BN那种在批方向计算均值和方差！公式如下：

![](https://latex.codecogs.com/gif.latex?LN(x_i)=\alpha*\frac{x_i-\mu_L}{\sqrt{\sigma_L^2+\varepsilon}}+\beta)

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-35-22.png)



#### 2.2.5 Feed Forward Neural Network

这给我们留下了一个小的挑战，前馈神经网络没法输入 8 个矩阵呀，这该怎么办呢？所以我们需要一种方式，把 8 个矩阵降为 1 个，首先，我们把 8 个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘，最后得到一个最终的矩阵。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-26_9-17-14.png)



### 2.3 Decoder层结构

根据上面的总体结构图可以看出，decoder部分其实和encoder部分大同小异，刚开始也是先添加一个位置向量Positional Encoding，方法和 2.2.1 节一样，接下来接的是masked mutil-head attetion，这里的mask也是transformer一个很关键的技术，下面我们会进行一一介绍。

其余的层结构与Encoder一样，请参考Encoder层结构。



#### 2.3.1 masked mutil-head attetion

**mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果**。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

1. **padding mask**

   什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

   具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！

   而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。

2. **Sequence mask**

   文章前面也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。

   那么具体怎么做呢？也很简单：**产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的**。

- 对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个mask相加作为attn_mask。
- 其他情况，attn_mask 一律等于 padding mask。



#### 2.3.2 Output层

当decoder层全部执行完毕后，怎么把得到的向量映射为我们需要的词呢，很简单，只需要在结尾再添加一个全连接层和softmax层，假如我们的词典是1w个词，那最终softmax会输入1w个词的概率，概率值最大的对应的词就是我们最终的结果。



### 2.4 动态流程图

编码器通过处理输入序列开启工作。顶端编码器的输出之后会变转化为一个包含向量K（键向量）和V（值向量）的注意力向量集 ，**这是并行化操作**。这些向量将被每个解码器用于自身的“编码-解码注意力层”，而这些层可以帮助解码器关注输入序列哪些位置合适：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64156846894583861613.gif)

在完成编码阶段后，则开始解码阶段。解码阶段的每个步骤都会输出一个输出序列（在这个例子里，是英语翻译的句子）的元素。

接下来的步骤重复了这个过程，直到到达一个特殊的终止符号，它表示transformer的解码器已经完成了它的输出。每个步骤的输出在下一个时间步被提供给底端解码器，并且就像编码器之前做的那样，这些解码器会输出它们的解码结果 。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64156846899939997439.gif)



## 3. Transformer为什么需要进行Multi-head Attention

原论文中说到进行Multi-head Attention的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来。其实直观上也可以想到，如果自己设计这样的一个模型，必然也不会只做一次attention，多次attention综合的结果至少能够起到增强模型的作用，也可以类比CNN中同时使用**多个卷积核**的作用，直观上讲，多头的注意力**有助于网络捕捉到更丰富的特征/信息**。



## 4. Transformer相比于RNN/LSTM，有什么优势？为什么？

1. RNN系列的模型，并行计算能力很差。RNN并行计算的问题就出在这里，因为 T 时刻的计算依赖 T-1 时刻的隐层计算结果，而 T-1 时刻的计算依赖 T-2 时刻的隐层计算结果，如此下去就形成了所谓的序列依赖关系。

2. Transformer的特征抽取能力比RNN系列的模型要好。

   具体实验对比可以参考：[放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)

   但是值得注意的是，并不是说Transformer就能够完全替代RNN系列的模型了，任何模型都有其适用范围，同样的，RNN系列模型在很多任务上还是首选，熟悉各种模型的内部原理，知其然且知其所以然，才能遇到新任务时，快速分析这时候该用什么样的模型，该怎么做好。  



## 5. 为什么说Transformer可以代替seq2seq？

**seq2seq缺点**：这里用代替这个词略显不妥当，seq2seq虽已老，但始终还是有其用武之地，seq2seq最大的问题在于**将Encoder端的所有信息压缩到一个固定长度的向量中**，并将其作为Decoder端首个隐藏状态的输入，来预测Decoder端第一个单词(token)的隐藏状态。在输入序列比较长的时候，这样做显然会损失Encoder端的很多信息，而且这样一股脑的把该固定向量送入Decoder端，Decoder端不能够关注到其想要关注的信息。  

**Transformer优点**：transformer不但对seq2seq模型这两点缺点有了实质性的改进(多头交互式attention模块)，而且还引入了self-attention模块，让源序列和目标序列首先“自关联”起来，这样的话，源序列和目标序列自身的embedding表示所蕴含的信息更加丰富，而且后续的FFN层也增强了模型的表达能力，并且Transformer并行计算的能力是远远超过seq2seq系列的模型，因此我认为这是transformer优于seq2seq模型的地方。  



## 6. 代码实现

地址：[https://github.com/Kyubyong/transformer](https://github.com/Kyubyong/transformer)

代码解读：[Transformer解析与tensorflow代码解读](https://www.cnblogs.com/zhouxiaosong/p/11032431.html)



## 7. 参考文献

- [Transformer模型详解](https://blog.csdn.net/u012526436/article/details/86295971)
- [图解Transformer（完整版）](https://blog.csdn.net/longxinchen_ml/article/details/86533005)
- [关于Transformer的若干问题整理记录](https://www.nowcoder.com/discuss/258321)



------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>









