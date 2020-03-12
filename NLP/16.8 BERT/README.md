## 目录
- [1. 什么是BERT](#1-什么是bert)
- [2. 从Word Embedding到Bert模型的发展](#2-从word-embedding到bert模型的发展)
  - [2.1 图像的预训练](#21-图像的预训练)
  - [2.2 Word Embedding](#22-word-embedding)
  - [2.3 ELMO](#23-elmo)
  - [2.4 GPT](#24-gpt)
  - [2.5 BERT](#25-bert)
- [3. BERT的评价](#3-bert的评价)
- [4. 代码实现](#4-代码实现)
- [5. 参考文献](#5-参考文献)

## 1. 什么是BERT

**BERT的全称是Bidirectional Encoder Representation from Transformers**，是Google2018年提出的预训练模型，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

Bert最近很火，应该是最近最火爆的AI进展，网上的评价很高，那么Bert值得这么高的评价吗？我个人判断是值得。那为什么会有这么高的评价呢？是因为它有重大的理论或者模型创新吗？其实并没有，从模型创新角度看一般，创新不算大。但是架不住效果太好了，基本刷新了很多NLP的任务的最好性能，有些任务还被刷爆了，这个才是关键。另外一点是Bert具备广泛的通用性，就是说绝大部分NLP任务都可以采用类似的两阶段模式直接去提升效果，这个第二关键。客观的说，把Bert当做最近两年NLP重大进展的集大成者更符合事实。



## 2. 从Word Embedding到Bert模型的发展

### 2.1 图像的预训练

自从深度学习火起来后，预训练过程就是做图像或者视频领域的一种比较常规的做法，有比较长的历史了，而且这种做法很有效，能明显促进应用的效果。

![](https://pic3.zhimg.com/80/v2-4c27ee0ff1fb87f27d55b007cb4ceb06_hd.jpg)

那么图像领域怎么做预训练呢，上图展示了这个过程，

1. 我们设计好网络结构以后，对于图像来说一般是CNN的多层叠加网络结构，可以先用某个训练集合比如训练集合A或者训练集合B对这个网络进行预先训练，在A任务上或者B任务上学会网络参数，然后存起来以备后用。

2. 假设我们面临第三个任务C，网络结构采取相同的网络结构，在比较浅的几层CNN结构，网络参数初始化的时候可以加载A任务或者B任务学习好的参数，其它CNN高层参数仍然随机初始化。

3. 之后我们用C任务的训练数据来训练网络，此时有两种做法：

   **一种**是浅层加载的参数在训练C任务过程中不动，这种方法被称为“Frozen”;

   **另一种**是底层网络参数尽管被初始化了，在C任务训练过程中仍然随着训练的进程不断改变，这种一般叫“Fine-Tuning”，顾名思义，就是更好地把参数进行调整使得更适应当前的C任务。

一般图像或者视频领域要做预训练一般都这么做。这样做的优点是：如果手头任务C的训练集合数据量较少的话，利用预训练出来的参数来训练任务C，加个预训练过程也能极大加快任务训练的收敛速度，所以这种预训练方式是老少皆宜的解决方案，另外疗效又好，所以在做图像处理领域很快就流行开来。

**为什么预训练可行**

对于层级的CNN结构来说，不同层级的神经元学习到了不同类型的图像特征，由底向上特征形成层级结构，所以预训练好的网络参数，尤其是底层的网络参数抽取出特征跟具体任务越无关，越具备任务的通用性，所以这是为何一般用底层预训练好的参数初始化新任务网络参数的原因。而高层特征跟任务关联较大，实际可以不用使用，或者采用Fine-tuning用新数据集合清洗掉高层无关的特征抽取器。



### 2.2 Word Embedding

![](https://pic2.zhimg.com/80/v2-e2842dd9bc442893bd53dd9fa32d6c9d_hd.jpg)

神经网络语言模型(NNLM)的思路。先说训练过程。学习任务是输入某个句中单词 ![[公式]](https://www.zhihu.com/equation?tex=W_t=（Bert）) 前面句子的t-1个单词，要求网络正确预测单词Bert，即最大化：

![[公式]](https://www.zhihu.com/equation?tex=++P%28W_t%3D%E2%80%9CBert%E2%80%9D%7CW_1%2CW_2%2C%E2%80%A6W_%28t-1%29%3B%CE%B8%29)

前面任意单词 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 用Onehot编码（比如：0001000）作为原始单词输入，之后乘以矩阵Q后获得向量 ![[公式]](https://www.zhihu.com/equation?tex=C%28W_i+%29) ，每个单词的 ![[公式]](https://www.zhihu.com/equation?tex=C%28W_i+%29) 拼接，上接隐层，然后接softmax去预测后面应该后续接哪个单词。这个 ![[公式]](https://www.zhihu.com/equation?tex=C%28W_i+%29) 是什么？这其实就是单词对应的Word Embedding值，那个矩阵Q包含V行，V代表词典大小，每一行内容代表对应单词的Word embedding值。只不过Q的内容也是网络参数，需要学习获得，训练刚开始用随机值初始化矩阵Q，当这个网络训练好之后，矩阵Q的内容被正确赋值，每一行代表一个单词对应的Word embedding值。所以你看，通过这个网络学习语言模型任务，这个网络不仅自己能够根据上文预测后接单词是什么，同时获得一个副产品，就是那个矩阵Q，这就是单词的Word Embedding。

2013年最火的用语言模型做Word Embedding的工具是Word2Vec，后来又出了Glove，Word2Vec。对于这两个模型不熟悉的可以参考我之前的文章，这里不再赘述：

- [Word2Vec](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.1%20Word%20Embedding)
- [GloVe](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.3%20GloVe)

上面这种模型做法就是18年之前NLP领域里面采用预训练的典型做法，之前说过，Word Embedding其实对于很多下游NLP任务是有帮助的，只是帮助没有大到闪瞎忘记戴墨镜的围观群众的双眼而已。那么新问题来了，为什么这样训练及使用Word Embedding的效果没有期待中那么好呢？答案很简单，因为Word Embedding有问题呗。这貌似是个比较弱智的答案，关键是Word Embedding存在什么问题？这其实是个好问题。

**这片在Word Embedding头上笼罩了好几年的乌云是什么？是多义词问题。**我们知道，多义词是自然语言中经常出现的现象，也是语言灵活性和高效性的一种体现。多义词对Word Embedding来说有什么负面影响？如上图所示，比如多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

有没有简单优美的解决方案呢？ELMO提供了一种简洁优雅的解决方案。



### 2.3 ELMO

ELMO是“Embedding from Language Models”的简称，其实这个名字并没有反应它的本质思想，提出ELMO的论文题目：“Deep contextualized word representation”更能体现其精髓，而精髓在哪里？在deep contextualized这个短语，一个是deep，一个是context，其中context更关键。

在此之前的Word Embedding本质上是个静态的方式，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的Word Embedding不会跟着上下文场景的变化而改变，所以对于比如Bank这个词，它事先学好的Word Embedding中混合了几种语义 ，在应用中来了个新句子，即使从上下文中（比如句子包含money等词）明显可以看出它代表的是“银行”的含义，但是对应的Word Embedding内容也不会变，它还是混合了多种语义。这是为何说它是静态的，这也是问题所在。

**ELMO的本质思想是**：我事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，不过这没关系。在我实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候我可以根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。

![](https://pic4.zhimg.com/80/v2-fe335ea9fdcd6e0e5ec4a9ac0e2290db_hd.jpg)

ELMO采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的Word Embedding作为新特征补充到下游任务中。

上图展示的是其预训练过程，它的网络结构采用了双层双向LSTM，目前语言模型训练的任务目标是根据单词 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 的上下文去正确预测单词 ![[公式]](https://www.zhihu.com/equation?tex=W_i) ， ![[公式]](https://www.zhihu.com/equation?tex=W_i) 之前的单词序列Context-before称为上文，之后的单词序列Context-after称为下文。

图中左端的前向双层LSTM代表正方向编码器，输入的是从左到右顺序的除了预测单词外 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 的上文Context-before；右端的逆向双层LSTM代表反方向编码器，输入的是从右到左的逆序的句子下文Context-after；每个编码器的深度都是两层LSTM叠加。

这个网络结构其实在NLP中是很常用的。使用这个网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子 ![[公式]](https://www.zhihu.com/equation?tex=Snew) ，句子中每个单词都能得到对应的三个Embedding:

- 最底层是单词的Word Embedding；
- 往上走是第一层双向LSTM中对应单词位置的Embedding，这层编码单词的句法信息更多一些；
- 再往上走是第二层LSTM中对应单词位置的Embedding，这层编码单词的语义信息更多一些。

也就是说，ELMO的预训练过程不仅仅学会单词的Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。

![](https://pic2.zhimg.com/80/v2-ef6513ff29e3234011221e4be2e97615_hd.jpg)

上面介绍的是ELMO的第一阶段：预训练阶段。那么预训练好网络结构后，**如何给下游任务使用呢**？上图展示了下游任务的使用过程，比如我们的下游任务仍然是QA问题:

1. 此时对于问句X，我们可以先将句子X作为预训练好的ELMO网络的输入，这样句子X中每个单词在ELMO网络中都能获得对应的三个Embedding；
2. 之后给予这三个Embedding中的每一个Embedding一个权重a，这个权重可以学习得来，根据各自权重累加求和，将三个Embedding整合成一个；
3. 然后将整合后的这个Embedding作为X句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用。对于上图所示下游任务QA中的回答句子Y来说也是如此处理。

因为ELMO给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为“Feature-based Pre-Training”。

**前面我们提到静态Word Embedding无法解决多义词的问题，那么ELMO引入上下文动态调整单词的embedding后多义词问题解决了吗？解决了，而且比我们期待的解决得还要好**。对于Glove训练出的Word Embedding来说，多义词比如play，根据它的embedding找出的最接近的其它单词大多数集中在体育领域，这很明显是因为训练数据中包含play的句子中体育领域的数量明显占优导致；而使用ELMO，根据上下文动态调整后的embedding不仅能够找出对应的“演出”的相同语义的句子，而且还可以保证找出的句子中的play对应的词性也是相同的，这是超出期待之处。之所以会这样，是因为我们上面提到过，第一层LSTM编码了很多句法信息，这在这里起到了重要作用。

**ELMO有什么值得改进的缺点呢**？

- 首先，一个非常明显的缺点在特征抽取器选择方面，ELMO使用了LSTM而不是新贵Transformer，Transformer是谷歌在17年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，很多研究已经证明了Transformer提取特征的能力是要远强于LSTM的。如果ELMO采取Transformer作为特征提取器，那么估计Bert的反响远不如现在的这种火爆场面。
- 另外一点，ELMO采取双向拼接这种融合特征的能力可能比Bert一体化的融合特征方式弱，但是，这只是一种从道理推断产生的怀疑，目前并没有具体实验说明这一点。



### 2.4 GPT

![](https://pic1.zhimg.com/80/v2-5028b1de8fb50e6630cc9839f0b16568_hd.jpg)

GPT是“Generative Pre-Training”的简称，从名字看其含义是指的生成式的预训练。GPT也采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过Fine-tuning的模式解决下游任务。

上图展示了GPT的预训练过程，其实和ELMO是类似的，主要不同在于两点：

- 首先，特征抽取器不是用的RNN，而是用的Transformer，上面提到过它的特征抽取能力要强于RNN，这个选择很明显是很明智的；
- 其次，GPT的预训练虽然仍然是以语言模型作为目标任务，但是采用的是单向的语言模型，所谓“单向”的含义是指：语言模型训练的任务目标是根据 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 单词的上下文去正确预测单词 ![[公式]](https://www.zhihu.com/equation?tex=W_i) ， ![[公式]](https://www.zhihu.com/equation?tex=W_i) 之前的单词序列Context-before称为上文，之后的单词序列Context-after称为下文。

如果对Transformer模型不太了解的，可以参考我写的文章：[Transformer](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.7%20Transformer)

ELMO在做语言模型预训练的时候，预测单词 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 同时使用了上文和下文，而GPT则只采用Context-before这个单词的上文来进行预测，而抛开了下文。这个选择现在看不是个太好的选择，原因很简单，它没有把单词的下文融合进来，这限制了其在更多应用场景的效果，比如阅读理解这种任务，在做任务的时候是可以允许同时看到上文和下文一起做决策的。如果预训练时候不把单词的下文嵌入到Word Embedding中，是很吃亏的，白白丢掉了很多信息。



### 2.5 BERT

Bert采用和GPT完全相同的两阶段模型，首先是语言模型预训练；其次是使用Fine-Tuning模式解决下游任务。和GPT的最主要不同在于在预训练阶段采用了类似ELMO的双向语言模型，即双向的Transformer，当然另外一点是语言模型的数据规模要比GPT大。所以这里Bert的预训练过程不必多讲了。模型结构如下：

![](https://github.com/NLP-LOVE/ML-NLP/raw/master/images/2019-9-28_21-34-11.png)

对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向rnn和双向rnn的区别，直觉上来讲效果会好一些。

对比ELMo，虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7C+w_1%2C+...w_%7Bi-1%7D%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7Cw_%7Bi%2B1%7D%2C+...w_n%29) 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 ![[公式]](https://www.zhihu.com/equation?tex=P%28w_i%7Cw_1%2C++...%2Cw_%7Bi-1%7D%2C+w_%7Bi%2B1%7D%2C...%2Cw_n%29) 作为目标函数训练LM。

BERT预训练模型分为以下三个步骤：**Embedding、Masked LM、Next Sentence Prediction**

#### 2.5.1 Embedding

这里的Embedding由三种Embedding求和而成：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-28_20-8-22.png)

- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
- Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的



#### 2.5.2 Masked LM

MLM可以理解为完形填空，作者会随机mask每一个句子中15%的词，用其上下文来做预测，例如：my dog is hairy → my dog is [MASK]

此处将hairy进行了mask处理，然后采用非监督学习的方法预测mask位置的词是什么，但是该方法有一个问题，因为是mask15%的词，其数量已经很高了，这样就会导致某些词在fine-tuning阶段从未见过，为了解决这个问题，作者做了如下的处理：

80%是采用[mask]，my dog is hairy → my dog is [MASK]

10%是随机取一个词来代替mask的词，my dog is hairy -> my dog is apple

10%保持不变，my dog is hairy -> my dog is hairy

**注意：这里的10%是15%需要mask中的10%**

那么为啥要以一定的概率使用随机词呢？这是因为transformer要保持对每个输入token分布式的表征，否则Transformer很可能会记住这个[MASK]就是"hairy"。至于使用随机词带来的负面影响，文章中解释说,所有其他的token(即非"hairy"的token)共享15%*10% = 1.5%的概率，其影响是可以忽略不计的。Transformer全局的可视，又增加了信息的获取，但是不让模型获取全量信息。



#### 2.5.3 Next Sentence Prediction

选择一些句子对A与B，其中50%的数据B是A的下一条句子，剩余50%的数据B是语料库中随机选择的，学习其中的相关性，添加这样的预训练的目的是目前很多NLP的任务比如QA和NLI都需要理解两个句子之间的关系，从而能让预训练的模型更好的适应这样的任务。
个人理解：

- Bert先是用Mask来提高视野范围的信息获取量，增加duplicate再随机Mask，这样跟RNN类方法依次训练预测没什么区别了除了mask不同位置外；
- 全局视野极大地降低了学习的难度，然后再用A+B/C来作为样本，这样每条样本都有50%的概率看到一半左右的噪声；
- 但直接学习Mask A+B/C是没法学习的，因为不知道哪些是噪声，所以又加上next_sentence预测任务，与MLM同时进行训练，这样用next来辅助模型对噪声/非噪声的辨识，用MLM来完成语义的大部分的学习。



## 3. BERT的评价

总结下BERT的主要贡献：

- 引入了Masked LM，使用双向LM做模型预训练。
- 为预训练引入了新目标NSP，它可以学习句子与句子间的关系。
- 进一步验证了更大的模型效果更好： 12 --> 24 层。
- 为下游任务引入了很通用的求解框架，不再为任务做模型定制。
- 刷新了多项NLP任务的记录，引爆了NLP无监督预训练技术。

**BERT优点**

- Transformer Encoder因为有Self-attention机制，因此BERT自带双向功能。
- 因为双向功能以及多层Self-attention机制的影响，使得BERT必须使用Cloze版的语言模型Masked-LM来完成token级别的预训练。
- 为了获取比词更高级别的句子级别的语义表征，BERT加入了Next Sentence Prediction来和Masked-LM一起做联合训练。
- 为了适配多任务下的迁移学习，BERT设计了更通用的输入层和输出层。
- 微调成本小。

**BERT缺点**

- task1的随机遮挡策略略显粗犷，推荐阅读《Data Nosing As Smoothing In Neural Network Language Models》。
- [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现。每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）。
- BERT对硬件资源的消耗巨大（大模型需要16个tpu，历时四天；更大的模型需要64个tpu，历时四天。

**评价**

Bert是NLP里里程碑式的工作，对于后面NLP的研究和工业应用会产生长久的影响，这点毫无疑问。但是从上文介绍也可以看出，从模型或者方法角度看，Bert借鉴了ELMO，GPT及CBOW，主要提出了Masked 语言模型及Next Sentence Prediction，但是这里Next Sentence Prediction基本不影响大局，而Masked LM明显借鉴了CBOW的思想。所以说Bert的模型没什么大的创新，更像最近几年NLP重要进展的集大成者，这点如果你看懂了上文估计也没有太大异议，如果你有大的异议，杠精这个大帽子我随时准备戴给你。如果归纳一下这些进展就是：

- 首先是两阶段模型，第一阶段双向语言模型预训练，这里注意要用双向而不是单向，第二阶段采用具体任务Fine-tuning或者做特征集成；
- 第二是特征抽取要用Transformer作为特征提取器而不是RNN或者CNN；
- 第三，双向语言模型可以采取CBOW的方法去做（当然我觉得这个是个细节问题，不算太关键，前两个因素比较关键）。

Bert最大的亮点在于效果好及普适性强，几乎所有NLP任务都可以套用Bert这种两阶段解决思路，而且效果应该会有明显提升。可以预见的是，未来一段时间在NLP应用领域，Transformer将占据主导地位，而且这种两阶段预训练方法也会主导各种应用。



## 4. 代码实现

[bert中文分类实践](https://github.com/NLP-LOVE/ML-NLP/blob/master/NLP/16.8%20BERT/bert-Chinese-classification-task.md)



## 5. 参考文献

- [【NLP】Google BERT详解](https://zhuanlan.zhihu.com/p/46652512)
- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
- [一文读懂BERT(原理篇)](https://blog.csdn.net/jiaowoshouzi/article/details/89073944)



------

> 作者:[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub:[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号:【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
