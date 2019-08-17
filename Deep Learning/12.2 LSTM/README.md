## 目录
- [1. 什么是LSTM](#1-什么是lstm)
- [2. 输⼊⻔、遗忘⻔和输出⻔](#2-输遗忘和输出)
- [3. 候选记忆细胞](#3-候选记忆细胞)
- [4. 记忆细胞](#4-记忆细胞)
- [5. 隐藏状态](#5-隐藏状态)
- [6. LSTM与GRU的区别](#6-lstm与gru的区别)
- [7. LSTM可以使用别的激活函数吗？](#7-lstm可以使用别的激活函数吗)
- [8. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.2%20LSTM/LSTM.ipynb)
- [9. 参考文献](#9-参考文献)

## 1. 什么是LSTM

在你阅读这篇文章时候，你都是基于自己已经拥有的对先前所见词的理解来推断当前词的真实含义。我们不会将所有的东西都全部丢弃，然后用空白的大脑进行思考。我们的思想拥有持久性。LSTM就是具备了这一特性。

这篇将介绍另⼀种常⽤的⻔控循环神经⽹络：**⻓短期记忆（long short-term memory，LSTM）[1]。**它⽐⻔控循环单元的结构稍微复杂⼀点，也是为了解决在RNN网络中梯度衰减的问题，是GRU的一种扩展。

可以先理解GRU的过程，在来理解LSTM会容易许多，链接地址：[三步理解--门控循环单元(GRU)](https://blog.csdn.net/weixin_41510260/article/details/99679481)

LSTM 中引⼊了3个⻔，即输⼊⻔（input gate）、遗忘⻔（forget gate）和输出⻔（output gate），以及与隐藏状态形状相同的记忆细胞（某些⽂献把记忆细胞当成⼀种特殊的隐藏状态），从而记录额外的信息。

## 2. 输⼊⻔、遗忘⻔和输出⻔

与⻔控循环单元中的重置⻔和更新⻔⼀样，⻓短期记忆的⻔的输⼊均为当前时间步输⼊Xt与上⼀时间步隐藏状态Ht−1，输出由激活函数为sigmoid函数的全连接层计算得到。如此⼀来，这3个⻔元素的值域均为[0, 1]。如下图所示：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-17_16-1-43.png)

具体来说，假设隐藏单元个数为 h，给定时间步 t 的小批量输⼊ ![](https://latex.codecogs.com/gif.latex?X_t\in_{}\mathbb{R}^{n*d})（样本数为n，输⼊个数为d）和上⼀时间步隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_{t-1}\in_{}\mathbb{R}^{n*h})。三个门的公式如下：

**输入门：** ![](https://latex.codecogs.com/gif.latex?I_t=\sigma(X_tW_{xi}+H_{t-1}W_{hi}+b_i))

**遗忘问：** ![](https://latex.codecogs.com/gif.latex?F_t=\sigma(X_tW_{xf}+H_{t-1}W_{hf}+b_f))

**输出门：** ![](https://latex.codecogs.com/gif.latex?O_t=\sigma(X_tW_{xo}+H_{t-1}W_{ho}+b_o))

## 3. 候选记忆细胞

接下来，⻓短期记忆需要计算候选记忆细胞 ![](https://latex.codecogs.com/gif.latex?\tilde{C}_t)。它的计算与上⾯介绍的3个⻔类似，但使⽤了值域在[−1, 1]的tanh函数作为激活函数，如下图所示：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-17_16-24-39.png)

具体来说，时间步t的候选记忆细胞计算如下：

![](https://latex.codecogs.com/gif.latex?\tilde{C}_t=tanh(X_tWxc+H_{t-1}W_{hc}+b_c))

## 4. 记忆细胞

我们可以通过元素值域在[0, 1]的输⼊⻔、遗忘⻔和输出⻔来控制隐藏状态中信息的流动，这⼀般也是通过使⽤按元素乘法（符号为⊙）来实现的。当前时间步记忆细胞![](https://latex.codecogs.com/gif.latex?H_{t}\in_{}\mathbb{R}^{n*h})的计算组合了上⼀时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘⻔和输⼊⻔来控制信息的流动：

![](https://latex.codecogs.com/gif.latex?C_t=F_t⊙C_{t-1}+I_t⊙\tilde{C}_t)

如下图所⽰，遗忘⻔控制上⼀时间步的记忆细胞Ct−1中的信息是否传递到当前时间步，而输⼊⻔则控制当前时间步的输⼊Xt通过候选记忆细胞C˜t如何流⼊当前时间步的记忆细胞。如果遗忘⻔⼀直近似1且输⼊⻔⼀直近似0，过去的记忆细胞将⼀直通过时间保存并传递⾄当前时间步。这个设计可以应对循环神经⽹络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较⼤的依赖关系。

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-17_16-32-50.png)

## 5. 隐藏状态

有了记忆细胞以后，接下来我们还可以通过输出⻔来控制从记忆细胞到隐藏状态Ht的信
息的流动：

![](https://latex.codecogs.com/gif.latex?H_t=O_t⊙tanh(C_t))

这⾥的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出⻔近似1时，记忆细胞信息将传递到隐藏状态供输出层使⽤；当输出⻔近似0时，记忆细胞信息只⾃⼰保留。**下图展⽰了⻓短期记忆中隐藏状态的全部计算：**

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-17_16-37-3.png)

## 6. LSTM与GRU的区别

LSTM与GRU二者结构十分相似，**不同在于**：

1. 新的记忆都是根据之前状态及输入进行计算，但是GRU中有一个重置门控制之前状态的进入量，而在LSTM里没有类似门；
2. 产生新的状态方式不同，LSTM有两个不同的门，分别是遗忘门(forget gate)和输入门(input gate)，而GRU只有一种更新门(update gate)；
3. LSTM对新产生的状态可以通过输出门(output gate)进行调节，而GRU对输出无任何调节。
4. GRU的优点是这是个更加简单的模型，所以更容易创建一个更大的网络，而且它只有两个门，在计算性上也运行得更快，然后它可以扩大模型的规模。 
5. LSTM更加强大和灵活，因为它有三个门而不是两个。

## 7. LSTM可以使用别的激活函数吗？

关于激活函数的选取，在LSTM中，遗忘门、输入门和输出门使用Sigmoid函数作为激活函数；在生成候选记忆时，使用双曲正切函数Tanh作为激活函数。

值得注意的是，这两个激活函数都是饱和的，也就是说在输入达到一定值的情况下，输出就不会发生明显变化了。如果是用非饱和的激活函数，例如ReLU，那么将难以实现门控的效果。

 Sigmoid函数的输出在0～1之间，符合门控的物理定义。且当输入较大或较小时，其输出会非常接近1或0，从而保证该门开或关。在生成候选记忆时，使用Tanh函数，是因为其输出在−1～1之间，这与大多数场景下特征分布是0中心的吻合。此外，Tanh函数在输入为0附近相比Sigmoid函数有更大的梯度，通常使模型收敛更快。

激活函数的选择也不是一成不变的，但要选择合理的激活函数。

## 8. 代码实现

[MIST数据分类--TensorFlow实现LSTM](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.2%20LSTM/LSTM.ipynb)

## 9. 参考文献

[《动手学--深度学习》](http://zh.gluon.ai)

------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
