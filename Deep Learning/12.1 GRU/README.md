## 目录
- [1. 什么是GRU](#1-什么是gru)
- [2. ⻔控循环单元](#2-⻔控循环单元)
  - [2.1 重置门和更新门](#21-重置门和更新门)
  - [2.2 候选隐藏状态](#22-候选隐藏状态)
  - [2.3 隐藏状态](#23-隐藏状态)
- [3. 代码实现GRU](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.1%20GRU/GRU.ipynb)
- [4. 参考文献](#4-参考文献)

## 1. 什么是GRU

在循环神经⽹络中的梯度计算⽅法中，我们发现，当时间步数较⼤或者时间步较小时，**循环神经⽹络的梯度较容易出现衰减或爆炸。虽然裁剪梯度可以应对梯度爆炸，但⽆法解决梯度衰减的问题。**通常由于这个原因，循环神经⽹络在实际中较难捕捉时间序列中时间步距离较⼤的依赖关系。 

**门控循环神经⽹络（gated recurrent neural network）的提出，正是为了更好地捕捉时间序列中时间步距离较⼤的依赖关系。** 它通过可以学习的⻔来控制信息的流动。其中，门控循环单元（gatedrecurrent unit，GRU）是⼀种常⽤的门控循环神经⽹络。

## 2. ⻔控循环单元

### 2.1 重置门和更新门

GRU它引⼊了**重置⻔（reset gate）和更新⻔（update gate）** 的概念，从而修改了循环神经⽹络中隐藏状态的计算⽅式。 

门控循环单元中的重置⻔和更新⻔的输⼊均为当前时间步输⼊ ![](https://latex.codecogs.com/gif.latex?X_t)与上⼀时间步隐藏状态![](https://latex.codecogs.com/gif.latex?H_{t-1})，输出由激活函数为sigmoid函数的全连接层计算得到。 如下图所示：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-16_13-36-14.png)

具体来说，假设隐藏单元个数为 h，给定时间步 t 的小批量输⼊ ![](https://latex.codecogs.com/gif.latex?X_t\in_{}\mathbb{R}^{n*d})（样本数为n，输⼊个数为d）和上⼀时间步隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_{t-1}\in_{}\mathbb{R}^{n*h})。重置⻔ ![](https://latex.codecogs.com/gif.latex?H_t\in_{}\mathbb{R}^{n*h})和更新⻔ ![](https://latex.codecogs.com/gif.latex?Z_t\in_{}\mathbb{R}^{n*h})的计算如下：

![](https://latex.codecogs.com/gif.latex?R_t=\sigma(X_tW_{xr}+H_{t-1}W_{hr}+b_r))

![](https://latex.codecogs.com/gif.latex?Z_t=\sigma(X_tW_{xz}+H_{t-1}W_{hz}+b_z))

sigmoid函数可以将元素的值变换到0和1之间。因此，重置⻔ ![](https://latex.codecogs.com/gif.latex?R_t)和更新⻔ ![](https://latex.codecogs.com/gif.latex?Z_t)中每个元素的值域都是[0*,* 1]。

### 2.2 候选隐藏状态

接下来，⻔控循环单元将计算候选隐藏状态来辅助稍后的隐藏状态计算。我们将当前时间步重置⻔的输出与上⼀时间步隐藏状态做按元素乘法（符号为*⊙*）。如果重置⻔中元素值接近0，那么意味着重置对应隐藏状态元素为0，即丢弃上⼀时间步的隐藏状态。如果元素值接近1，那么表⽰保留上⼀时间步的隐藏状态。然后，将按元素乘法的结果与当前时间步的输⼊连结，再通过含激活函数tanh的全连接层计算出候选隐藏状态，其所有元素的值域为[-1,1]。

 ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-16_13-49-52.png)

具体来说，时间步 t 的候选隐藏状态 ![](https://latex.codecogs.com/gif.latex?\tilde{H}\in_{}\mathbb{R}^{n*h})的计算为：

![](https://latex.codecogs.com/gif.latex?\tilde{H}_t=tanh(X_tW_{xh}+(R_t⊙H_{t-1})W_{hh}+b_h))

从上⾯这个公式可以看出，重置⻔控制了上⼀时间步的隐藏状态如何流⼊当前时间步的候选隐藏状态。而上⼀时间步的隐藏状态可能包含了时间序列截⾄上⼀时间步的全部历史信息。因此，重置⻔可以⽤来丢弃与预测⽆关的历史信息。

### 2.3 隐藏状态

最后，时间步*t*的隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_t\in_{}\mathbb{R}^{n*h})的计算使⽤当前时间步的更新⻔ ![](https://latex.codecogs.com/gif.latex?Z_t)来对上⼀时间步的隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_{t-1})和当前时间步的候选隐藏状态 ![](https://latex.codecogs.com/gif.latex?\tilde{H}_t)做组合：

 ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-16_13-58-58.png)

值得注意的是，**更新⻔可以控制隐藏状态应该如何被包含当前时间步信息的候选隐藏状态所更新，** 如上图所⽰。假设更新⻔在时间步![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-16_15-26-24.png)之间⼀直近似1。那么，在时间步![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-16_15-27-55.png)间的输⼊信息⼏乎没有流⼊时间步 t 的隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_t)实际上，这可以看作是较早时刻的隐藏状态 ![](https://latex.codecogs.com/gif.latex?H_{t^{′}-1})直通过时间保存并传递⾄当前时间步 t。这个设计可以应对循环神经⽹络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较⼤的依赖关系。 

我们对⻔控循环单元的设计稍作总结：

- 重置⻔有助于捕捉时间序列⾥短期的依赖关系；
- 更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。

## 3. 代码实现GRU

[MNIST--GRU实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/12.1%20GRU/GRU.ipynb)

## 4. 参考文献

[《动手学--深度学习》](http://zh.gluon.ai)

------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>
