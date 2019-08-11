## 目录
- [1. 什么是CNN](#1-什么是cnn)
  - [1.1 输入层](#11-输入层)
  - [1.2 卷积计算层(conv)](#12-卷积计算层conv)
  - [1.3 激励层](#13-激励层)
  - [1.4 池化层](#14-池化层)
  - [1.5 全连接层](#15-全连接层)
  - [1.6 层次结构小结](#16-层次结构小结)
  - [1.7 CNN优缺点](#17-cnn优缺点)
- [2. 典型CNN发展历程](#2-典型cnn发展历程)
- [3. 图像相关任务](#3-图像相关任务)
  - [3.1 图像识别与定位](#31-图像识别与定位)
  - [3.2 物体检测(object detection)](#32-物体检测object-detection)
  - [3.3 语义(图像)分割](#33-语义图像分割)
- [4. 代码实现CNN](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/11.%20CNN/CNN.ipynb)
- [5. 参考文献](#5-参考文献)

## 1. 什么是CNN

> 卷积神经网络（Convolutional Neural Networks, CNN）是一类包含[卷积](https://baike.baidu.com/item/卷积/9411006)计算且具有深度结构的[前馈神经网络](https://baike.baidu.com/item/前馈神经网络/7580523)（Feedforward Neural Networks），是[深度学习](https://baike.baidu.com/item/深度学习/3729729)（deep learning）的代表算法之一。

我们先来看卷积神经网络各个层级结构图：

![](http://wx2.sinaimg.cn/mw690/00630Defgy1g5pqv7pv9uj30yv0gpn1e.jpg)

上图中CNN要做的事情是：给定一张图片，是车还是马未知，是什么车也未知，现在需要模型判断这张图片里具体是一个什么东西，总之输出一个结果：如果是车 那是什么车。

- 最左边是数据**输入层**(input layer)，对数据做一些处理，比如去均值（把输入数据各个维度都中心化为0，避免数据过多偏差，影响训练效果）、归一化（把所有的数据都归一到同样的范围）、PCA/白化等等。CNN只对训练集做“去均值”这一步。
- CONV：卷积计算层(conv layer)，线性乘积求和。
- RELU：激励层(activation layer)，下文有提到：ReLU是激活函数的一种。
- POOL：池化层(pooling layer)，简言之，即取区域平均或最大。
- FC：全连接层(FC layer)。

这几个部分中，卷积计算层是CNN的核心。

### 1.1 输入层

在做输入的时候，需要把图片处理成同样大小的图片才能够进行处理。

常见的处理数据的方式有：

1. 去均值(**常用**)

   - **AlexNet**：训练集中100万张图片，对每个像素点求均值，得到均值图像，当训练时用原图减去均值图像。
   - **VGG**：对所有输入在三个颜色通道R/G/B上取均值，只会得到3个值，当训练时减去对应的颜色通道均值。(**此种方法效率高**)

   **TIPS:**在训练集和测试集上减去训练集的均值。

2. 归一化

   幅度归一化到同样的范围。

3. PCA/白化(**很少用**)

   - 用PCA降维
   - 白化是对数据每个特征轴上的幅度归一化。

### 1.2 卷积计算层(conv)

对图像（不同的数据窗口数据）和滤波矩阵（一组固定的权重：因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器filter）做**内积**（逐个元素相乘再求和）的操作就是所谓的『卷积』操作，也是卷积神经网络的名字来源。

滤波器filter是什么呢！请看下图。图中左边部分是原始输入数据，图中中间部分是滤波器filter，图中右边是输出的新的二维数据。

![image](https://ws3.sinaimg.cn/large/00630Defgy1g2djuqln1xj30i80dy104.jpg)

不同的滤波器filter会得到不同的输出数据，比如颜色深浅、轮廓。**相当于提取图像的不同特征，模型就能够学习到多种特征。**用不同的滤波器filter，提取想要的关于图像的特定信息：颜色深浅或轮廓。如下图所示。

![](http://wx3.sinaimg.cn/mw690/00630Defgy1g5r30db3jpj30hv0b1wq3.jpg)

在CNN中，滤波器filter（带着一组固定权重的神经元）对局部输入数据进行卷积计算。每计算完一个数据窗口内的局部数据后，数据窗口不断平移滑动，直到计算完所有数据。这个过程中，有这么几个参数： 

- 深度depth：神经元个数，决定输出的depth厚度。同时代表滤波器个数。
- 步长stride：决定滑动多少步可以到边缘。
- 填充值zero-padding：在外围边缘补充若干圈0，方便从初始位置以步长为单位可以刚好滑倒末尾位置，通俗地讲就是为了总长能被步长整除。 

![](http://wx3.sinaimg.cn/mw690/00630Defgy1g5r34yi6p4j308g05baaj.jpg)

![](https://mlnotebook.github.io/img/CNN/convSobel.gif)

- **参数共享机制**

  假设每个神经元连接数据窗的权重是固定对的。固定每个神经元连接权重，可以看做模板，每个神经元只关注**一个特性(模板)**，这使得需要估算的权重个数减少：一层中从1亿到3.5万。

- 一组固定的权重和不同窗口内数据做**内积**：卷积

- 作用在于捕捉某一种模式，具体表现为很大的值。

**卷积操作的本质特性包括稀疏交互和参数共享**。

### 1.3 激励层

把卷积层输出结果做非线性映射。

激活函数有：

![UTOOLS1556084241657.png](https://i.loli.net/2019/04/24/5cbff6153eef3.png)

- sigmoid：在两端斜率接近于0，梯度消失。
- ReLu：修正线性单元，有可能出现斜率为0，但概率很小，因为mini-batch是一批样本损失求导之和。

**TIPS:**

- CNN慎用sigmoid！慎用sigmoid！慎用sigmoid！
- 首先试RELU，因为快，但要小心点。
- 如果RELU失效，请用 Leaky ReLU或者Maxout。
- 某些情况下tanh倒是有不错的结果，但是很少。

### 1.4 池化层

也叫**下采样层**，就算通过了卷积层，纬度还是很高 ，需要进行池化层操作。

- 夹在连续的卷积层中间。
- 压缩数据和参数的量，降低维度。
- 减小过拟合。
- 具有特征不变性。

方式有：**Max pooling、average pooling**

![image](https://ws1.sinaimg.cn/large/00630Defgy1g2doejs89ej30ez0dhdj1.jpg)

**Max pooling**

取出每个部分的最大值作为输出，例如上图左上角的4个黄色方块取最大值为3作为输出，以此类推。

**average pooling**

每个部分进行计算得到平均值作为输出，例如上图左上角的4个黄色方块取得平均值2作为输出，以此类推。

### 1.5 全连接层

全连接层的每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全连接的特性，一般全连接层的参数也是最多的。

- 两层之间所有神经元都有权重连接
- 通常全连接层在卷积神经网络尾部

### 1.6 层次结构小结

| CNN层次结构 | 作用                                                       |
| ----------- | ---------------------------------------------------------- |
| 输入层      | 卷积网络的原始输入，可以是原始或预处理后的像素矩阵         |
| 卷积层      | 参数共享、局部连接，利用平移不变性从全局特征图提取局部特征 |
| 激活层      | 将卷积层的输出结果进行非线性映射                           |
| 池化层      | 进一步筛选特征，可以有效减少后续网络层次所需的参数量       |
| 全连接层    | 用于把该层之前提取到的特征综合起来。                       |

### 1.7 CNN优缺点

**优点：**

- 共享卷积核，优化计算量。
- 无需手动选取特征，训练好权重，即得特征。
- 深层次的网络抽取图像信息丰富，表达效果好。
- 保持了层级网络结构。
- 不同层次有不同形式与功能。

**缺点：**

- 需要调参，需要大样本量，GPU等硬件依赖。
- 物理含义不明确。

**与NLP/Speech共性：**

都存在局部与整体的关系，由低层次的特征经过组合，组成高层次的特征，并且得到不同特征之间的空间相关性。

## 2. 典型CNN发展历程

- LeNet，这是最早用于数字识别的CNN 
- AlexNet， 2012 ILSVRC比赛远超第2名的CNN，比LeNet更深，用多层小卷积层叠加替换单大卷积层。 
- ZF Net， 2013 ILSVRC比赛冠军 
- GoogLeNet， 2014 ILSVRC比赛冠军 
- VGGNet， 2014 ILSVRC比赛中的模型，图像识别略差于GoogLeNet，但是在很多图像转化学习问题(比如objectdetection)上效果很好 
- ResNet(深度残差网络（Deep Residual Network，ResNet）)， 2015ILSVRC比赛冠军，结构修正(残差学习)以适应深层次CNN训练。 
- DenseNet， CVPR2017 best paper，把ResNet的add变成concat 

## 3. 图像相关任务

![image](https://ws1.sinaimg.cn/large/00630Defly1g2lmm3f2e5j30wj0clwtx.jpg)

### 3.1 图像识别与定位

1. **classification：**C个类别识别

   - **input**：Image
   - **Output**：类别标签
   - **Evaluation metric**：准确率

2. **Localization定位)**

   - **Input**：Image

   - **Output**：物体边界框(x,y,w,h)

   - **Evaluation metric**：交并准则(IOU) > 0.5   图中阴影部分所占面积

     ![image](https://ws1.sinaimg.cn/large/00630Defly1g2lmuahlsoj30df08rweb.jpg)

#### 3.1.1 思路1：识别+定位过程

1. **识别**可以看作多分类问题(**用softmax**)，用别人训练好的CNN模型做fine-tune

2. **定位**的目标是(x,y,w,h)是连续值，当回归问题解决(**mse**)

   在**1**的CNN尾部展开(例如把最后一层拿开)，接上一个(x,y,w,h)的神经网络，成为**classification+regression的模型**。

   更细致的识别可以提前规定好有k个组成部分，做成k个部分的回归，

   **例如：**框出两只眼睛和两条腿，4元祖*4=16(个连续值)

3. Regression部分用欧氏距离损失，使用SGD训练。

![image](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1517393046_527.png)

#### 3.1.2 思路2：图窗+识别

- 类似刚才的classification+regression思路
- 咱们取不同大小的“框”
- 让框出现在不同的位置
- 判定得分
- 按照得分的高低对“结果框”做抽样和合并

![](http://wx1.sinaimg.cn/mw690/00630Defgy1g5r50m0d5wj30ow0e1dpg.jpg)

### 3.2 物体检测(object detection)

#### 3.2.1 过程

当图像有很多物体怎么办的？难度可是一下暴增啊。

那任务就变成了：多物体识别+定位多个物体，那把这个任务看做分类问题？

![image](https://wx4.sinaimg.cn/large/00630Defly1g2lnprfz23j30fp0gytix.jpg)

看成分类问题有何不妥？

- 你需要找很多位置， 给很多个不同大小的框
- 你还需要对框内的图像分类
- 当然， 如果你的GPU很强大， 恩， 那加油做吧…

**边缘策略：**想办法先找到可能包含内容的图框(**候选框**)，然后进行分类问题的识别。

**方法**：根据RGB值做区域融合。**fast-CNN**，共享图窗，从而加速候选框的形成。

- **R-CNN => fast-CNN => faster-RCNN** 速度对比

  ![image](https://wx3.sinaimg.cn/large/00630Defly1g2lnwlqz87j30rk090418)

#### 3.2.2 R-CNN

R-CNN的简要步骤如下：

1. 输入测试图像。
2. 利用选择性搜索Selective Search算法在图像中从下到上提取2000个左右的可能包含物体的候选区域Region Proposal。
3. 因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN，将CNN的fc7层的输出作为特征。
4. 将每个Region Proposal提取到的CNN特征输入到SVM进行分类。

#### 3.2.3 SPP-Net

SPP：Spatial Pyramid Pooling（空间金字塔池化），SPP-Net是出自2015年发表在IEEE上的论文。

众所周知，CNN一般都含有卷积部分和全连接部分，其中，卷积层不需要固定尺寸的图像，而全连接层是需要固定大小的输入。所以当全连接层面对各种尺寸的输入数据时，就需要对输入数据进行crop（crop就是从一个大图扣出网络输入大小的patch，比如227×227），或warp（把一个边界框bounding box(红框)的内容resize成227×227）等一系列操作以统一图片的尺寸大小，比如224*224（ImageNet）、32*32(LenNet)、96*96等。

![](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1525249316_603.png)

所以才如你在上文中看到的，在R-CNN中，“因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN”。

但warp/crop这种预处理，导致的问题要么被拉伸变形、要么物体不全，限制了识别精确度。没太明白？说句人话就是，一张16:9比例的图片你硬是要Resize成1:1的图片，你说图片失真不？

SPP Net的作者Kaiming He等人逆向思考，既然由于全连接FC层的存在，普通的CNN需要通过固定输入图片的大小来使得全连接层的输入固定。那借鉴卷积层可以适应任何尺寸，为何不能在卷积层的最后加入某种结构，使得后面全连接层得到的输入变成固定的呢？

这个“化腐朽为神奇”的结构就是spatial pyramid pooling layer。

它的特点有两个:

1. **结合空间金字塔方法实现CNNs的多尺度输入。**

   SPP Net的第一个贡献就是在最后一个卷积层后，接入了金字塔池化层，保证传到下一层全连接层的输入固定。

   换句话说，在普通的CNN机构中，输入图像的尺寸往往是固定的（比如224*224像素），输出则是一个固定维数的向量。SPP Net在普通的CNN结构中加入了ROI池化层（ROI Pooling），使得网络的输入图像可以是任意尺寸的，输出则不变，同样是一个固定维数的向量。

   简言之，CNN原本只能固定输入、固定输出，CNN加上SSP之后，便能任意输入、固定输出。神奇吧？

2. **只对原图提取一次卷积特征**

   在R-CNN中，每个候选框先resize到统一大小，然后分别作为CNN的输入，这样是很低效的。

   而SPP Net根据这个缺点做了优化：只对原图进行一次卷积计算，便得到整张图的卷积特征feature map，然后找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer和之后的层，完成特征提取工作。

如此这般，R-CNN要对每个区域计算卷积，而SPPNet只需要计算一次卷积，从而节省了大量的计算时间，比R-CNN有一百倍左右的提速。

#### 3.2.4 Fast R-CNN

SPP Net真是个好方法，R-CNN的进阶版Fast R-CNN就是在R-CNN的基础上采纳了SPP Net方法，对R-CNN作了改进，使得性能进一步提高。

R-CNN有一些相当大的缺点（把这些缺点都改掉了，就成了Fast R-CNN）。

**大缺点：**由于每一个候选框都要独自经过CNN，这使得花费的时间非常多。

**解决：**共享卷积层，现在不是每一个候选框都当做输入进入CNN了，而是输入一张完整的图片，在第五个卷积层再得到每个候选框的特征。

原来的方法：许多候选框（比如两千个）-->CNN-->得到每个候选框的特征-->分类+回归

现在的方法：一张完整图片-->CNN-->得到每张候选框的特征-->分类+回归

所以容易看见，Fast R-CNN相对于R-CNN的提速原因就在于：不过不像R-CNN把每个候选区域给深度网络提特征，而是整张图提一次特征，再把候选框映射到conv5上，而SPP只需要计算一次特征，剩下的只需要在conv5层上操作就可以了。

**算法步骤：**

1. 在图像中确定约1000-2000个候选框 (使用选择性搜索)。
2. 对整张图片输进CNN，得到feature map。
3. 找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer和之后的层。
4. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类。
5. 对于属于某一类别的候选框，用回归器进一步调整其位置。

#### 3.2.5 Faster R-CNN

Fast R-CNN存在的问题：存在瓶颈：选择性搜索，找出所有的候选框，这个也非常耗时。那我们能不能找出一个更加高效的方法来求出这些候选框呢？

解决：加入一个提取边缘的神经网络，也就说找到候选框的工作也交给神经网络来做了。

所以，rgbd在Fast R-CNN中引入Region Proposal Network(RPN)替代Selective Search，同时引入anchor box应对目标形状的变化问题（anchor就是位置和大小固定的box，可以理解成事先设置好的固定的proposal）。这就是Faster R-CNN。

**算法步骤：**

1. 对整张图片输进CNN，得到feature map。
2. 卷积特征输入到RPN，得到候选框的特征信息。
3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类。
4. 对于属于某一类别的候选框，用回归器进一步调整其位置。

#### 3.2.6 YOLO

Faster R-CNN的方法目前是主流的目标检测方法，但是速度上并不能满足实时的要求。YOLO一类的方法慢慢显现出其重要性，这类方法使用了回归的思想，利用整张图作为网络的输入，直接在图像的多个位置上回归出这个位置的目标边框，以及目标所属的类别。

我们直接看上面YOLO的目标检测的流程图：

![](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1525171091_647.jpg)

1. 给个一个输入图像，首先将图像划分成7\*7的网格。
2. 对于每个网格，我们都预测2个边框（包括每个边框是目标的置信度以及每个边框区域在多个类别上的概率）。
3. 根据上一步可以预测出7*7*2个目标窗口，然后根据阈值去除可能性比较低的目标窗口，最后NMS去除冗余窗口即可。

**小结：**YOLO将目标检测任务转换成一个回归问题，大大加快了检测的速度，使得YOLO可以每秒处理45张图像。而且由于每个网络预测目标窗口时使用的是全图信息，使得false positive比例大幅降低（充分的上下文信息）。

但是YOLO也存在问题：没有了Region Proposal机制，只使用7*7的网格回归会使得目标不能非常精准的定位，这也导致了YOLO的检测精度并不是很高。

#### 3.2.7 SSD

SSD: Single Shot MultiBox Detector。上面分析了YOLO存在的问题，使用整图特征在7*7的粗糙网格内回归对目标的定位并不是很精准。那是不是可以结合region proposal的思想实现精准一些的定位？SSD结合YOLO的回归思想以及Faster R-CNN的anchor机制做到了这点。

![](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1525171268_230.jpg)

上图是SSD的一个框架图，首先SSD获取目标位置和类别的方法跟YOLO一样，都是使用回归，但是YOLO预测某个位置使用的是全图的特征，SSD预测某个位置使用的是这个位置周围的特征（感觉更合理一些）。

那么如何建立某个位置和其特征的对应关系呢？可能你已经想到了，使用Faster R-CNN的anchor机制。如SSD的框架图所示，假如某一层特征图(图b)大小是8\*8，那么就使用3*3的滑窗提取每个位置的特征，然后这个特征回归得到目标的坐标信息和类别信息(图c)。

不同于Faster R-CNN，这个anchor是在多个feature map上，这样可以利用多层的特征并且自然的达到多尺度（不同层的feature map 3*3滑窗感受野不同）。

小结：SSD结合了YOLO中的回归思想和Faster R-CNN中的anchor机制，使用全图各个位置的多尺度区域特征进行回归，既保持了YOLO速度快的特性，也保证了窗口预测的跟Faster R-CNN一样比较精准。SSD在VOC2007上mAP可以达到72.1%，速度在GPU上达到58帧每秒。

### 3.3 语义(图像)分割

识别图上pixel的类别，用全卷积网络。

![image](https://ws3.sinaimg.cn/large/00630Defly1g2lo1v1001j30nq0j4wu9.jpg)

## 4. 代码实现CNN

[cifar10数据集分类--CNN](https://github.com/NLP-LOVE/ML-NLP/blob/master/Deep%20Learning/11.%20CNN/CNN.ipynb)

## 5. 参考文献

1. [基于深度学习的目标检测技术演进：R-CNN、Fast R-CNN、Faster R-CNN、YOLO、SSD](https://www.julyedu.com/question/big/kp_id/32/ques_id/2103)
2. [通俗理解卷积神经网络](https://blog.csdn.net/v_july_v/article/details/51812459)

------

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>


