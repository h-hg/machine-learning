# CNN

MINIST 数据集为黑白

卷积运算，与信号处理的卷积运算是等价的。

output 的 (x, y) = 核偏移(x,y) 后与 input 卷积的结果。 

多个核心，代表不同的角度去观看这一张图片，例如边缘检测、人脸检测等等。

先猜测一下，对于一张图片（只考虑黑白，单层通道），如果有多个 kernel 的话，那么卷积后的 Tensor 的三维的

概念说明

- Input_channels：黑白为 1，彩色为 3，
- kernel_channels：kernel 的个数
- kernel_size：
- stride：两个标量
- padding：两个标量。

一个例子

输入 x： [b, 3, 28, 28]，图片的大小为 28 * 28，通道数为 RGB，3 个，一共 b 张图片

one kernel：[3, 4, 4] 表示与其对应的 3 个通道，4 * 4 表示卷积和的大小。

multi-kernel：[16, 3, 4, 4] 16 表示一共 16 个卷积核

bias：[16] 每一个卷积核都有一个偏置。

out：[b, 16, 25, 25]

```python
layer = torch.nn.Conv2d(3, 16, kernel_size=4, stride=1, padding=0)
# 这是使用类进行调用，类里面封装好了参数
```

函数的调用方法

```python
x = torch.ones(1, 3, 28, 28)
w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
out = F.conv2(x, w, b, stride=1, padding=1)
```

下采样，feature map 变小，如 pooling，上采样，将 feature map 变大

这中下采样如何求导，如果要自己定义的话，该怎么实现。例如 subsampling，隔行采样。

```python
torch.nn.MaxPool2d
torch.nn.AvgPool2d
```

上采样

```python
import torch.nn.functional as F
F.interpolate
```

batch normalization

在 cv 中，输入为 [b, c, HW]，经过 batch normalization 后，输出 c 个均值和方差，每一个均值方差都是利用了 b 张图片的某个通道的数值。

其中均值和方差都是不用学的（即无需 requires_grad=True），z' = gamma * (x - u) / delta + beta，gamma 和 beta 都是需要学习的，这里是引入偏移。

图像处理比较特殊，应该它有多个u 和 delta

```python
import torch
x = torch.rand(100, 3, 28 * 28)
# 100 张 3 三通道 大小为 28 * 28 的图片
layer = torch.nn.BatchNorm1d(3) # 3 表示通道数
out = layer(x)
print(layer.running_mean) # 当前 batch 的均值（为什么视频说这个是全局的均值，即总的均值，而不是当前 batch 的均值，后面再查查文档）
print(layer.running_var) # 当前 batch 的方差
```
 
```python
import torch
x = torch.rand(100, 3, 28, 28)
layer = torch.nn.BatchNorm2d(3)
out = layer(x)
layer.weight # gamma
layer.bias # beta
```

同 `dropout` 一样，有分 train 模式和 valid 模式。需要使用 `.eval()`

优点

1. converge faster
2. better performance
3. robust

## LeNet-5

- CNN

## AlexNet

分布式训练（可以理解为利用多块 GPU），现在 PyTorch 等框架对这些的支持都挺好的了，无需自己写。

- Max pooling
- ReLU
- dropout

## VGG

## GoogLeNet

同时使用多个不同大小的 kernel，然后 引入 Filter conncatenation，合并这些结果

## ResNet

深度残差网络，保证 30 层网络至少比 22 好