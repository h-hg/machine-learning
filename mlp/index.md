# 多层感知机

## Single Dense Layer

单层全连接层如下图所示：

![single-dense](single-dense2.drawio.svg)

其中：

- $\boldsymbol{a}^{(l-1)} = (a_1^{(l-1)}, \dots, a_{n_{l-1}}^{(l-1)})^\mathrm{T}$：第 $l-1$ 层的输入向量
- $\boldsymbol{w_i}^{(l)} = (w_{1i}^{(l)}, \dots, w_{n_{l}i}^{(l)})^\mathrm{T}$：第 $l$ 层权重矩阵的第 $i$ 个（列）分量
- $\boldsymbol{b} = (b_1^{(l)}, \dots, b_{n_l}^{(l)})^\mathrm{T}$：第 $l$ 层的偏置向量
- $\boldsymbol{z}^{(l)} = (z_1^{(l)}, \dots, z_{n_l}^{(l)})^\mathrm{T}$：第 $l$ 层的中间输出向量（不考虑激活函数）
- $\boldsymbol{g}^{(l)} = (g_1^{(l)}, g_2^{l}, \dots, g_{n_l}^{(l)})^\mathrm{T}$：第 $l$ 层的激活函数构成的向量函数
- $\boldsymbol{a}^{(l)} = (a_1^{(l)}, \dots, a_{n_{l}}^{(l)})^\mathrm{T}$：第 $l$ 层的输出向量

则

$$
z_i^{(l)} = (\boldsymbol{a}^{l-1})^\mathrm{T} \cdot \boldsymbol{w_i}^{(l)} + b_i^{(l)} \\
\boldsymbol{a}^{(l)} = \boldsymbol{g}^{(l)}(\boldsymbol{z}^{(l)})
$$

记

$$
\mathbf{W}^{(l)} = (\boldsymbol{w_1}^{(l)}, \dots, \boldsymbol{w_{n_l}}^{(l)}) =
\begin{bmatrix}
    w_{11}^{(l)} & w_{12}^{(l)} & \dots & w_{1n_l}^{(l)} \\
    w_{21}^{(l)} & w_{22}^{(l)} & \dots & w_{2n_l}^{(l)} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{n_{l-1}1}^{(l)} & w_{n_{l-1}2}^{(l)} & \dots & w_{n_{l-1}n_l}^{(l)}
\end{bmatrix}
_{n_{l-1} \times n_l}
$$

则

$$
\boldsymbol{z}^{(l)} = ({\boldsymbol{a}^{(l-1)}}^\mathrm{T} \cdot \mathbf{W}^{(l)} + {\boldsymbol{b}^{(l)}}^\mathrm{T})^\mathrm{T}
$$

上面的数学公式只是每次考虑到输入一个样本（向量） $\boldsymbol{a}^{(l-1)}$，现在我们考虑输入多个样本（矩阵）。

记

$$
\mathbf{A}^{l-1} = (\boldsymbol{a_1}^{(l-1)}, \dots, \boldsymbol{a_m}^{(l-1)})^\mathrm{T} \in \mathbb{R}^{m \times n_{l-1}} \\
\mathbf{Z}^{l} = (\boldsymbol{z_1}^{(l)}, \dots, \boldsymbol{z_m}^{(l)})^\mathrm{T} \in \mathbb{R}^{m \times n_l}
$$

其中 $m$ 为样本的个数，则

$$
\mathbf{Z}^{l} = \mathbf{A}^{l-1} \mathbf{W}^{(l)} + \boldsymbol{1}_{m \times 1} {\boldsymbol{b}^{(l)}}^\mathrm{T}
$$

## 细节

[权重如何初始化](https://medium.com/ai-in-plain-english/weight-initialization-in-neural-network-9b3935192f6)

## 实现

### Numpy

### Tensorflow

### Pytorch

## 参考资料

- [Deep Dive into Math Behind Deep Networks](https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba)
- [Let’s code a Neural Network in plain NumPy](https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)