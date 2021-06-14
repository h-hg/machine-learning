# 感知机

简介：一种二分类的**线性**分类器，由 Rosenblatt 于 1957 年提出，是神经网络和支持向量机的基础。

- 输入：$\mathbb{D} = \{(\boldsymbol{x_1}, y_1)), \dots, (\boldsymbol{x_m}, y_m)\}, y_i \in \{-1, +1\}$
- 模型：$f(\boldsymbol{x}) = \mathrm{sign}(\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b)$
- 原理：寻找一个超平面 $S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 0$，使得不同类别的特征分布在超平面两侧，即
  - 正类: $\boldsymbol{w}^\mathrm{T} + b > 0$
  - 负类: $\boldsymbol{w}^\mathrm{T} + b < 0$

关于超平面 $S$ 的表示

假设 $S_0: \boldsymbol{w_0}^\mathrm{T} \boldsymbol{x} + b_0 = 0$ 是找到的最好的超平面，它在表示形式上有无数中表示，如 $(k\boldsymbol{w_0})^\mathrm{T} \boldsymbol{x} + kb_0 = 0, k \in \mathbb{R}$，为此引入规范化，定义要找的超平面满足 $\|\boldsymbol{w}\| = 1$，在此规范下，超平面唯一

点 $\boldsymbol{x_i}$ 到超平面 $S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 0$ 的距离为

$$
\begin{aligned}
    \cfrac{|\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b|}{\|\boldsymbol{w}\|}
    &= |\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b| \\
    &=
    \begin{cases}
        y_i (\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b)  & \boldsymbol{x_i} \text{ is classified correctly} \\
        - y_i (\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) & \boldsymbol{x_i} \text{ is is misclassified}
    \end{cases}  
\end{aligned}
$$

如何评价超平面 $S$ 的优劣

- 大多数的点被分类正确
- 分类不正确的点到超平面的距离越短

记 $\mathbb{D}_{error} = \{ \boldsymbol{x} \mid y_i (\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \le 0\}$，即 $\boldsymbol{x}$ 被分类错误的集合。

由此可以定义损失函数

$$
L = \sum_{\boldsymbol{x_i} \in \mathbb{D}_{error}} - y_i (\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b)
$$

故

$$
\begin{cases}
    \cfrac{\partial L}{\partial \boldsymbol{w}} = \sum_{\boldsymbol{x_i} \in \mathbb{D}_{error}} - y_i \boldsymbol{x_i} \\
    \cfrac{\partial L}{\partial b} = \sum_{\boldsymbol{x_i} \in \mathbb{D}_{error}} - y_i
\end{cases}
$$

## 实现

[numpy matrix](np-matrix.py ':include :type=code python')

## 对偶形式

假设 $\boldsymbol{w}$ 和 $b$ 都被初始化为零，$\boldsymbol{x_i}$ 在**所有**的迭代次数中被分类错误的次数为 $\alpha_i$，那么由梯度下降更新公式

$$
\begin{cases}
    \boldsymbol{w} = \boldsymbol{w} + \eta \sum_{\boldsymbol{x_i} \in \mathbb{D}_{error}} y_i \boldsymbol{x_i} \\
    b = b + \eta \sum_{\boldsymbol{x_i} \in \mathbb{D}_{error}} y_i
\end{cases}
$$

可以得到

$$
\begin{cases}
    \boldsymbol{w} = \eta \sum_{i=1}^m \alpha_i y_i \boldsymbol{x_i} \\
    b = \eta \sum_{i=1}^m \alpha_i y_i
\end{cases}
$$

$\alpha_i$ 的含义：如果 $\alpha_i$ 越大，说明这个样本点经常被误分。什么样的样本点容易被误分？很明显就是离超平面很近的点，超平面稍微移动一下，这个样本点就从正变负（或从负变正）

此时，我们的模型在表示形式上变成了

$$
\begin{aligned}
    f(\boldsymbol{x}) &= \mathrm{sign}(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) \\
    &= \mathrm{sign}((\eta \sum_{i=1}^m \alpha_i y_i \boldsymbol{x_i})^\mathrm{T} \boldsymbol{x} + \eta \sum_{i=1}^m \alpha_i y_i) \\
    &= \mathrm{sign}(\eta \sum_{i=1}^m \alpha_i y_i \boldsymbol{x_i}^\mathrm{T} \boldsymbol{x} + \eta \sum_{i=1}^m \alpha_i y_i) \\
\end{aligned}
$$


记 $\mathbf{X} = [\boldsymbol{x_1}, \dots, \boldsymbol{x_m}]^\mathrm{T}$，$\mathbf{X}$ 的行向量 Gram 矩阵 $\mathbf{G}_{m \times m} =[\boldsymbol{g_1}, \cdots, \boldsymbol{g_m}] = \mathbf{X} \mathbf{X}^\mathrm{T}$，即 $g_{ij} = \boldsymbol{x_i}^\mathrm{T} \boldsymbol{x_j}$。

则有

$$
\begin{aligned}
    f(\boldsymbol{x_i}) &= \mathrm{sign}(\eta \sum_{j=1}^m \alpha_j y_j \boldsymbol{x_j}^\mathrm{T} \boldsymbol{x_i} + \eta \sum_{i=j}^m \alpha_j y_j) \\
    &= \mathrm{sign}(\eta \sum_{j=1}^m \alpha_j y_j g_{ji} + \eta \sum_{i=j}^m \alpha_j y_j)
\end{aligned}
$$

相应地，学习的参数不再是 $\boldsymbol{w}$ 和 $b$，而是 $\alpha_i$

如何更新 $\alpha_i$，如果 $y_i(\eta \sum_{j=1}^m \alpha_j y_j g_{ji} + \eta \sum_{i=j}^m \alpha_j y_j) \le 0$，则$\alpha = \alpha + 1$

对偶形式的目的是通过先计算出 $\mathbf{G}$（时间复杂度 $\Theta(m^2)$）来降低训练时每次迭代的运算量，但是并不是在任何情况下都能降低运算量。

计算一个样本点 $\boldsymbol{x_i}$ 是否需要更新参数的复杂度

- 非对偶：$y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \le 0$，$\Theta(n)$ 
- 对偶：$y_i(\eta \sum_{j=1}^m \alpha_j y_j g_{ji} + \eta \sum_{i=j}^m \alpha_j y_j) \le 0$，$\Theta(m^2 + m) = \Theta(m^2)$

故只有 $m^2 < n$ 才起作用，即数据集大小远小于特征空间的维度时。

下面给出如何一次性计算多个 $f(\boldsymbol{x_i})$

$$
\begin{aligned}
    f(\boldsymbol{x_i}) &= \mathrm{sign}(\eta \sum_{j=1}^m \alpha_j y_j g_{ji} + \eta \sum_{i=j}^m \alpha_j y_j) \\
    &= \mathrm{sign}(\eta (\boldsymbol{\alpha} \odot \boldsymbol{y})^\mathrm{T} \boldsymbol{g_i} + \eta (\boldsymbol{\alpha} \odot \boldsymbol{y})^\mathrm{T} \boldsymbol{1}_{m \times 1})\\
    &= \mathrm{sign}(\eta (\boldsymbol{\alpha} \odot \boldsymbol{y})^\mathrm{T} (\boldsymbol{g_i} + \boldsymbol{1}_{m \times 1}))
\end{aligned}
$$

故

$$
[f(\boldsymbol{x_1}), f(\boldsymbol{x_2}), \dots, f(\boldsymbol{x_m})] = \mathrm{sign}(\eta (\boldsymbol{\alpha} \odot \boldsymbol{y})^\mathrm{T}(\mathbf{G} + \mathbf{1}_{m \times m}))
$$

## 参考资料

- 《统计学习方法》
- [知乎 - 如何理解感知机学习算法的对偶形式？](https://www.zhihu.com/question/26526858)
- [2. 感知机(Perceptron)基本形式和对偶形式实现](https://www.cnblogs.com/huangyc/p/10294583.html)
