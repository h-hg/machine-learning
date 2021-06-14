# 支持向量机

- 训练数据集：$\mathbb{D} = \{(\boldsymbol{x_1}, y_1), (\boldsymbol{x_m}, y_m)\}, y_i \in \{+1, -1\}$
- 基本原理：寻找一个超平面$S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 0$ 将正负样本分离开
- 模型：$f(\boldsymbol{x}) = \mathrm{sign}(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b)$

## 线性可分支持向量机

英文名：linear support vector machine in linearly separable case

首先给出什么是线性可分性，给定一个数据集 $\mathbb{D}={(\boldsymbol{x_1}, y_1), \dots, (\boldsymbol{x_m}, y_m)}, y_i \in \{+1, -1\}$，如果存在一个超平面 $S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 0$ 能够将数据集的正负类**完全**正确划分到超平面的两侧。即对所有 $y_i = -1$，有 $\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b < 0$，对所有 $y_i = +1$，有 $\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b > 0$。

线性可分SVM

当训练数据线性可分时，通过硬间隔(hard margin，什么是硬、软间隔下面会讲)最大化可以学习得到一个线性分类器，即硬间隔SVM，如上图的的H3。

### 几何间隔

一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度，离超平面远的点确信度高，离超平面近的点确信度低，即 $\cfrac{|\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b|}{\|\boldsymbol{w}\|}$ 表示确信度。

从感知机一节的知识中，可以得知 $\cfrac{y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b)}{\|\boldsymbol{w}\|}$ 的符号可以表示分类的正确与否，同时它的绝对值表示点与超平面的距离，即确信度。

定义样本点 $(\boldsymbol{x_i}, y_i)$ 的几何间隔

$$
\gamma_i = \cfrac{y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i} + b)}{\|\boldsymbol{w}\|}
$$

定义超平面 $S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b =0$ 关于训练数据集 $\mathbb{D}$ 的几何间隔

$$
\gamma = \min_{1,\dots, m} \cfrac{y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i} + b)}{\|\boldsymbol{w}\|}
$$

### 间隔最大化

对于线性可分的训练数据集来说，线性可分分离超平面有无穷多个，但是几何间隔最大的平面是唯一的，这里的间隔最大化又称为硬件间隔最大化。

间隔最大化的直观解释是：对训练数据集找到几何间隔最大的超平面意味着以充分大的确信度对训练数据进行分类。也就是说，不仅将正负样本点分开，且对最难分的点（离超平面最近的点）也有足够大的确信度将它们分开。

几何间隔最大的分离平面，即最大间隔分离超平面，可以表示为下面的约束问题

$$
\begin{aligned}
    &\max_{\boldsymbol{w}, b} \quad \gamma \\
    &\text{s.t.} \quad \cfrac{y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b)}{\|\boldsymbol{w}\|} \ge \gamma \\
\end{aligned} \tag{1}
$$

### 函数间隔

定义样本点 $(\boldsymbol{x_i}, y_i)$ 的函数间隔

$$
\hat{\gamma_i} = y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i} + b)
$$

定义超平面 $S: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b =0$ 关于训练数据集 $\mathbb{D}$ 的函数间隔

$$
\hat{\gamma} = \min_{1,\dots, m} y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x_i} + b)
$$

考虑几何间隔与函数间隔之间的关系，可将 $(1)$ 改写为

$$
\begin{aligned}
    &\max_{\boldsymbol{w}, b} \quad \cfrac{\hat{\gamma}}{\|\boldsymbol{w}\|} \\
    &\text{s.t.} \quad y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) \ge \hat{\gamma} \\
\end{aligned} \tag{2}
$$

由感知机一节中，可以得知函数间隔会随着参数的比例而改变，如将 $\boldsymbol{w}$ 和 $b$ 按比例改变为 $\lambda \boldsymbol{w}$ 和 $\lambda b$，此时，函数间隔由 $\hat{\gamma}$ 变成 $\lambda \hat{\gamma}$。由于比例 $\lambda$ 对优化问题的不等式没有产生影响，为了达到简化的目的，不妨取 $\hat{\gamma} = 1$，即 $\lambda$ 总是取那个使得 $\hat{\gamma} = 1$ 的值。$(2)$ 可改写为

$$
\begin{aligned}
    &\min_{\boldsymbol{w}, b} \quad \cfrac{1}{\|\boldsymbol{w}\|} \\
    &\text{s.t.} \quad y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) \ge 1 \\
\end{aligned} \tag{3}
$$

$(2)$ 等价于

$$
\begin{aligned}
    &\max_{\boldsymbol{w}, b} \quad \cfrac{1}{2} \|\boldsymbol{w}\|^2 \\
    &\text{s.t.} \quad y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) -1 \ge 0 \\
\end{aligned} \tag{3}
$$

## 支持向量

在线性可分的情况下，训练集的样本点与分离超平面距离最近的样本点的实力被称为支持向量（support vector）。支持向量是下面不等式等号成立的点

$$
\cfrac{y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b)}{\|\boldsymbol{w}\|} \ge \gamma
$$

如果从函数间隔的角度看待的话，即是下面不等式等号成立的点

$$
y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) \ge 1
$$

对于正类的点，支持向量在超平面

$$
H_1: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = 1
$$

对于负类的点，支持向量在超平面

$$
H_2: \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b = -1
$$

### 对偶算法

通过求解对偶问题（dual problem）得到原始问题（primal problem）的最优解，这样做的优点有如下：

1. 对偶问题往往更加容易求解
2. 自然引入核函数，进而推广到非线性分类问题

## 线性支持向量机

当训练数据不能线性可分但是可以近似线性可分时，通俗地讲就是训练数据有一些特异点（outlier），将这些特异点除去后，剩下大部分样本组成的几何是线性可分的。

线性不可分意味着某些样本点 $(\boldsymbol{x_i}, y_i)$ 不能满足函数间隔大于等于 1 的约束条件。为了解决这个问题，可以对每一个样本点 $(\boldsymbol{x_i}, y_i)$ 引进一个松弛变量（Slack variable） $\xi_i \ge 0$，使得函数间隔加上松弛变量大于等于 1。

$$
y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b) + \xi_i \ge 1
$$

即

$$
y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b) \ge 1 - \xi_i
$$

当然松弛变量越小越好，即对于每一个松弛变量都需要支付额外的代价，目标函数（损失函数）由原来的 $\cfrac{1}{2} \|\boldsymbol{w}\|^2$ 变成了

$$
\min_{\boldsymbol{w}, b} \cfrac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m \xi_i
$$

其中 $C > 0$ 称为惩罚参数，一般由应用问题决定，$C$ 值变大，对误分类的惩罚增加。

上面的目标函数包含两部分

- $\min_{\boldsymbol{w}, b} \cfrac{1}{2} \|\boldsymbol{w}\|^2$：表示间隔尽量大
- $\min_{\boldsymbol{w}, b} \sum_{i=1}^m \xi_i$：表示误分类的个数尽量小

$C$ 是调和这两个目标的系数。

这种引入松弛变量后最大化间隔的方法被称为软间隔最大化，其表示如下

$$
\begin{aligned}
    \min_{\boldsymbol{w}, b} \quad & \cfrac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m \xi_i \\
    \text{s.t.} \quad &  y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x} + b) \ge 1 - \xi_i, i = 1, \dots, m \\
    \quad & \xi_i \ge 0, i = 1, \dots, m
\end{aligned} \tag{4}
$$

### 对偶算法

### 合页损失

由 $(4)$ 可得 $\xi_i \ge 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b)$

由于要最小化 $\xi_i$，同时要让 $\xi_i \ge 0$，故

$$
\begin{aligned}
    \xi_i &=
    \begin{cases}
        1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \ge 0 \\
        0 & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) < 0
    \end{cases} \\
    &= \max \{0, 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \}
\end{aligned}
$$

由此，$(4)$ 可改写为

$$
\min_{\boldsymbol{w}, b} \quad \cfrac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{i=1}^m \max \{0, 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \}
$$

使用梯度下降进行求解

记：

$$
f = \max \{0, 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \}
$$

则

$$
\cfrac{\partial f}{\partial \boldsymbol{w}} =
\begin{cases}
    - y_i \boldsymbol{x_i} & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \ge 0 \\
    0 & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) < 0
\end{cases} \\
\cfrac{\partial f}{\partial b} =
\begin{cases}
    - y_i & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \ge 0 \\
    0 & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) < 0
\end{cases}
$$

故

$$
\cfrac{\partial L}{\partial \boldsymbol{w}} =
\begin{cases}
    \boldsymbol{w} + C \sum_{i=1}^m - y_i \boldsymbol{x_i} & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \ge 0 \\
    0 & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) < 0
\end{cases} \\
\cfrac{\partial L}{\partial b} =
\begin{cases}
    C \sum_{i=1}^m - y_i & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) \ge 0 \\
    0 & 1 - y_i(\boldsymbol{w}^\mathrm{T} \boldsymbol{x_i} + b) < 0
\end{cases}
$$

## 非线性支持向量机

非线性问题往往不好求解，所以希望用解线性分类问题的方法解决这个问题，所采用的方法是进行一个非线性变换，将非线性问题变换为线性问题，通过解变换后的线性问题的方法求解原来的非线性问题。

设 $\mathcal{X}$ 是输入空间（欧式空间 $\mathbb{R}^n$ 的子集或离散集合），又设 $\mathcal{H}$ 为特征空间（希尔伯特空间）。

$$
\exist \phi(\boldsymbol{x}) \colon \mathcal{X} \mapsto \mathcal{H} \\
\text{s.t.} \forall \boldsymbol{x}, \boldsymbol{z} \in \mathcal{X}, \exist K(\boldsymbol{x}, \boldsymbol{z}) = \phi(\boldsymbol{x})^\mathrm \phi(\boldsymbol{z})
$$

则称 $K(\boldsymbol{x}, \boldsymbol{z})$ 为核函数，$\phi(\boldsymbol{x})$ 为映射函数

|名称|表达式|参数|
|:-:|:-:|:-:|
|线性核|$K(\boldsymbol{x_i}, \boldsymbol{x_j}) = \boldsymbol{x_i}^\mathrm{T}\boldsymbol{x_j}$||
|多项式|$K(\boldsymbol{x_i}, \boldsymbol{x_j}) = (\boldsymbol{x_i}^\mathrm{T}\boldsymbol{x_j})^d$| $d \ge 1$ 为多项式的次数|
|高斯核|$K(\boldsymbol{x_i}, \boldsymbol{x_j}) = \exp(\cfrac{\|\boldsymbol{x_i} - \boldsymbol{x_j}\|^2}{2\sigma^2})$|$\sigma > 0$ 为高斯核的带宽（width）|
|拉普拉斯核|$K(\boldsymbol{x_i}, \boldsymbol{x_j}) = \exp(\cfrac{\|\boldsymbol{x_i} - \boldsymbol{x_j}\|^2}{\sigma})$|$\sigma > 0$|
|Sigmoid 核|$K(\boldsymbol{x_i}, \boldsymbol{x_j}) = \tanh(\beta \boldsymbol{x_i}^\mathrm{T}\boldsymbol{x_j} + \theta})$|$\beta > 0, \theta < 0$|

核矩阵（Kernal matrix）（也是 Gram 矩阵）

$k_{ij} = K(\boldsymbol{x_i}, \boldsymbol{x_j})$

## 参考资料

- 《统计学习方法》
- [看了这篇文章你还不懂SVM你就来打我](https://tangshusen.me/2018/10/27/SVM/)
