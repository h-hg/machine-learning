# 求导

||标量 $f$|向量 $\boldsymbol{f}_{n}$|矩阵 $\mathbf{F}_{m,n}$|
|:-:|:-:|:-:|:-:|
|标量 $x$|$\cfrac{\mathrm{d} f}{\mathrm{d} x}$|$\cfrac{\partial \boldsymbol{f}}{\partial x}$|$\cfrac{\partial \mathbf{F}}{\partial x}$|
|向量 $\boldsymbol{x}$|$\cfrac{\partial f}{\partial \boldsymbol{x}}$|$\cfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}$|$\cfrac{\partial \mathbf{F}}{\partial \boldsymbol{x}}$|
|矩阵 $\mathbf{X}$|$\cfrac{\partial f}{\partial \mathbf{X}}$|$\cfrac{\partial \boldsymbol{f}}{\partial \mathbf{X}}$|$\cfrac{\partial \mathbf{F}}{\partial \boldsymbol{X}}$|

## 标量对标量求导

$\cfrac{\mathrm{d} f}{\mathrm{d} x} = f'(x)$

$\mathrm{d} f = f'(x) \mathrm{d} x = \cfrac{\mathrm{d} f}{\mathrm{d} x} \mathrm{d} x$

## 标量对向量求导

定义：

$$
\cfrac{\partial f}{\partial \boldsymbol{x}} = (\cfrac{\partial f}{\partial x_i}, \dots, \cfrac{\partial f}{\partial x_n})^\mathrm{T} =
\begin{bmatrix}
    \cfrac{\partial f}{\partial x_i} \\
    \vdots \\
    \cfrac{\partial f}{\partial x_n} \\
\end{bmatrix}
$$

从一个简单的例子说明：$f = \boldsymbol{x}^\mathrm{T} \cdot \boldsymbol{y} + \boldsymbol{y}^\mathrm{T} \cdot \boldsymbol{x}$

根据定义，对第 $i$ 个分量 $x_i$ 进行求导

$$
\cfrac{\partial f}{\partial x_i} = \cfrac{\partial \sum_{j=1}^n x_j y_j}{\partial x_i} = \cfrac{\partial (x_i y_i)}{\partial x_i} = y_i
$$

故 $\cfrac{\partial f}{\partial \boldsymbol{x}} = \boldsymbol{y}$

由多元微分的知识，可知：$\mathrm{d} f = \sum_{i=1}^n \cfrac{\partial f}{\partial x_i}$

记

$$
\cfrac{\partial f}{\partial \boldsymbol{x}} = (\cfrac{\partial f}{\partial x_1}, \dots, \cfrac{\partial f}{\partial x_n})^\mathrm{T} \\
\mathrm{d} \boldsymbol{x} = (\mathrm{d} x_1, \dots, \mathrm{d} x_n)^\mathrm{T}
$$

则 $\mathrm{d} f = \sum_{i=1}^n \cfrac{\partial f}{\partial x_i} = \cfrac{\partial f}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}$。

具体如何求，可以参考下面标量对矩阵求导

## 标量对矩阵求导

定义：

$$
\cfrac{\partial f}{\partial \mathbf{X}} =
\begin{bmatrix}
    \cfrac{\partial f}{\partial x_{11}} & \cfrac{\partial f}{\partial x_{12}} & \dots  & \cfrac{\partial f}{\partial x_{1n}} \\
    \cfrac{\partial f}{\partial x_{21}} & \cfrac{\partial f}{\partial x_{22}} & \dots  & \cfrac{\partial f}{\partial x_{2n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \cfrac{\partial f}{\partial x_{n1}} & \cfrac{\partial f}{\partial x_{n2}} & \dots  & \cfrac{\partial f}{\partial x_{nn}} \\
\end{bmatrix}
$$

把上面多元微积分中的梯度与微分之间的联系扩展到矩阵，由此定义

$$
\mathrm{d} f = \sum_{i=1}^m \sum_{j=1}^n \cfrac{\partial f}{\partial x_{ij}} \mathrm{d} x_{ij}
$$

记
$$
\mathrm{d} \mathbf{X} =
\begin{bmatrix}
    \mathrm{d} x_{11} & \mathrm{d} x_{12} & \dots & \mathrm{d} x_{1n} \\
    \mathrm{d} x_{21} & \mathrm{d} x_{22} & \dots & \mathrm{d} x_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \mathrm{d} x_{n1} & \mathrm{d} x_{n2} & \dots &\mathrm{d} x_{nn} \\
\end{bmatrix}
$$

即 $\mathrm{d} \mathbf{X}$ 是矩阵 $\mathbf{X}$各元素微分构成的矩阵。

由 trace 的性质，可得

$$
\sum_{i=1}^m \sum_{j=1}^n \cfrac{\partial f}{\partial x_{ij}} \mathrm{d} x_{ij} = \mathrm{tr}(\cfrac{\partial f}{\partial \mathbf{X}}^\mathrm{T} \mathrm{d} \mathbf{X})
$$

即

$$
\mathrm{d} f = \mathrm{tr}(\cfrac{\partial f}{\partial \mathbf{X}}^\mathrm{T} \mathrm{d} \mathbf{X})
$$

常用的矩阵微分运算法则

1. 加减法：$\mathrm{d} (\mathbf{X} \pm \mathbf{Y}) = \mathrm{d} \mathbf{X} \pm \mathrm{d} \mathbf{Y}$
2. 矩阵乘法：$\mathrm{d}(\mathbf{X} \mathbf{Y}) = (\mathrm{d} \mathbf{X})\mathbf{Y} + \mathbf{X} \mathrm{d} \mathbf{Y}$
3. 转置：$\mathrm{d} (\mathbf{X}^\mathrm{T}) = (\mathrm{d}\mathbf{X})^\mathrm{T}$
4. 迹：$\mathrm{d}\mathrm{tr}(\mathbf{X}) = \mathrm{tr}(\mathrm{d} \mathbf{X})$
5. 逐元素乘法：$\mathrm{d} (\mathbf{X} \odot \mathbf{Y}) = \mathrm{d} \mathbf{X} \odot \mathbf{Y} + \mathbf{X} \odot \mathrm{d} \mathbf{Y}$
6. 逆：$\mathrm{d} \mathbf{X}^{-1} = -\mathbf{X}^{-1}\mathrm{d}\mathbf{X}\mathbf{X}^{-1}$
7. 行列式：$\mathrm{d} |\mathbf{X}| = \mathrm{tr}(\mathbf{X}^* \mathrm{d} \mathbf{X})$
8. 逐元素函数：$\mathrm{d} \delta(\mathbf{X}) = \delta'(\mathbf{X}) \odot \mathrm{d} \mathbf{X}$

如何计算：

1. 根据给定的 $f$ 寻找 $\mathrm{d} f$
2. 给 $\mathrm{d} f$ 套上 $\mathrm{tr}$。等号左边 $\mathrm{d} f$ 是个标量，所以不受影响，等号右边可以根据迹的技巧进行化简
3. 等号右边化简之后先找到 $\mathrm{d} X$，根据到导数与微分的连续 $\mathrm{d} f = \mathrm{tr}(\cfrac{\partial f}{\partial \mathbf{X}}^\mathrm{T} \mathrm{d} \mathbf{X})$ 得到 $\cfrac{\partial f}{\partial \mathbf{X}}$

特别地，若矩阵退化为向量，对照导数与微分的联系 $\mathrm{d} f = \cfrac{\partial f}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}$，即能得到导数。与上面的步骤相比，它少去了步骤二。

例子：

**eg 1.** $f = \boldsymbol{a}^\mathrm{T} \mathbf{X} \boldsymbol{b}$，求 $\cfrac{\partial f}{\partial \mathbf{X}}$，其中 $\boldsymbol{a}_{m \times 1}, \mathbf{X}_{m \times n}, \boldsymbol{b}_{n \times 1}$

**step** 1. 先求 $\mathrm{d} f$

根据矩阵乘法微分可得

$$
\mathrm{d} f
= \mathrm{d}(\boldsymbol{a}^\mathrm{T} \mathbf{X} \boldsymbol{b})
= \mathrm{d} \boldsymbol{a}^\mathrm{T} \mathbf{X} \boldsymbol{b}+\boldsymbol{a}^\mathrm{T} \mathrm{d} \mathbf{X} \boldsymbol{b}+\boldsymbol{a}^\mathrm{T}\mathbf{X} \mathrm{d} \boldsymbol{b}
= \boldsymbol{a}^\mathrm{T} \mathrm{d}\mathbf{X} \boldsymbol{b}
$$

由于 $\boldsymbol{a}, \boldsymbol{b}$ 是常量，$\mathrm{d} \boldsymbol{a} = \boldsymbol{0}, \mathrm{d} \boldsymbol{b} = \boldsymbol{0}$，即

$$
\mathrm{d} f  = \boldsymbol{a}^\mathrm{T} \mathrm{d} \mathbf{X}\boldsymbol{b}
$$

**step 2.** 给 $\mathrm{d} f$ 套上 $\mathrm{tr}$

$$
\mathrm{d} f = \mathrm{tr}(\mathrm{d} f) = \mathrm{tr}(\boldsymbol{a}^\mathrm{T} \mathrm{d} \mathbf{X}\boldsymbol{b})
$$

**step 3.** 进行化简

$$
\mathrm{tr}(\boldsymbol{a}^\mathrm{T} \mathrm{d} \mathbf{X} \boldsymbol{b})
= \mathrm{tr}(\boldsymbol{b} \boldsymbol{a}^\mathrm{T} \mathrm{d} \mathbf{X})
= \mathrm{tr}((\boldsymbol{a} \boldsymbol{b}^\mathrm{T})^\mathrm{T} \mathrm{d} \mathbf{X})
$$

由 $\mathrm{d} f = \mathrm{tr}(\cfrac{\partial f}{\partial \mathbf{X}_{ij}}^\mathrm{T} \mathrm{d} \mathbf{X}_{ij})$，得 $\cfrac{\partial f}{\partial \mathbf{X}} = \boldsymbol{a} \boldsymbol{b}^\mathrm{T}$

**eg** 2. $f = \boldsymbol{a}^\mathrm{T}\exp(\mathbf{X}\boldsymbol{b})$，求 $\cfrac{\partial f}{\partial \mathbf{X}}$，其中 $\boldsymbol{a}_{m \times 1}, \mathbf{X}_{m \times n}, \boldsymbol{b}_{n \times 1}$

**step 1.** 先求 $\mathrm{d} f$

$$
\mathrm{d} f = \boldsymbol{a}^\mathrm{T}(\exp(X\boldsymbol{b})\odot (\mathrm{d} \boldsymbol{X} \boldsymbol{b}))
$$

**step 2.** 给 $\mathrm{d} f$ 套上 $\mathrm{tr}$

$$
\mathrm{d} f
= \mathrm{tr}(\mathrm{d} f)
= \mathrm{tr}(\boldsymbol{a}^\mathrm{T}(\exp(X\boldsymbol{b})\odot (\mathrm{d} \boldsymbol{X} \boldsymbol{b})))
$$

**step 3.** 进行化简

$$
\begin{aligned}
    \mathrm{d} f
    &= \mathrm{tr}(\boldsymbol{a}^\mathrm{T}(\exp(\boldsymbol{X}\boldsymbol{b})\odot (\mathrm{d} \boldsymbol{\boldsymbol{X}} \boldsymbol{b}))) \\
    &= \mathrm{tr}((\boldsymbol{a} \odot \exp(\boldsymbol{X}\boldsymbol{b}))^\mathrm{T}\mathrm{d}\boldsymbol{X} \boldsymbol{b} \\
    &= \mathrm{tr}(\boldsymbol{b}(\boldsymbol{a}\odot \exp(\boldsymbol{X}\boldsymbol{b}))^\mathrm{T}\mathrm{d}\boldsymbol{X}) \\
    &= \mathrm{tr}(((\boldsymbol{a}\odot \exp(\boldsymbol{X}\boldsymbol{b}))\boldsymbol{b}^\mathrm{T})^\mathrm{T}\mathrm{d}\boldsymbol{X})
\end{aligned}
$$

故 $\cfrac{\partial f}{\partial \mathbf{X}} = (\boldsymbol{a}\odot \exp(\boldsymbol{X}\boldsymbol{b}))\boldsymbol{b}^\mathrm{T}$

**eg 3.** 最小二乘法：$f = (\mathbf{X}\boldsymbol{w}-\boldsymbol{y})^2$，求 $\cfrac{\partial f}{\partial \boldsymbol{w}}$，其中 $\boldsymbol{y}_{m \times 1}, \mathbf{X}_{m \times n}, \boldsymbol{w}_{n \times 1}$

$$
(\mathbf{X}\boldsymbol{w}-\boldsymbol{y})^2 = (\mathbf{X}\boldsymbol{w}-\boldsymbol{y})^\mathrm{T}(\mathbf{X}\boldsymbol{w}-\boldsymbol{y}) \\
$$

$$
\mathrm{d} f = \mathrm{tr}(\mathbf{X}\mathrm{d} \boldsymbol{w})^T(\mathbf{X}\boldsymbol{w}- \boldsymbol{y})+ (\mathbf{X}\boldsymbol{w}- \boldsymbol{y})^T(\mathbf{X}\boldsymbol{dw})
$$

## 向量对标量求导

$\boldsymbol{f}$ 对 $x$ 求导，定义如下

$$
\cfrac{\partial \boldsymbol{f}}{\partial x} = (\cfrac{\partial f_1}{\partial x}, \cfrac{\partial f_2}{\partial x}, \dots, \cfrac{\partial f_n}{\partial x})^\mathrm{T}
$$

## 向量对向量求导

这好像跟雅可比矩阵不一样。

这是矩阵对矩阵求导的特例，在此特例下不用矩阵对矩阵求导的方法，计算更快。

$\boldsymbol{f}_n$ 对 $\boldsymbol{x}_m$ 求导，定义如下

$$
\cfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} =
(
\cfrac{\partial f_1}{\partial \boldsymbol{x}},
\cfrac{\partial f_2}{\partial \boldsymbol{x}},
\ldots ,
\cfrac{\partial f_n}{\partial \boldsymbol{x}}
)
= 
\begin{bmatrix}
    \cfrac{\partial f_1}{\partial x_1} &\cfrac{\partial f_2}{\partial x_1}  &\cdots &\cfrac{\partial f_n}{\partial x_1}  \\
    \cfrac{\partial f_1}{\partial x_2} &\cfrac{\partial f_2}{\partial x_2}  &\cdots &\cfrac{\partial f_n}{\partial x_2} \\
    \vdots& \vdots & \ddots & \vdots \\
    \cfrac{\partial f_1}{\partial x_m} &\cfrac{\partial f_2}{\partial x_m}  &\cdots &\cfrac{\partial f_n}{\partial x_m}
\end{bmatrix}
_{m \times n}
$$

由一元微分学，可知 $\mathrm{d} f_i = \cfrac{\partial f_i}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}$，由此定义

$$
\mathrm{d} \boldsymbol{f} = (\mathrm{d} f_1, \mathrm{d} f_2, \dots, \mathrm{d} f_n)^\mathrm{T}
$$

那么

$$
\mathrm{d} \boldsymbol{f}
=(\cfrac{\partial f_1}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}, \cfrac{\partial f_2}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}, \dots, \cfrac{\partial f_n}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x})^\mathrm{T}
= \cfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}^\mathrm{T} \mathrm{d} \boldsymbol{x}
$$

至此便可以使用上面标量对矩阵求导的方法对向量对向量求导进行求解

**eg 1.** $\boldsymbol{f} = \mathbf{A} \boldsymbol{x}$，求 $\cfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}$

$$
\mathrm{d} \boldsymbol{f} = \mathbf{A} \mathrm{d} \boldsymbol{x}
$$

$$
\cfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} = \mathbf{A}^\mathrm{T}
$$

## 向量对矩阵求导

参考矩阵对矩阵求导

## 矩阵对标量求导

$$
\cfrac{\partial \mathbf{F}}{\partial x} =
\begin{bmatrix}
    \cfrac{\partial f_{11}}{\partial x} & \cfrac{\partial f_{12}}{\partial x} & \dots  & \cfrac{\partial f_{1n}}{\partial x} \\
    \cfrac{\partial f_{21}}{\partial x} & \cfrac{\partial f_{22}}{\partial x} & \dots  & \cfrac{\partial f_{2n}}{\partial x} \\
    \vdots & \vdots & \ddots & \vdots \\
    \cfrac{\partial f_{m1}}{\partial x} & \cfrac{\partial f_{m2}}{\partial x} & \dots  & \cfrac{\partial f_{mn}}{\partial x} \\
\end{bmatrix}
_{m \times n}
$$

## 矩阵对向量求导

参考矩阵对矩阵求导

## 矩阵对矩阵求导

首先引入向量化，定义

$$
\mathbf{X}_{mn} = (\boldsymbol{x_1}, \dots, \boldsymbol{x_n}) \\
\mathrm{vec}(\mathbf{X}_{mn}) = (\boldsymbol{x_1}^\mathrm{T}, \dots, \boldsymbol{x_n}^\mathrm{T}) = (x_{11}, \dots, x_{m1}, x_{12}, \dots, x_{m2}, \dots, x_{1n}, \dots, x_{mn})^\mathrm{T}
$$

向量化的性质

1. 线性：$\mathrm{vec}(\mathbf{A} + \mathrm{B}) = \mathrm{vec}(\mathrm{A}) + \mathrm{vec}(\mathrm{B})$
2. 矩阵乘法：$\mathrm{vec}(\mathbf{A} \mathbf{X} \mathbf{B}) = (\mathbf{B}^\mathrm{T} \otimes \mathbf{A}) \mathrm{vec}(\mathbf{X})$
3. 转置：$\mathrm{vec}(\mathbf{A}^\mathrm{T}) = \mathbf{K}_{mn} \mathrm{vec}(\mathbf{A})$，其中 $\mathbf{A} \in \mathbb{R}^{mn}$，$\mathbf{K}_{mn} \in \mathbb{R}^{mn \times mn}$ 是交换矩阵（commutation matrix）
4. 逐元素乘法：$\mathbf{vec}(\mathbf{A} \odot \mathbf{X}) = \mathrm{diag}(\mathbf{A})\mathrm{vec}(\mathbf(X))$，其中 $\mathrm{diag}(\mathbf{A}) \in \mathbb{R}^{mn \times mn}$ 是用 $\mathbf{A}$ 的元素（按列优先）排成的对角阵。

$\mathbf{F}_{m,n}$ 对 $\mathbf{X}_{p,q}$ 求导

$$
\mathrm{vec}(\mathbf{F}_{pq}) = (f_{11}, \dots, f_{p1}, \dots, f_{1q}, \dots, f_{pq})^\mathrm{T} \\
\mathrm{vec}(\mathbf{X}_{mn}) = (x_{11}, \dots, x_{m1}, \dots, x_{1n}, \dots, x_{mn})^\mathrm{T} \\
$$

$$
\cfrac{\partial \mathbf{F}}{\partial \mathbf{X}} = \cfrac{\partial \mathrm{vec}(\mathbf{F})}{\partial \mathrm{vec}(\mathbf{X})} =
\begin{bmatrix}
    \cfrac{\partial f_{11}}{\partial x_{11}} & \dots & \cfrac{\partial f_{p1}}{\partial x_{11}} & \dots & \cfrac{\partial f_{1q}}{\partial x_{11}} & \dots & \cfrac{\partial f_{pq}}{\partial x_{11}} \\
    \vdots & \dots & \vdots & \dots & \vdots & \dots & \vdots \\
    \cfrac{\partial f_{11}}{\partial x_{m1}} & \dots & \cfrac{\partial f_{p1}}{\partial x_{m1}} & \dots & \cfrac{\partial f_{1q}}{\partial x_{m1}} & \dots & \cfrac{\partial f_{pq}}{\partial x_{m1}} \\
    \vdots & \dots & \vdots & \dots & \vdots & \dots & \vdots \\
    \cfrac{\partial f_{11}}{\partial x_{1n}} & \dots & \cfrac{\partial f_{p1}}{\partial x_{1n}} & \dots & \cfrac{\partial f_{1q}}{\partial x_{1n}} & \dots & \cfrac{\partial f_{pq}}{\partial x_{1n}} \\
    \vdots & \dots & \vdots & \dots & \vdots & \dots & \vdots \\
    \cfrac{\partial f_{11}}{\partial x_{mn}} & \dots & \cfrac{\partial f_{p1}}{\partial x_{mn}} & \dots & \cfrac{\partial f_{1q}}{\partial x_{mn}} & \dots & \cfrac{\partial f_{pq}}{\partial x_{mn}} \\
\end{bmatrix}
_{mn \times pq}
$$

上面的定义，可能跟前面的标量对矩阵的求导定义产生冲突

- 1
- 2

由 $\cfrac{\partial \mathbf{F}}{\partial \mathbf{X}} = \cfrac{\partial \mathrm{vec}(\mathbf{F})}{\partial \mathrm{vec}(\mathbf{X})}$，得

$$
\mathrm{vec}(\mathrm{d} \mathbf{F}) = \cfrac{\partial \mathbf{F}}{\partial \mathbf{X}}^\mathrm{T} \mathrm{vec}(\mathrm{d} \mathbf{X})
$$

微分法求矩阵对矩阵导数的基本思想，算法如下

1. 根据给定的 $\mathbf{F}$ 寻找 $\mathrm{d} \mathbf{F}$
2. 将 $\mathrm{d} \mathbf{F}$ 向量化为 $\mathrm{vec}(\mathrm{d} \mathbf{F})$，使用矩阵等价变形和向量化的技巧化简
3. 等号右边化简之后先找到 $\mathrm{d} \mathbf{X}$，根据 $\mathrm{vec}(\mathrm{d} \mathbf{F}) = \cfrac{\partial \mathbf{F}}{\partial \mathbf{X}}^\mathrm{T} \mathrm{vec}(\mathrm{d} \mathbf{X})$ 得到 $\cfrac{\partial \mathbf{F}}{\partial \mathbf{X}}$

**eg 1.** $\mathbf{F} = \mathbf{A}\mathbf{W}, \mathbf{W}_{mn}$，求 $\cfrac{\partial \mathbf{F}}{\partial \mathbf{W}}$

**step 1.** 求 $\mathrm{d} \mathbf{F}$

$$
\mathrm{d} \mathbf{F} = \mathbf{A} \mathrm{d} \mathbf{W}
$$

**step 2.** 将 $\mathrm{d} \mathbf{F}$ 向量化为 $\mathrm{vec}(\mathrm{d} \mathbf{F})$

$$
\mathrm{vec}(\mathrm{d} \mathbf{F}) = \mathrm{vec}(\mathbf{A} \mathrm{d} \mathbf{W}) = (\mathbf{I}_n \otimes \mathbf{A}) \mathrm{vec}(\mathrm{d} \mathbf{W})
$$

**step 3.** 求得 $\cfrac{\partial \mathbf{F}}{\partial \mathbf{W}}$

$$
\cfrac{\partial \mathbf{F}}{\partial \mathbf{W}} = \mathbf{I}_n \otimes \mathbf{A}^\mathrm{T}
$$

## 最后

如果不知道结果是否正确，可以到 [Matrix Calculus](http://www.matrixcalculus.org/) 进行验证

## 参考资料

- [矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)
- [矩阵求导术（下）](https://zhuanlan.zhihu.com/p/24863977)
- [机器学习中的矩阵向量求导(一) 求导定义与求导布局](https://www.cnblogs.com/pinard/p/10750718.html)
- [机器学习中的矩阵向量求导(二) 矩阵向量求导之定义法](https://www.cnblogs.com/pinard/p/10773942.html)
- [机器学习中的矩阵向量求导(三) 矩阵向量求导之微分法](https://www.cnblogs.com/pinard/p/10791506.html)
- [机器学习中的矩阵向量求导(四) 矩阵向量求导链式法则](https://www.cnblogs.com/pinard/p/10825264.html)
- [机器学习中的矩阵向量求导(五) 矩阵对矩阵的求导](https://www.cnblogs.com/pinard/p/10930902.html)
- [机器学习中的数学理论1：三步搞定矩阵求导](https://zhuanlan.zhihu.com/p/262751195)
- [Wiki - Matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus)
- 张贤达 - 《矩阵分析与应用》

$\int_1 \log x \, \mathrm{d} x$
