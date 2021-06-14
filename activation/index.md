# 激活函数

## Sigmoid

$$
\mathrm{sigmoid}(z) = \cfrac{1}{1 + e^{-z}}
$$

$$
\cfrac{\mathrm{d}\ \mathrm{sigmoid}(z)}{\mathrm{d} z} = \cfrac{e^{-z}}{(1 + e^{-z})^2} = \mathrm{sigmoid}(z)(1 - \mathrm{sigmoid}(z))
$$

## tanh

$$
\tanh(x) = \cfrac{\sinh(x)}{\cosh(x)} = \cfrac{\cfrac{e^x - e^x}{2}}{\cfrac{e^x + e^x}{2}} = \cfrac{1 - e^{-2x}}{1 + e^{-2x}}
$$

$$
\cfrac{\mathrm{d} \tanh(x)}{\mathrm{d} x} = \mathrm{sech}^2(x) = 1 - \tanh^2(x)
$$

## ReLU

$$
\mathrm{relu}(z) = \max\{0, z\}
$$

$$
\cfrac{\mathrm{d}\ \mathrm{relu}(z)}{\mathrm{d} z} =
\begin{cases}
    0 & z < 0 \\
    1 & z \ge 0
\end{cases}
$$

## Leaky ReLU

$$
\mathrm{leaky\_relu}(z) =
\begin{cases}
    z & z > 0 \\
    \alpha z & z \le 0
\end{cases}
$$

## ELU

$$
\mathrm{elu}(x) =
\begin{cases}
    x & x \ge 0 \\
    \alpha(e^x - 1) & x < 0
\end{cases}
$$

## Softmax

$$
\mathrm{softmax}(\boldsymbol{z}) = (\cfrac{e^{z_1}}{\sum_i^n e^{z_i}}, \dots, \cfrac{e^{z_n}}{\sum_i^n e^{z_i}})^\mathrm{T}=\cfrac{\exp(\boldsymbol{z})}{\boldsymbol{1}_{n \times 1}^\mathrm{T} \cdot \exp(\boldsymbol{z})}, \boldsymbol{1} \in \mathbb{R}^n
$$

记

$$
a_i = \cfrac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}} \\
\boldsymbol{a} = (a_1, \dots, a_n)^\mathrm{T}
$$

由链式法则可得

$$
\cfrac{\partial L}{\partial z_i} = \sum_{s=1}^n\cfrac{\partial L}{\partial a_s} \cfrac{\partial a_s}{\partial z_i}
$$

由

$$
\cfrac{\partial a_s}{\partial z_i} =
\begin{cases}
    \cfrac{e^{z_i}\sum_{j=1}e^{z_j} - e^{z_i}e^{z_i}}{(\sum_{j=1}e^{z_j})^2} = \cfrac{e^z_i}{\sum_{j=1}e^{z_j}} (1 - \cfrac{e^z_i}{\sum_{j=1}e^{z_j}}) = a_i (1 - a_i) & s = i \\
    \cfrac{e^{z_s}e^{z_i}}{(\sum_{j=1}e^{z_j})^2} = \cfrac{e^{z_s}}{\sum_{j=1}e^{z_j}}  \cfrac{e^{z_i}}{\sum_{j=1}e^{z_j}} = - a_s a_i& s \neq i
\end{cases}
$$

得

$$
\cfrac{\partial L}{\partial z_i} = \sum_{s=1}^n\cfrac{\partial L}{\partial a_s} \cfrac{\partial a_s}{\partial z_i} = \sum_{s=1 \& s \neq i}^n\cfrac{\partial L}{\partial a_s} (-a_sa_i) + \cfrac{\partial L}{\partial a_s} a_i(1 - a_i)
$$

Tensorflow

```python

```

softmax + cross_entropy 的一起求导

$$
l(\boldsymbol{y}, \boldsymbol{\hat{y}}) = -\boldsymbol{y}^\mathrm{T} \ln \cfrac{\exp(\boldsymbol{z})}{\boldsymbol{1}_{n \times 1} \exp(\boldsymbol{z})}
$$

由

$$
\begin{cases}
    \boldsymbol{\hat{y}}^\mathrm{T}\boldsymbol{1}_{n \times 1} = 1 \\
    \ln \cfrac{\boldsymbol{y}}{a} = \ln \boldsymbol{y} - \boldsymbol{1}_{n \times 1} \ln a
\end{cases}
$$

可得

$$
\begin{aligned}
    l &= - \boldsymbol{y}^\mathrm{T}(\boldsymbol{z} - \boldsymbol{1}_{n \times 1} \ln(\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z})) \\
    &= - \boldsymbol{y}^\mathrm{T} \boldsymbol{z} + \ln(\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z})
\end{aligned}
$$

$$
\begin{aligned}
    \mathrm{d} l &= - \boldsymbol{y}^\mathrm{T} \mathrm{d} \boldsymbol{z} + \cfrac{\mathrm{d} (\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z})}{\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z}} \\
    &= - \boldsymbol{y}^\mathrm{T} \mathrm{d} \boldsymbol{z} + \cfrac{\boldsymbol{1}_{n \times 1}^\mathrm{T} \mathrm{d} \boldsymbol{z}}{\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z}} \\
    &= (- \boldsymbol{y}^\mathrm{T} + \cfrac{\boldsymbol{1}_{n \times 1}^\mathrm{T}}{\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z}})\mathrm{d} \boldsymbol{z}
\end{aligned}
$$

故

$$
\cfrac{\partial l}{\partial \boldsymbol{z}} = -\boldsymbol{y} + \cfrac{\boldsymbol{1}_{n \times 1}}{\boldsymbol{1}_{n \times 1}^\mathrm{T} \boldsymbol{z}}
$$

## Maxout

$$
\mathrm{maxout}(\boldsymbol{x}) = max\{x_1, \dots, x_n\}
$$

## 参考资料

- [一文概览深度学习中的激活函数（入门篇）](https://zhuanlan.zhihu.com/p/98472075)
- [一文详解Softmax函数](https://zhuanlan.zhihu.com/p/105722023)
