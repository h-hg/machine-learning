# 线性回归

模型：$\hat{y_i} = f(\boldsymbol{x_i}) = \boldsymbol{x_i}^\mathrm{T}\boldsymbol{w} + b$，其中 $\boldsymbol{x_i}, \boldsymbol{w} \in \mathbb{R}^{n \times 1},$

损失函数：$L(\boldsymbol{w}, b) = \sum_{i=1}^m(\hat{y_i} - y_i)^2 = \sum_{i=1}^m({\boldsymbol{x_i}^\mathrm{T} \boldsymbol{w} + b - y_i})^2$

利用标量对向量直接求导，得到

$$
\left\{
    \begin{aligned}
        \cfrac{\partial L(\boldsymbol{w}, b)}{\partial \boldsymbol{w}} &= \sum_{i=1}^{m} 2(\boldsymbol{x_i}^\mathrm{T} \boldsymbol{w} + b - y_i) \boldsymbol{x_i} \\
        \cfrac{\partial L(\boldsymbol{w}, b)}{\partial b} &= \sum_{i=1}^{m} 2(\boldsymbol{x_i}^\mathrm{T} \boldsymbol{w} + b - y_i)
    \end{aligned}
\right.
$$

转化为矩阵的求导

记

$$
\mathbf{X} = (\boldsymbol{x_1}, \boldsymbol{x_2}, \dots, \boldsymbol{x_m})^\mathrm{T} \\
\boldsymbol{y} = (y_1, y_2, \dots, y_m)^\mathrm{T}
$$

则 $L(\boldsymbol{w}, b) = \|\mathbf{X}\boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}\| = (\mathbf{X}\boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T}(\mathbf{X}\boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})$

$$
\begin{aligned}
    \mathrm{d} L(\boldsymbol{w}, b)
    &= \mathrm{d}((\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T}) \cdot (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}) + (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \cdot \mathrm{d} (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}) \\
    &= 2 (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \cdot \mathrm{d} (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}) \\
    &= 2 (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \mathbf{X} \mathrm{d} \boldsymbol{w} + 2 (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \mathrm{d}(b\boldsymbol{1}_{m \times 1})\\
\end{aligned}
$$

$$
\begin{aligned}
    \mathrm{tr}(L(\boldsymbol{w}, b))
    &= \mathrm{tr}(2 (\mathbf{X} \boldsymbol{w} - \boldsymbol{y})^\mathrm{T} \mathbf{X} \mathrm{d} \boldsymbol{w}) + \mathrm{tr}(2 (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \mathrm{d}(b\boldsymbol{1}_{m \times 1}))\\
    &=2 \mathrm{tr}((\mathbf{X}^\mathrm{T}(\mathbf{X} \boldsymbol{w} - \boldsymbol{y}))^\mathrm{T} \mathrm{d} \boldsymbol{w}) + 2 \mathrm{tr}((\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y})^\mathrm{T} \mathrm{d}(b\boldsymbol{1}_{m \times 1}))
\end{aligned}
$$

$$
\cfrac{\partial L(\boldsymbol{w}, b)}{\partial \boldsymbol{w}} = 2 \mathbf{X}^\mathrm{T}(\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}) \\
\cfrac{\partial L(\boldsymbol{w}, b)}{\partial (b\boldsymbol{1}_{m \times 1})} = 2 (\mathbf{X} \boldsymbol{w} + b\boldsymbol{1}_{m \times 1} - \boldsymbol{y}) \\
$$

由

$$
\cfrac{\partial (b\boldsymbol{1}_{m \times 1})}{\partial b} = \boldsymbol{1} \in \mathbb{R}^m
$$

得

$$
\cfrac{\partial L(\boldsymbol{w}, b)}{\partial b} = \cfrac{\partial L(\boldsymbol{w}, b)}{\partial (b\boldsymbol{1}_{m \times 1})} \cfrac{\partial (b\boldsymbol{1}_{m \times 1})}{\partial b} = \sum_{i=1}^m \cfrac{\partial L(\boldsymbol{w}, b)}{\partial (b\boldsymbol{1}_{m \times 1})}_i
$$

第一个等号后面表示两个向量相乘，第二个等号后面表示求向量所有元素的和。

```python
def back_grad(self, X, y):
    z = self.forward(X)
    grad_w = np.mean((z - y) * X, axis=0).reshape((-1, 1))
    grad_b = np.mean((z - y), axis=0)
    return grad_w, grad_b
```

## 实现

### 数据处理

### numpy

[](np.py ':include :type=code python')

注：

1. 在使用 $\| \hat{\mathbf{X}} \boldsymbol{w} + b - \boldsymbol{y} \|$ 作为损失函数时出现了，数据溢出的现象（待探究）。
2. 如果 $\mathbf{X}$ 没有标准化（归一化处理），出现了奇怪的曲线。

### tensorflow

### pytorch

低级 API

[](torch-low.py ':include :type=code python')

中阶 API

[](torch-mid.py ':include :type=code python')

这里还遇到了一种更奇怪的实现，直接使用乘法，而不是矩阵乘法

[](torch-mid2.py ':include :type=code python')

## 最小二乘法法

转化为矩阵形式，记：

$$
\dot{\boldsymbol{x_i}}= (\boldsymbol{x_i}, 1) = (x_{i1}, \dots, x_{in}, 1)^\mathrm{T} \\
\dot{\boldsymbol{w}} = (\boldsymbol{w}, 1) = (w_1, \dots, w_n, b)^\mathrm{T} \\
\dot{\mathbf{X}} = (\dot{\boldsymbol{x_1}}, \dots, \dot{\boldsymbol{x_m}})^\mathrm{T} \\
$$

则 $f(\boldsymbol{x_i}) = \dot{\boldsymbol{x_i}}^\mathrm{T} \cdot \dot{\boldsymbol{w}}$，$L(\boldsymbol{w}, b) = L(\dot{\boldsymbol{w}})= \|\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}\|^2$

求导：

$$
L(\dot{\boldsymbol{w}}) = \|\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}\|^2 = (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T}(\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})
$$

$$
\begin{aligned}
    \mathrm{d} L(\dot{\boldsymbol{w}}, b)
    &= \mathrm{d}((\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T}) \cdot (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}) + (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T} \cdot \mathrm{d} (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}) \\
    &= 2 (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T} \cdot \mathrm{d} (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}) \\
    &= 2 (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T} \dot{\mathbf{X}} \mathrm{d} \dot{\boldsymbol{w}} \\
\end{aligned}
$$

$$
\begin{aligned}
    \mathrm{tr}(L(\dot{\boldsymbol{w}}, b))
    &= \mathrm{tr}(2 (\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})^\mathrm{T} \dot{\mathbf{X}} \mathrm{d} \dot{\boldsymbol{w}}) \\
    &=2 \mathrm{tr}((\dot{\mathbf{X}}^\mathrm{T}(\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y}))^\mathrm{T} \mathrm{d} \dot{\boldsymbol{w}})
\end{aligned}
$$

$$
\cfrac{\partial L(\dot{\boldsymbol{w}}, b)}{\partial \dot{\boldsymbol{w}}} = 2 \dot{\mathbf{X}}^\mathrm{T}(\dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y})
$$

令 $\cfrac{\partial L(\dot{\boldsymbol{w}}, b)}{\partial \dot{\boldsymbol{w}}} = \boldsymbol{0}$，得

$$
\dot{\mathbf{X}}^\mathrm{T}\dot{\mathbf{X}} \dot{\boldsymbol{w}} = \dot{\mathbf{X}}^\mathrm{T} \boldsymbol{y} \\
\dot{\boldsymbol{w}} = (\dot{\mathbf{X}}^\mathrm{T}\dot{\mathbf{X}})^{-1}\dot{\mathbf{X}}^\mathrm{T} \boldsymbol{y}
$$

这种直接求数学解析式求参数被成为解析解（Closed Form Solution），与之相对便是上面的迭代解

[](closed-form-solution.py ':include :type=code python')

## 广义线性模型

上面的模型是使用 $\boldsymbol{w_i}^\mathrm{T} \boldsymbol{x} + b$ 去逼近 $y_i$，如果 $y_i$ 呈现指数分布的话，我们可以使用 $e^{\boldsymbol{w_i}^\mathrm{T} \boldsymbol{x} + b}$ 去逼近 $y_i$，为了减少模型实现的修改，上述表达可转化用 $\boldsymbol{w_i}^\mathrm{T} \boldsymbol{x} + b$ 去逼近 $\ln y_i$。

上面的表达便是“对数线性回归”（log-linear regression）。

更为一般的，考虑单调函数 $g(\cdot)$，使用 $g(\boldsymbol{w_i}^\mathrm{T} \boldsymbol{x} + b)$ 去逼近 $y_i$（换言之，使用 $\boldsymbol{w_i}^\mathrm{T} \boldsymbol{x} + b$ 逼近 $g^{-1}(y_i)$） 便是“广义线性模型”（generalized linear model）

## 参考资料

- [使用numpy编写神经网络，完成boston房价预测问题](https://blog.csdn.net/KaelCui/article/details/105804164)
- [波士顿房价预测——线性模型（numpy实现）](https://blog.csdn.net/qq_36560894/article/details/104125289)
- [Let’s code a Neural Network in plain NumPy](https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)