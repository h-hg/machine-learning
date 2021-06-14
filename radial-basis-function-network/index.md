# 径向基神经网络

## 模型

$$
f(\boldsymbol{x}) = \sum_{i=1}^n w_i \varphi(\|\boldsymbol{x} - \boldsymbol{c_i}\|) + b
$$

- $\boldsymbol{c_i}$：第 $i$ 个中心
- $n$：中心个数
- $\varphi$：径向基函数

## 径向基函数

$$
r_{ij} = \|\boldsymbol{x_i} - \boldsymbol{c_j} \|
$$

- [百度百科](https://baike.baidu.com/item/径向基函数/3687692)
- [wiki](https://en.wikipedia.org/wiki/Radial_basis_function)

### 高斯函数

英文：Gaussian

$$
\varphi(r) = \exp(-\cfrac{1}{2 \sigma^2}r^2)
$$

## 多二次函数

英文：multiquadric

## 逆二次函数

英文：inverse quadratic

## 逆多二次函数

英文：inverse multiquadric

## 中心

### K-means

## 矩阵计算

利用最小二乘法的思想

记

$$
\dot{x_{ij}} = \varphi(r_{ij}) = \varphi(\|\boldsymbol{x_i} - \boldsymbol{c_j} \|)
$$

$$
\dot{\boldsymbol{x_{i}}} = [1, \dot{x_{i1}}, \dots, \dot{x_{in}}]^\mathrm{T}
$$

$$
\dot{\mathbf{X}}
= [\dot{\boldsymbol{x_{1}}}, \dots, \dot{\boldsymbol{x_{m}}}]^\mathrm{T}
=
\begin{bmatrix}
    1 & \dot{x_{11}} & \dots & \dot{x_{1n}}\\
    1 & \vdots & \dots & \vdots\\
    1 & \dot{x_{m1}} & \dots & \dot{x_{mn}}
\end{bmatrix}
$$

$$
\dot{\boldsymbol{w}} = [b, w_1, \dots, w_n]^\mathrm{T}
$$

则

$$
L = \| \dot{\mathbf{X}} \dot{\boldsymbol{w}} - \boldsymbol{y} \|
$$

## 反向传播的实现

- [【机器学习】RBF神经网络原理与Python实现](https://blog.csdn.net/Luqiang_Shi/article/details/84450655)

## 参考资料

- [2. Radial-Basis Function Networks (RBFN)](https://zhuanlan.zhihu.com/p/63153823)