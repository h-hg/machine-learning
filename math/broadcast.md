# NumPy 数学表示

## 广播

这里写关于 numpy 的广播机制如何使用数学表达式进行表达，仅限于向量和矩阵

### 向量与标量

$$
\boldsymbol{x}_{n \times 1} + a \boldsymbol{1_{n \times 1}} =
\begin{bmatrix}
    x_{1} + a  \\
    x_{2} + a  \\
    \vdots  \\
    x_{n} + a
\end{bmatrix}
$$

### 矩阵与标量

$$
\mathbf{X}_{m \times n} + a \boldsymbol{1_{m \times n}} =
\begin{bmatrix}
    x_{11} + a & x_{12} + a  & \dots  & x_{1n} + a \\
    x_{21} + a & x_{22} + a  & \dots  & x_{2n} + a \\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} + a & x_{m2} + a  & \dots  & x_{mn} + a
\end{bmatrix}
$$

### 矩阵与向量

矩阵与向量按列相加

$$
\mathbf{X}_{m \times n} + \boldsymbol{a}_{m \times 1} \cdot \boldsymbol{1}_{n \times 1}^\mathrm{T} = 
\begin{bmatrix}
    x_{11} + a_1 & x_{12} + a_1 & \dots & x_{1n} + a_1\\
    x_{21} + a_2 & x_{22} + a_2 & \dots & x_{2n} + a_2\\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} + a_m & x_{m2} + a_m & \dots  & x_{mn} + a_m
\end{bmatrix}
$$

将 $+$ 换成 $\odot$ 则变成矩阵与向量按列逐元素乘法

矩阵与向量按行相加

$$
\mathbf{X}_{m \times n} + \boldsymbol{1}_{m \times 1} \cdot \boldsymbol{x}_{n \times 1} ^\mathrm{T} =
\begin{bmatrix}
    x_{11} + a_1 & x_{12} + a_2 & \dots & x_{1n} a_n\\
    x_{21} + a_1 & x_{22} + a_2 & \dots & x_{2n} a_n\\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} + a_1 & x_{m2} + a_2 & \dots & x_{mn} + a_n
\end{bmatrix}
$$

将 $+$ 换成 $\odot$ 则变成矩阵与向量按行逐元素乘法

## 求和

### 向量

$$
\boldsymbol{1}_{n \times 1}^\mathrm{T} \cdot \boldsymbol{x}_{n \times 1} = \sum_{i=1}^n x_i
$$

### 矩阵

按列求和

$$
\mathbf{X}_{m \times n} \cdot \boldsymbol{1}_{n \times 1} =
\begin{bmatrix}
    \sum_{i=1}^n x_{1j} \\
    \vdots \\
    \sum_{i=1}^n x_{mj} \\
\end{bmatrix}
$$

按行求和

$$
\boldsymbol{1}_{m \times 1}^\mathrm{T} \cdot \mathbf{X}_{m \times n} = (\sum_{i=1}^m x_{i1}, \dots, \sum_{j=1}^m x_{in})
$$

全部元素

$$
\boldsymbol{1}_{m \times 1}^\mathrm{T} \cdot \mathbf{X}_{m \times n} \cdot \boldsymbol{1}_{n \times 1} = \sum_{ij} x_{ij}
$$
