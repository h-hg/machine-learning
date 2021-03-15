# 矩阵的基本运算

$$
\mathbf{X} =
\begin{bmatrix}
    x_{11} & x_{12}  & \dots  & x_{1n} \\
    x_{21} & x_{22}  & \dots  & x_{2n} \\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} & x_{m2}  & \dots  & x_{mn}
\end{bmatrix}
= [a_{ij}]_{mn}
$$

## 加减法

$$
\mathbf{A} + \mathbf{B} =
\begin{bmatrix}
    a_{11} + b_{11} & a_{12} + b_{12} & \dots  & a_{1n} + b_{1n} \\
    a_{21} + b_{21} & a_{22} + b_{22} & \dots  & a_{2n} + b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} + b_{m1} & a_{m2} + b_{m2} & \dots  & a_{mn} + b_{mn}
\end{bmatrix}
$$

## 数乘

$$
k\mathbf{A} =
\begin{bmatrix}
    ka_{11} & ka_{12} & \dots  & ka_{1n} \\
    ka_{21} & ka_{22} & \dots  & ka_{2n} \\
    \vdots & \vdots   & \ddots & \vdots \\
    ka_{m1} & ka_{m2} & \dots  & ka_{mn}
\end{bmatrix}
$$

1. $(\lambda \mu \mathbf{A}) = \lambda(\mu \mathbf{A})$
2. $(\lambda + \mu )\mathbf{A} = \lambda \mathbf{A} + \mu \mathbf{A}$
3. $\lambda (\mathbf{A}+\mathbf{B}) = \lambda \mathbf{A} + \lambda \mathbf{B}$

## 矩阵乘法

$$
\mathbf{C} = \mathbf{A} \mathbf{B} =
\begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1p} \\
    a_{21} & a_{22} & \dots  & a_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \dots  & a_{ip} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \dots  & a_{mp}
\end{bmatrix}
\begin{bmatrix}
    b_{11} & b_{12} & \dots  & b_{1j} & \dots & b_{1n} \\
    b_{21} & b_{22} & \dots  & b_{2j} & \dots  & b_{2n} \\
    \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
    b_{p1} & b_{p2} & \dots  & b_{pj} & \dots  & b_{pn}
\end{bmatrix} \\
c_{ij} = a_{i1}b_{1j}  + a_{i2}b_{2j} + \ldots + a_{ip}b_{pj} = \sum_{k=1}^p a_{ik}b_{kj}
$$

1. 结合律：$(\mathbf{A} \mathbf{B}) \mathbf{C} =\mathbf{A}(\mathbf{B} \mathbf{C})$
2. 数乘：$\lambda (\mathbf{A}\mathbf{B}) = (\lambda \mathbf{A})\mathbf{B} = \mathbf{A}(\lambda \mathbf{B})$
3. 分配律
   - $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A} \mathbf{C}$
   - $(\mathbf{B} + \mathbf{C})\mathbf{A} = \mathbf{B} \mathbf{A} + \mathbf{C} \mathbf{A}$

## 转置

1. $(\mathbf{A}^\mathrm{T})^=\mathbf{A}$
2. $(\mathbf{A} \pm \mathbf{B})^\mathrm{T}=\mathbf{A}^\mathrm{T} \pm \mathbf{B}^\mathrm{T}$
3. $(\lambda \mathbf{A})^\mathrm{T}=\lambda \mathbf{A}^\mathrm{T}$
4. $(\mathbf{A}\mathbf{B})^\mathrm{T}=\mathbf{B}^T\mathbf{A}^\mathrm{T}$

## 共轭

$\overline{\mathbf{A}} = [\overline{a_{ij}}]$

1. $\overline{(\mathbf{A} + \mathbf{B})} = (\overline{\mathbf{A}} + \overline{\mathbf{B}})$

## 共轭转置

$\mathbf{A}^\mathrm{H} = (\overline{\mathbf{A}})^\mathrm{H} = \overline{\mathbf{A}^\mathrm{H}}$

## 行列式

1. $|\mathbf{A}^\mathrm{T}|=|\mathbf{A}|$
2. $|\lambda \mathbf{A}|=\lambda ^n |\mathbf{A}|$
3. $|\mathbf{A}\mathbf{B}|=|\mathbf{A}||\mathbf{B}|$

## 逆矩阵

1. $\mathbf{A}^{-1}=\cfrac{1}{|\mathbf{A}|}\mathbf{A}^*$
2. $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$
3. $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
4. $(\mathbf{A}^\mathrm{T})^{-1} = (\mathbf{A}^{-1})^\mathrm{T}$
5. $(\overline{\mathbf{A}})^{-1} = \overline{\mathbf{A}^{-1}}$
6. $(\overline{\mathbf{A}})^\mathrm{H} = \overline{\mathbf{A}^\mathrm{H}}$

## 特征值

$$
\mathrm{A}\boldsymbol{x} = \lambda \boldsymbol{x}
$$

## 哈达马积（Hadamard product）

- 表示：逐元素乘积
- 符号：$\odot$ 或者 $\circ$

$$
(\mathbf{A} \odot \mathbf{B})_{ij} = (\mathbf{A} \circ \mathbf{B})_{ij} = a_{ij}b_{ij}
$$

1. $\mathbf{A} \odot \mathbf{B} = \mathbf{B} \odot \mathbf{A}$
2. $(\mathbf{A} \odot \mathbf{B}) \odot C = \mathbf{A} \odot (\mathbf{B} \odot \mathbf{C})$
3. $(\mathbf{A} + \mathbf{B}) \odot C = \mathbf{A} \odot \mathbf{C} + \mathbf{B} \odot \mathbf{C}$

## 海森矩阵

假设函数 $f \colon \mathbb{R}^n \mapsto \mathbb{R}$ 的输入是一个 $n$ 维向量 $\boldsymbol{x}=(x_{1}, x_{2}, ..., x_{n})^\mathrm{T}$，输出是标量。假定函数 $f$ 所有的二阶偏导数都存在，$f$ 的海森矩阵H是一个 $n$ 行 $n$ 列的矩阵：

$$
\mathrm{H} =
\begin{bmatrix}
    \cfrac{\partial^2 f}{\partial x_1^2} & \cfrac{\partial^2 f}{\partial x_1 \partial x_2} & \dots  & \cfrac{\partial^2 f}{\partial x_1 \partial x_n} \\
    \cfrac{\partial^2 f}{\partial x_2 \partial x_1} & \cfrac{\partial^2 f}{\partial x_2^2} & \dots  & \cfrac{\partial^2 f}{\partial x_2 \partial x_n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \cfrac{\partial^2 f}{\partial x_n \partial x_1} & \cfrac{\partial^2 f}{\partial x_n \partial x_2} & \dots  & \cfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

## 矩阵的迹

$$
\mathrm{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}
$$

1. 标量：$a = \mathrm{tr}(a)$
2. 方阵：$\mathrm{tr}(\mathbf{A}^\mathrm{T}) = \mathrm{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$
3. 线性：$\mathrm{tr}(\mathbf{A} \pm \mathbf{B}) = \mathrm{tr}(\mathbf{A}) \pm \mathrm{tr}(\mathbf{B}) = \sum_i (a_{ii} + b_{ii})$
4. 矩阵乘法：
    - $\mathrm{tr}(\mathbf{A} \mathbf{B}) = \mathrm{tr}(\mathbf{B} \mathbf{A}) = \sum_{i,j} a_{ij} b_{ji}$
    - $\mathrm{tr}(\mathbf{A}^\mathrm{T} \mathbf{B}) = \sum_{i,j} a_{ij} b_{ij}$
5. 矩阵乘法/逐元素乘法交换：$\mathrm{tr}(\mathbf{A}^\mathrm{T}(\mathbf{B} \odot \mathbf{C})) = \mathrm{tr}((\mathbf{A} \odot \mathbf{B})^\mathrm{T}\mathbf{C}) = \sum_{i,j} a_{ij} b_{ij} c_{ij}$

## 克罗内克积（Kronecker product）

$$
\mathbf{X}_{mn} \otimes \mathbf{Y}_{pq} =
\begin{bmatrix}
    x_{11}\mathbf{Y} & x_{12}\mathbf{Y} & \dots x_{1n}\mathbf{Y} \\
    x_{21}\mathbf{Y} & x_{22}\mathbf{Y} & \dots x_{2n}\mathbf{Y} \\
    \vdots & \vdots & \ldots & \vdots \\  
    x_{m1}\mathbf{Y} & x_{m2}\mathbf{Y} & \dots x_{mn}\mathbf{Y} \\
\end{bmatrix}
_{mp \times nq}
$$

1. $(\mathbf{A} \otimes \mathbf{B})^\mathrm{T} = \mathbf{A}^\mathrm{T} \otimes \mathbf{B}^\mathrm{T}$
2. 双线性结合律：
    - $\mathbf{A} \otimes (\mathbf{B} + \mathbf{C}) = \mathbf{A} \otimes \mathbf{B} + \mathbf{A} \otimes \mathbf{C}$
    - $(\mathbf{A} + \mathbf{B}) \otimes \mathbf{C} = \mathbf{A} \otimes \mathbf{C} + \mathbf{B} \otimes \mathbf{C}$
3. 混合乘积性质：$(\mathbf{A} \otimes \mathbf{B})(\mathbf{C} \otimes \mathbf{D}) = (\mathbf{A} \mathbf{C}) \otimes (\mathbf{B} \mathbf{D})$

## Gram 矩阵

列向量 Gram 矩阵

$\mathbf{A}_{m \times n} = (\boldsymbol{a_1}, \dots, \boldsymbol{a_n})$

$$
\begin{aligned}
    \mathbf{G} &= \mathbf{A}^\mathrm{T} \mathbf{A} \\
    &= \begin{bmatrix}
        \boldsymbol{a_1}^\mathrm{T} \\
        \boldsymbol{a_2}^\mathrm{T} \\
        \vdots \\
        \boldsymbol{a_n}^\mathrm{T} \\
    \end{bmatrix}
    [\boldsymbol{a_1}, \boldsymbol{a_2}, \cdots, \boldsymbol{a_n}] \\
    &= \begin{bmatrix}
        \boldsymbol{a_1}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_1}^\mathrm{T} \boldsymbol{a_n} \\
        \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_n} \\
        \vdots & \vdots & \cdots & \vdots \\
        \boldsymbol{a_n}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_n}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_n}^\mathrm{T} \boldsymbol{a_n} \\
    \end{bmatrix}_{n \times n}
\end{aligned}
$$

特点：元素 $g_{ij}$ 是矩阵的列向量 $\boldsymbol{a_i}$ 和列向量 $\boldsymbol{a_j}$ 的乘积

行向量 Gram 矩阵

$\mathbf{A}_{m \times n} = (\boldsymbol{a_1}, \dots, \boldsymbol{a_m})^\mathrm{T}$

$$
\begin{aligned}
    \mathbf{G} &= \mathbf{A} \mathbf{A}^\mathrm{T} \\
    &= \begin{bmatrix}
        \boldsymbol{a_1}^\mathrm{T} \\
        \boldsymbol{a_2}^\mathrm{T} \\
        \vdots \\
        \boldsymbol{a_n}^\mathrm{T} \\
    \end{bmatrix}
    [\boldsymbol{a_1}, \boldsymbol{a_2}, \cdots, \boldsymbol{a_n}] \\
    &= \begin{bmatrix}
        \boldsymbol{a_1}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_1}^\mathrm{T} \boldsymbol{a_m} \\
        \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_2}^\mathrm{T} \boldsymbol{a_m} \\
        \vdots & \vdots & \cdots & \vdots \\
        \boldsymbol{a_m}^\mathrm{T} \boldsymbol{a_1} & \boldsymbol{a_m}^\mathrm{T} \boldsymbol{a_2} & \cdots & \boldsymbol{a_m}^\mathrm{T} \boldsymbol{a_m} \\
    \end{bmatrix}_{m \times m}
\end{aligned}
$$

特点：元素 $g_{ij}$ 是矩阵的行向量 $\boldsymbol{a_i}$ 和行向量 $\boldsymbol{a_j}$ 的乘积