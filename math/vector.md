# 向量

本笔记所指的向量默认是列向量。

$$
\boldsymbol{x} =
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{n}
\end{bmatrix}
= [x_{1}, x_{2}, ..., x_{n}]^\mathrm{T}
$$

我们将各元素均为实数的 $n$ 维向量 $\boldsymbol{x}$ 记作 $\boldsymbol{x} \in \mathbb{R}$ 或 $\boldsymbol{x} \in \mathbb{R}^{n \times 1}$。

1. 加减法：$\boldsymbol{x} \pm \boldsymbol{y} = \sum_i x_i - y_i$
2. 数量积（内积）：$\boldsymbol{x}^\mathrm{T} \cdot \boldsymbol{y} = \boldsymbol{x} \cdot \boldsymbol{y}^\mathrm{T} = \boldsymbol{x} \cdot \boldsymbol{y} = \sum_{i=1}^n x_i y_i$（注：$\cdot$ 可省略）
   1. 交换律：$\boldsymbol{x}^\mathrm{T} \boldsymbol{y} = \boldsymbol{y}^\mathrm{T} \boldsymbol{x}$
   2. 分配律：$(\boldsymbol{x} + \boldsymbol{y})^\mathrm{T}\boldsymbol{z} = \boldsymbol{x}^\mathrm{T}\boldsymbol{z} + \boldsymbol{y}^\mathrm{T}\boldsymbol{z}$
3. 向量积（外积）：$\boldsymbol{x} \times \boldsymbol{y}$
4. 混合积：$(\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{z}) = (\boldsymbol{x} \times \boldsymbol{y}) \cdot \boldsymbol{z}$