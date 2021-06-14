# Exponential Loss

$$
\mathrm{exponential\_loss} = \cfrac{1}{m} \exp(-y \hat{y_i}) \\
y_i \in \{+1, -1\}, \hat{y_i} \in \mathbb{R}
$$

- 常应用于 AdaBoost 的二分类

- 分类错误时：$l \in [1, +\infin)$
- 分类正确时：$l \in (0, 1)$
