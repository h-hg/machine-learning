# 拉格朗日乘子法

## 等式约束

$$
\min_{x} f(\boldsymbol{x}) \\
\text{s.t.} g_i(\boldsymbol{x}) = 0, i = 1, 2, \dots, k
$$

拉格朗日函数

$$
L(\boldsymbol{x}, \boldsymbol{\alpha}) = f(\boldsymbol{x}) + \sum_{i=1}^k\alpha_i g_i(\boldsymbol{x}), \alpha_i \neq 0
$$

求解

$$
\nabla L(\boldsymbol{x}, \boldsymbol{\alpha}) = \boldsymbol{0}
$$

## 不等式约束

![例图](https://pic2.zhimg.com/80/v2-c10cb303f541b82b0883132f8938adcd_720w.jpg)

