# 朴素贝叶斯

朴素贝叶斯（naive Bayes）法是基于贝叶斯定理与特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的联合概率分布；然后基于此模型，对给定的输入 $\boldsymbol{x}$，利用贝叶斯定理求出后验概率最大的输出 $y$。

输入：$\mathbb{D} = \{(\boldsymbol{x_1}, y_1), \dots, (\boldsymbol{x_m}, y_m)\}, y_i \in \{c_1, \dots, c_k\}$

## 概念

- 先验概率：$P(Y=c_k)$

为什么先验概率是 $P(Y=c_k)$，而不是 $P(\boldsymbol{X} = \boldsymbol{x})$

## 参考资料

- [Building a Naive Bayes classifier from scratch with NumPy](https://geoffruddock.com/naive-bayes-from-scratch-with-numpy/)
