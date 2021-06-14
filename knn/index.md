# K 近邻法

输入：$\mathbb{D} = \{(\boldsymbol{x_1}, y_1), \dots, (\boldsymbol{x_m}, y_m)\}, y_i \in \{c_1, \dots, c_k\}$

模型：

$$
y = \argmax_{c_j} \sum_{\boldsymbol{x_i} \in \{\boldsymbol{x} \mid \boldsymbol{x} \text{ is one of k-nearest neighbors}\}} \mathrm{I}(y_i = c_j)
$$

KD tree to be continued.
