# 交叉熵损失函数

$$
\mathrm{cross\_entropy}(\boldsymbol{y}, \boldsymbol{\hat{y}}) = - \sum_{i=1}^n y_i \ln \hat{y_i} = - \boldsymbol{y}^\mathrm{T} \ln \boldsymbol{\hat{y}}
$$

## 推导

下面从两个角度推导这个公式。

样本 $\mathbb{D} = {(\boldsymbol{x_1}, y_1), \dots, (\boldsymbol{x_m}, y_n)}$ 独立同分步，模型学到的为 $P(y|\boldsymbol{x})$，即输入是 $\boldsymbol{x}$，得到 $y$ 的概率为 $P(y|\boldsymbol{x})$。

### 最大似然估计

由独立同分布，可得到，在机器学到的模型下，联合分布如下

$$
P(y_1, \dots, y_n | \boldsymbol{x_1}, \dots, \boldsymbol{x_m}) = \prod_{i=1}^n P(y_i|\boldsymbol{x_i})
$$

使用最大似然估计法（Maximum Likelihood Estimation, MLE）来估计模型的参数 $\theta$

$$
\begin{aligned}
\boldsymbol{\theta} &= \argmax_{\boldsymbol{\theta}} \prod_{i=1}^n P(y_i | \boldsymbol{x_i}) \\
\iff \\
\boldsymbol{\theta} &= \argmax_{\boldsymbol{\theta}} \ln(\prod_{i=1}^n P(y_i|\boldsymbol{x_i})) \\
&= \argmax_{\boldsymbol{\theta}} \sum_{i=1}^n \ln P(y_i|\boldsymbol{x_i}) \\
\iff \\
\boldsymbol{\theta} &= - \cfrac{1}{m} \argmin_{\boldsymbol{\theta}} \sum_{i=1}^n \ln P(y_i|\boldsymbol{x_i})
\end{aligned}
$$

### 交叉熵

#### 信息论

在信息论中，信息（information）一般可以被表述为不确定性的程度，有如下特性

- 一定发生的事件没有信息
- 很有可能发生的事件几乎没有信息
- 随机事件拥有更多的信息
- 独立事件可以增加信息——抛两次正面的骰子的信息量大于抛一次正面骰子的信息量

事件 $x$ 的信息可以形式化为：

$$
\mathrm{I}(x) = - \ln(P(x))
$$

#### 熵

在信息论中，熵（entropy）用于衡量信息的多少，被定义为：

$$
\mathrm{H}(x) = - E[\ln P(x)]
$$

对于离散型，则为

$$
\mathrm{H}(x) = -\sum P(x) \ln P(x)
$$

### 相对熵

相对熵（KL Divergence）定义了两个两个分布 $P$、$Q$ 的距离，定义为

$$
D_{KL}(P || Q) = E(\ln \cfrac{P(x)}{Q(x)})
$$

对于离散型而言，

## 参考资料

- [手推softmax、交叉熵函数求导](https://zhuanlan.zhihu.com/p/60042105)
- [](https://jhui.github.io/2017/01/05/Deep-learning-Information-theory/)