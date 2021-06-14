# AdaBoost

步骤

**step 1.** 初始化训练数据的权重分布

$$
\boldsymbol{w}^{(1)} = [w_1^{(1)}, \dots, w_m^{(1)}]^\mathrm{T}, w_i^{(1)} = \cfrac{1}{m}
$$

**step 2.** 重复下面的步骤 $M$ 次直至得到 $M$ 个基本分类器

**step 2.1** 使用具有权值分布 $\boldsymbol{w_i}$ 的训练数据集进行学习，得到基本的分类器 $M_i(\boldsymbol{x})$

$$
M_i(\boldsymbol{x}) \colon \mathcal{X} \mapsto \{+1, -1\}
$$

**step 2.2.** 计算 $M_i(\boldsymbol{x})$ 在训练数据集上的分类误差率

$$
e_i = \sum_{j=1}^m P(M_i(\boldsymbol{x_j}) \neq y_j) = \sum_{j=1}^N w_j^{(i)} I(M_i(\boldsymbol{x_j}) \neq y_j)
$$

**step 2.3.** 计算 $M_i(\boldsymbol{x})$ 的系数

$$
\alpha_i = \cfrac{1}{2} \log \cfrac{1 - e_i}{e_i}
$$

**step 2.4.** 更新训练数据集权值分布

$$
w_j^{(i+1)} = \cfrac{w_j^{(i)} \exp(-\alpha y_j M_i(\boldsymbol{x_i}))}{\sum_{k=1}^m w_k^{(i)} \exp(-\alpha y_k M_i(\boldsymbol{x_k}))}
$$

**step 3.** 构建基本分类器的线性组合

$$
f(\boldsymbol{x}) = \mathrm{sign}(\sum_{i=1}^M \alpha_i M_i(\boldsymbol{x}))
$$

## 参考资料

- [Building an AdaBoost classifier from scratch in Python](https://geoffruddock.com/adaboost-from-scratch-in-python/)
- [Adaboost Algorithm using numpy in Python](https://anujkatiyal.com/blog/2017/10/24/ml-adaboost/)
- [5. AdaBoost Algorithm](https://www.nathanieldake.com/Machine_Learning/06-Ensemble_methods-05-AdaBoost.html)
- 《统计学习方法》 - 李航
- [This handout](https://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/AdaBoost.pdf)
