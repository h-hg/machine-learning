标量对多向量链式求导，中间变量都是向量的情况：

$$
\frac{\partial z}{\partial \mathbf{x}}=\left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^{T} \frac{\partial z}{\partial \mathbf{y}} \\ \frac{\partial z}{\partial \mathbf{y}_{1}}=\left(\frac{\partial \mathbf{y}_{n}}{\partial \mathbf{y}_{n-1}} \frac{\partial \mathbf{y}_{n-1}}{\partial \mathbf{y}_{n-2}} \dots \frac{\partial \mathbf{y}_{2}}{\partial \mathbf{y}_{1}}\right)^{T} \frac{\partial z}{\partial \mathbf{y}_{n}} \\
$$

标量对多矩阵链式求导，中间变量都是矩阵的情况：

$$
z=f(Y), Y=A X+B \rightarrow \frac{\partial z}{\partial X}=A^{T} \frac{\partial z}{\partial Y}  \\  z=f(Y), Y=X A+B \rightarrow \frac{\partial z}{\partial X}=\frac{\partial z}{\partial Y} A^{T}\\
$$

来自 [神经网络反向传播矩阵求导](https://zhuanlan.zhihu.com/p/83859554)