# PyTorch 梯度

在这一文章主要有这几部分内容

1. 简单说明一下 PyTorch 的自动求导机制
2. PyTorch 自动求导的使用

## Tensor 结构

这里使用 `numpy.ndarray` 来构建一个简化版的 PyTorch 来说明自动求导机制。

|属性|类型|解释|
|:-:|:-:|:-:|
|`values`|`numpy.ndarray`|存储 Tensor 的数据|
|`requires_grad`|`bool`|表明该 Tensor 是否需要求导|
|`grad`|`ndarray`|存储梯度值|
|`grad_fn`||梯度函数|
|`is_leaf`|`bool`|是否为叶子节点|

## 计算图

计算图在实现上有两种，一种是静态计算图，如 TensorFlow 1.x，另一种是动态计算图，如 TensorFlow 2.x 和 PyTorch。这里主要讲解动态计算图。

计算图在计算的流动方向上，可以分为前向计算图和反向计算图。对于简单的单层线性回归：$x w + b$，其构成前向传播图如下。在前向图中，流动的计算的中间结果（数值）。

![](https://pic1.zhimg.com/v2-5718765bd9bc83fc986d269df2e25530_r.jpg)

在梯度下降中，通过构建反向传播来计算梯度，其反向传播图如下所示。反向图中流动的是梯度值以及中间运算的反向操作（即梯度函数）。

![](https://pic1.zhimg.com/v2-5718765bd9bc83fc986d269df2e25530_r.jpg)

这两幅计算图高度相似，实际上可以将两者结合到一起，只定义其中一个，使用一些属性来存储节点之间的关系。例如可以使用如下属性

1. `grad`：存储节点的梯度
2. `dependency`：用来存储该节点的依赖节点，即图中下面连接节点。
3. `grad_fn`：用来存储该对依赖节点的梯度函数。

![https://pic1.zhimg.com/80/v2-ab45db1b7e2108dc08d214767f608860_720w.jpg](https://pic1.zhimg.com/80/v2-ab45db1b7e2108dc08d214767f608860_720w.jpg)

在有了这些属性后，如何获取这些属性呢？一个最简单的方法便是使用重载运算符（如 `+`、`-`、`/`、`*` 和 `@` 等），在每一次获得新的 Tensor 的时候顺便记录这些属性。

有了 `dependency` 后，实际上就是构建了计算图的图形结构，通过 `grad_fn` 便可从最上层的节点（一般是 loss 函数值）执行从顶到下的遍历（例如 DFS ）计算每一个节点的梯度值。 PyTorch 将这个函数封装成 `Torch.Tensor.backward`。

## 计算图的优化

上面的实现是对所有节点都存储了梯度值，这对于深度神经网络的存储带来了挑战，但是实际上不是所有节点都需要存储梯度值，只要模型参数的梯度值有被存储即可。模型参数一般都是叶子节点，即它不由其他 Tensor 构建而来，换句话说它不依赖于其他 Tensor。但是并不是所有叶子节点都必须存储梯度值，例如训练数据一般都是叶子节点，但是不需要存储其的梯度值。所有 PyTorch 在 `torch.Tensor` 引入两个属性来进行设置，当且仅当 `requires_grad` 和 `is_leaf` 两者都为 `True` 的时候才存储 Tensor 的 `grad`。一般来说将模型的参数设置 `requires_grad=True`，并且它一般是叶子节点。值的注意的是，若一个 Tensor 的依赖 Tensors 中有一个的 `requires_grad` 为 `True`，那么它的 `requires_grad` 也为 `True`，这是因为需要将它的 `grad` 反向传播给该依赖 Tensor，但它并不是叶子节点，所以最终它的 `grad` 不会被存储。

此外，维护 `dependency`、`grad_fn` 也需要消耗较大的存储，PyTorch 默认在调用 `torch.Tensor.backward` 后也对这些属性进行了移除，需要再次运行前向计算以再次生成计算图。

至此，可以对 PyTorch 的动态计算图做出如下归纳。

1. 计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。
2. 计算图在反向传播后立即销毁。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

## `torch.Tensor.backward`、`torch.autograd.grad`

PyTorch提供两种求梯度的方法：`torch.Tensor.backward` 和 `torch.autograd.grad`，两者区别在于前者是给叶子节点填充 `.grad` 字段，而后者是直接返回梯度。

```python
import torch
x = torch.tensor(3.)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
y = x * w + b
y.backward()
print(w.grad, b.grad) # 2.0, 1.0
```

```python
import torch
x = torch.tensor(3.)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
y = x * w + b
w_grad, b_grad = torch.autograd.grad(y, [w, b])
print(w_grad, b_grad) # 2.0, 1.0
```

值得主意的是，如果 `torch.autograd,grad` 的参数只填 `w` 或者 `b` 的话，需要指定 `retain_graph=True`，这是因为 PyTorch 默认计算图在反向传播后进行销毁，加上该设置将保留计算图。

```python
import torch
x = torch.tensor(3.)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
y = x * w + b
w_grad = torch.autograd.grad(y, w, retain_graph=True)
b_grad = torch.autograd.grad(y, b)
print(w_grad[0], b_grad[0]) # 2.0, 1.0
```

简单地说，`x.backward()` 等价于 `torch.autograd.grad(x)`。


## 多级梯度

例子：$z = x^2 y$，计算 $\frac{\partial^2 z}{\partial^2 x}$

```python
import torch
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = x * x * y
grad_x = torch.autograd.grad(z, x, retain_graph=True)
grad_xx = torch.autograd.grad(grad_x, x) # error
```

虽然`retain_graph=True` 保留了计算图和中间变量梯度， 但没有保存`grad_x` 的运算方式，需要使用`create_graph=True`在保留原图的基础上再建立额外的求导计算图，也就是会把 $\frac{\partial y}{\partial x}$ 这样的运算存下来。`torch.tensor.backward` 也同理。

```python
grad_x = torch.autograd.grad(z, x, create_graph=True) # 12
grad_xx = torch.autograd.grad(grad_x, x)             # 6
```

```python
z.backward(create_graph=True)
grad_xx = torch.autograd.grad(x.grad, x)
print(x.grad, grad_xx[0])                           # 12, 6
```

## 梯度清零

感觉这个例子用得，不好，非要扯上高阶梯度，太难理解了。
继续上面的例子。

```python
import torch
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = x * x * y
z.backward(create_graph=True)
print(x.grad)                                        # 12
x.grad.backward()
print(x.grad)                                        # 18
```

在调用 `z.backward(create_graph=True)` 后，`x.grad` 中存储的是 $\frac{\partial z}{\partial x}$，`x.grad.backward()` 表示  $\frac{\partial z}{\partial x}$ 对 $x$ 进行求梯度（这个值为 6），并将梯度值存储在 `x` 的梯度中（即 `x.grad`）。PyTorch 默认会累加梯度，`x.grad` 的本身的值为 12，进行累加后，变成了 18。这也是为什么在进行训练的时候需要对梯度进行清零的原因。

```python
import torch
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = x * x * y
z.backward(create_graph=True)
print(x.grad)                                        # 12
x.grad.data.zero_()
print(x.grad)                                        # 0
x.grad.backward()
print(x.grad)                                        # 6
```

## 张量梯度

上面的例子中，都是标量对标量的梯度（准确地说，应该就是求导）。对于学过大学的高等数学的话，很容易理解标量对向量（或者矩阵，甚至张量）的求导。这里主要讲述向量对向量的梯度，这个理解了，就很容易理解 Tensor（多维数组）对 Tensor 的求导。

```python
x = torch.tensor([2., 3.], requires_grad=True)
y = x * x
y.backward() # error
```

报错了，因为 PyTorch 只能标量对标量、标量对向量的求导。向量 $y$ 对向量 $x$ 的梯度可定义为

$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} \\
\frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2}
\end{bmatrix}
$$

而我们想得到的是 $[\frac{\partial y_1}{\partial x_1}, \frac{\partial y_2}{\partial x_2}]$

注意到 $\frac{\partial y_1}{\partial x_2}$ 和 $\frac{\partial y_2}{\partial x_1}$ 都为 0，故

$$
\frac{\partial y_1}{\partial x_1} = 
[1, 1]
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} \\
\frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2}
\end{bmatrix}
$$

所以在代码中应该如下写，其中 `torch.ones_likes(y)` 表明传入梯度值。

```python
import torch
x = torch.tensor([2., 3.], requires_grad=True)
y = x * x
y.backward(torch.ones_likes(y))
print(x.grad) # (4, 6)
```

这里有一个简易的应用，比如有两个 loss 值，想给它们分配 0.4 和 0.6 的权重的话，只需要传入 $[0.4, 0.6]$  的向量即可。

```python
import torch
x = torch.tensor([2., 3.], requires_grad=True)
y = x * x
y.backward(torch.tensor([0.4, 0.6]))
print(x.grad) # (1.6, 3.6)
```

## 禁用梯度

PyTorch 中可以使用 `torch.no_grad` 这个上下文管理器在禁用梯度

```python
import torch
x = torch.tensor([2., 3.], requires_grad=True)
with torch.no_grad():
  x += 1
  y = x * x
print(x)               # [3, 4] requires_grad=True
print(x._version)      # 1
print(y.requires_grad) # False
```

在 `torch.no_grad` 的上下文中，任何操作都不会被记录到梯度中，但是 in-place 操作还是会被记录。

如果想让某个函数不被记录，使用 `@torch.no_grad()` 装饰器

```python
import torch
x = torch.tensor([2., 3.], requires_grad=True)

@torch.no_grad()
def f(x):
    x += 1
    return x * x
f(x)
print(x)               # [3, 4] requires_grad=True
print(x._version)      # 1
print(y.requires_grad) # False
```

## 切断反向传播

先说说为什么需要切断反向传播，例如有两个模型 A 和 B，希望 A 的输出用于 B 的输入，在训练 B 模型的时候，肯定不希望反向传播时会传到 A 那边去，这时候需要切断 A 和 B 的“联系”。

Pytorch 总的来说切断反向传播，有两个方法：`torch.Tensor.data` 和 `torch.Tensor.detach`。

- 相同点：两者都是返回一个共享数据的 Tensor，即修改该 Tensor，原 Tensor 也跟着变化，并且新的 Tensor 的 `requires_grad` 为 `False`，即不可求导的。
- 不同点：`torch.Tensor.data` 的新 Tensor 不会被 autograd 记录，所以它是不安全的。

`torch.Tensor.data` 是 0.4 版本遗留下来的东西，建议使用 `torch.Tensor.detach`。具体的话见下面的例子。

```python
import torch
x = torch.tensor([1., 2.], requires_grad =True)
y = x.sigmoid()
y_ = y.data
y_.zero_()
z = y.sum()
z.backward()
print(x.grad) # [0, 0]
```

```python
import torch
x = torch.tensor([1., 2.], requires_grad =True)
y = x.sigmoid()
y_ = y.detach()
y_.zero_()
z = y.sum()
z.backward() # error
```

使用 `.data`，虽然 `x.grad` 可以正常输出，但是它的值是错误的。而使用 `.detach()` 直接爆出了运行错误。

很多解释是说这是因为 `y` 作为整个计算图的一个节点，在没有完成反向传播前，不能修改其值，但只是浅层的原因，深层的可以再看下面的例子。

```python
import torch
x = torch.tensor([1., 2.], requires_grad =True)
y = x * x
y_ = y.data
y_.zero_()
z = y.sum()
z.backward()
print(x.grad) # [2, 4]
```

```python
import torch
x = torch.tensor([1., 2.], requires_grad =True)
y = x * x
y_ = y.detach()
y_.zero_()
z = y.sum()
z.backward()
print(x.grad) # [2, 4]

```

上面这两个例子，都可以正常输出，其输出值都是正确的。其实原因在于 $\mathrm{sigmoid}$ 函数，很容易得知 $\mathrm{sigmoid}'(x) = \mathrm{sigmoid}(x)(1-\mathrm{sigmoid}(x))$，对应代码中 `y * (1 - y)`，而在反向传播前，将 `y` 修改成 0，使得 `y` 传给 `x` 的梯度值为 0，导致最终的结果为 0。而 $y = x^2$ 的梯度值为  $2 x$，修改 `y` 不会影响传递的梯度值，所以结果是正确的。

从上面这几个例子可以看出，为什么要 `.detach` 而不是 `.data` 了，毕竟有错误还有抛出运行错误。

## 参考资料

- [知乎 - 一文搞懂 PyTorch 内部机制](https://zhuanlan.zhihu.com/p/338256656)
- [一文解释PyTorch求导相关 (backward, autograd.grad)](https://zhuanlan.zhihu.com/p/279758736)
- [神经网络自动求导的设计与实现](https://zhuanlan.zhihu.com/p/82582926)
- [eat_pytorch_in_20_days - 2-3,动态计算图](https://lyhue1991.github.io/eat_pytorch_in_20_days/2-3,动态计算图.html)
- [PyTorch：view() 与 reshape() 区别详解](https://blog.csdn.net/Flag_ing/article/details/109129752)
- [pytorch中的detach和data](https://zhuanlan.zhihu.com/p/83329768)
- [pytorch中的.detach和.data深入详解](https://blog.csdn.net/qq_27825451/article/details/96837905)

