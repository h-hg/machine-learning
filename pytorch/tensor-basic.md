# Tensor Basic

以 numpy 为基准介绍 PyTorch Tensor 的基本方法

## 属性

|描述|numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`.dtype`|`.dtype`|`.dtype`|
||`.ndim`|`.ndim`|`.ndim`|
||`.shape`|`.shape`|`.shape`|
||`.size`|`.size()`||

## 创建

|描述|numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.array`|`torch.tensor`|`tf.constant`|

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.zeros`|`torch.zeros`|`tf.zeros`|
||`np.zeros_like`|`torch.zeros_like`|`tf.zeros_like`|
||`np.ones`|`torch.ones`|`tf.ones`|
||`np.ones_like`|`torch.ones_like`|`tf.ones_like`|
||`np.empty`|`torch.empty`||

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.arange`|`torch.arange`, `torch.range`|`tf.range`|
||`np.linspace`|`torch.linspace`|`tf.linspace`|
||`np.logspace`|`torch.logspace`|`tf.logspace`|
||`np.full`|`torch.full`|`tf.fill`|

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
|[0, 1] 均匀分布|`np.random.rand`|`torch.rand`|`tf.random.uniform`|
|标准正态分布 N(0, 1)|`np.random.randn`|`torch.randn`|`tf.random.normal`|
|区间内随机整数|`np.random.randint`|`torch.randint`||

## NumPy 兼容

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
|from `np.array`|-|`.numpy()`|`.numpy()`|
|to `np.array`|-|`torch.from_numpy`|`tf.convert_to_tensor`|

值得注意的是，PyTorch 上面两个关于 numpy 的转化，转换前后两个向量是共享内存，一个向量的修改，会影响另一个向量。

