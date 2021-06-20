# 维度

## 转置

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`.T`|`.T`||

高级转置函数

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.transpose`|`torch.transpose`|`tf.transpose`|

`transpose` 函数在这三个库的参数都不一样，具体使用见参考文档。
（好像 pytorch 中是另一个函数名）

下面是 TensorFlow `transpose` 的使用教程。

```python
a = tf.random.normal((4, 3, 2, 1)) # (4 ,3, 2, 1)
tf.transpose(a) # (1, 2, 3, 4)
tf.transpose(a, perm=[0, 1, 3, 2]) # (4, 3, 1, 2)
```

PyTorch 的 `transpose` 只能交换两个维度，高级的使用 `torch.Tensor.permute`

```python
a.shape                # (5, 6, 7, 8)
a.permute(0, 2, 3, 1)  # (5, 7, 8, 6)
```

## reshape

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.reshape`, `.reshape`|`torch.reshape`, `.reshape`|`tf.reshape`|
||`.T`|`.t()`||

由于 TensorFlow 对 Tensor 默认是 constant，所以无法直接在 Tensor 对象调用 reshape 函数。

## 压缩与扩展

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`.squeeze`, `np.squeeze`|`.squeeze`, `torch.squeeze`|`tf.squeeze`|
||`.expand_dims`, `np.expand_dims`|`.unsqueeze`, `torch.unsqueeze`|`tf.expand_dims`|


```python
a.shape                     # (1, 2, 1, 1, 3)
torch.squeeze(a)            # (2, 3)
torch.squeeze(a, axis=0)    # (2, 1, 1, 3)
torch.squeeze(a, axis=-2)   # (1, 2, 1, 3)

a.shape                     # (3, 4, 5)
torch.unsqueeze(a, axis=1)  # (3, 1, 4, 5)
torch.unsqueeze(a, axis=-1) # (3, 4, 5, 1)
```

`axis` 表示新轴的位置

## 广播

与 numpy 一致

查看什么时候可以 broadcast

```python
def detect(shape1, shape2):
  n1, n2 = len(shape1), len(shape2)
  diff = abs(n1 - n2)
  n = max(n1, n2)
  if n1 < n2:
    shape1 = ([1] * diff) + shape1
  else:
    shape2 = ([1] * diff) + shape2

  for i in range(n-1, -1, -1):
    if shape1[i] == 1 or shape2[i] == 1:
      continue
    elif shape1[i] != shape2[i]:
      return False
  return True
```


||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`.brand`, `np.squeeze`|`.squeeze`, `torch.squeeze`|`tf.squeeze`|
||`.expand_dims`, `np.expand_dims`|`.unsqueeze`, `torch.unsqueeze`|`tf.expand_dims`|


```python
a.shape                     # (4, 32, 14, 14)
b.shape                     # (1, 32, 12, 1)
b.expand([4, 32, 12, 14])   # (4, 32, 12, 14)
b.expand([-1, 32, -1, 14])  # (1, 32, 12, 14)
```

`torch.Tensor.expand` 没有复制出新的内存

```python
a.shape                     # (3, 4)
a.repeat([2, 5])            # (3 * 2. 4 * 5)
```

## 合并与分割

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.concatenate`|`torch.cat`|`tf.concat`|
||`np.stack`|`torch.stack`|`tf.stack`|
||`np.split`|`torch.split`|`tf.unstack`|


```python
a.shape                       # (4, 35, 8)
b.shape                       # (2, 35, 8)
torch.cat([a, b], axis=0)     # (6, 35, 8)

a.shape                       # (4, 35, 8)
b.shape                       # (4, 3, 8)
torch.cat([a, b], axis=1)     # (4, 38, 8)

a.shape                       # (4, 35, 3)
b.shape                       # (4, 35, 8)
torch.cat([a, b], axis=-1)    # (4, 38, 11)
```

```python
a.shape                       # (4, 35, 8)
b.shape                       # (4, 35, 8)
torch.stack([a, b], axis=0)   # (2, 4, 35, 8)
torch.stack([a, b], axis=3)   # (4, 35, 8, 2)
```

```python
a.shape                       # (2, 35, 8)
a1, a2 = torch.split(a, 2, axis=0)
a1.shape                      # (1, 35, 8)
a2.shape                      # (1, 35, 8)

a.shape                       # (3, 35, 8)
a1, a2 = torch.split(a, [1, 2], axis=0)
a1.shape                      # (1, 35, 8)
a2.shape                      # (2, 35, 8)
```
