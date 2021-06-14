# 维度

## 转置

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`.T`|`.t()`|-|

高级转置函数

||numpy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.reshape`, `.reshape`|`torch.reshape`, `.reshape`|`tf.reshape`|
||`.T`|`.t()`||

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.transpose`|`torch.transpose`|`tf.transpose`|

`transpose` 函数在这三个库的参数都不一样，具体使用见参考文档。

下面是 TensorFlow `transpose` 的使用教程。

```python
a = tf.random.normal((4, 3, 2, 1)) # (4 ,3, 2, 1)
tf.transpose(a) # (1, 2, 3, 4)
tf.transpose(a, perm=[0, 1, 3, 2]) # (4, 3, 1, 2)
```

- `tf.expand_dims`

```python
a = tf.random.normal([3, 4, 5])
tf.expend_dims(a, axis=1) # (3, 1, 4, 5)

tf.expand_dims(a, axis=-1) # (3, 4, 5, 1)
```

关于 `axis` 的理解：

- 正数：表示在该轴前面增加
- 负数：表示在该轴后面增加

## reshape

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.reshape`, `.reshape`|`torch.reshape`, `.reshape`|`tf.reshape`|

由于 TensorFlow 对 Tensor 默认是 constant，所以无法直接在 Tensor 对象调用 reshape 函数。

## 分割合并

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.reshape`, `.reshape`|`torch.reshape`, `.reshape`|`tf.reshape`|

- `tf.squeeze`

```python
a = tf.zeros((1, 2, 1, 1, 3))

tf.squeeze(a) # (2, 3)
tf.squeeze(a, axis=0) # (2, 1, 1, 3)
tf.squeeze(a, axis=-2) # (1, 2, 1, 3)
```

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

## 合并与分割


- `tf.concat`：`np.concatenate`

```python
a = tf.ones((4, 35, 8))
b = tf.ones((2, 35, 8))
c = tf.concat((a, b), axis=0) # (6, 35, 8)

a = tf.ones((4, 35, 8))
b = tf.ones((4, 3, 8))
c = tf.concat((a, b), axis=1) # (4, 38, 8)

a = tf.ones((4, 35, 8))
b = tf.ones((4, 35, 8))
c = tf.concat((a, b), axis=-1) # (4, 38, 16)
# -1 表示倒数第一个轴
```

- `tf.stack`

```python
a = tf.ones((4, 35, 8))
b = tf.ones((4, 35, 8))
c = tf.stack((a, b), axis=0) # (2, 4, 38, 8)
d = tf.stack((a, b), axis=3) # (4, 38, 8, 2)

```

- `tf.unstack`

```python
a = tf.ones((2, 35, 8))
a1, a2 = tf.unstack(a, axis=0)
# a1 (35, 8)
# a2 (35, 8)

a = tf.ones((3, 35, 8))
a1, a2, a3 = tf.unstack(a, axis=0)
# a1, a2, a3 (35, 8)

a = tf.ones((4, 35, 8))
a1, a2 = tf.unstack(a, axis=0, num_or_size_splits=[1, 3])
# a1 (1, 35, 8)
# a2 (3, 35, 8)
```