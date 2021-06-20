# 切片

## 基本切片

三者在基本切片上无差别

语法：`start:end:stop`

- `start`：默认为 0
- `end`：默认为该维度的长度
- `step`：默认为 1

eg.

- `start:end` = `start:end:1`
- `::step` = `0:n:step`

负数的意义：

- 如果 `start` 或 `end` 为负数的话，表示倒数第几个
- 如果 `step` 为负数的话，表示从尾部向前，也就是倒序的功能

```python
a = torch.arange(10)
b = a[::-2] # [9 7 5 3 1]
```

`...` 表示多个 `:`，表示冒号的数量为自动推导

## 整数索引

NumPy 和 PyTorch 可以直接使用整数索引，但是 TensorFlow 不可以。

||NumPy|PyTorch|TensorFlow|
||`np.take`|`torch.gather`|`tf.gather`|
||`np.take`|x|`tf.gather_nd`|

- `tf.gather`
- `tf.gather_nd`：`nd` 表示 n dimension，即多维
- `tf.boolean_mask`

```python
# 假设 a 的 shape 为 (3, 4, 5)

b = tf.gather(a, axis=1, indices=[2, 3]) 
# 等价于 numpy 的 a[:, [2, 3], :]
# b 的 shape 为 (3, 2, 5)
```

```python
b = tf.gather_nd(a, [0])
# 等价于 numpy a[0]，shape (4, 5)

b = tf.gather_nd(a, [0, 2])
# 等价于 numpy a[0, 2]，shape (5, )

b = tf.gather_nd(a, [0, 1, 2])
# 等价于 numpy a[0, 1, 2]，shape ()

b = tf.gather_nd(a, [[0, 1, 2]])
# 等价于 numpy [a[0, 1, 2]]，shape (1, )

b = tf.gather_nd(a, [[0, 1], [1, 2]])
# 等价如下，shape (2, 5)
"""
[

  a[0, 1],
  a[1, 2]
]
"""

b = tf.gather_nd(a, [[[0,1], [1, 2]]])
# 等价如下，shape (1, 2, 5)
"""
[
  [
    a[0, 1],
    a[1, 2]
  ]
]
"""
```

```python
b = tf.boolean_mask(a, mask=(a < 3))
# 等价于 numpy a[a < 3]
```

## bolean 索引

NumPy 和 PyTorch 可以直接使用布尔索引，但是 TensorFlow 不可以。

||NumPy|PyTorch|TensorFlow|
||x|x|`tf.boolean_mask`|
