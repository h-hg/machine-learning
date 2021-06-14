# Tensor

- Eager mode

基本数据类型

- int
- float
- double
- bool
- string

屏蔽 tensorflow 的 warning

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
```

## 创建

`tf.constant` 的概念来源于 1.x

## 设备

```python
# 创建一个 cpu 的 tensor
with tf.device("cpu"):
  a_cpu = tf.constant(1)

# 创建一个 gpu 的 tensor
with tf.device("gpu"):
  a_gpu = tf.constant(1)

# 获取 tensor 的设置
a.device

# tensor: gpu -> cpu
a = a_gpu.cpu()

# tensor: cpu -> gpu
a = a_cpu.gpu()

```

注意：

- 两个处于不同设备的 tensor 不能直接参与运算
- 创建的 tensor 默认设备取决于你安装的 tensorlow 版本

## 类型

```python
# tensor -> numpy
a_np = a.numpy()

# numpy -> tensor
a = tf.constant(a) # tf.convert_to_tensor(a)

# 类型转换
a = tf.cast(a, dtype=tf.double)

# 判断
isinstance(a, tf.Tensor)
tf.is_tensor(a) # 推荐
```

## Variable

```python
b = tf.Variable(1, name="a")
b.name # a
b.trainable # True，即需要梯度信息
isinstance(b, tf.Tensor) # False
isinstance(b, tf.Variable) # True
tf.is_tensor(b) # True
```

这里表明 tf.Variable 是 tf.Tensor 的子类。

## 初始化

创建的 Tensor 创建函数

- tf.zeros
- tf.ones
- tf.fill
- tf.random.normal：正态分布
- tf.random.uniform：均匀分布
- tf.range
- tf.constant
- tf.Variable

## 切片

### 基本

语法：`start:end:step`

- `start` 默认为 0
- `end` 默认为 `n`（总的元素个数）
- `step` 默认为 1

即有

- `start:end = start:end:1`
- `::step = 0:n:step`

负数的意义：

- 如果 `start` 或 `end` 为负数的话，表示倒数第几个
- 如果 `step` 为负数的话，表示从尾部向前，也就是倒序的功能

```python
a = tf.range(10)
b = a[::-2] # [9 7 5 3 1]
```

`...` 表示多个 `:`，表示冒号的数量为自动推导

### 高级

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

## 维度变换

- `tf.reshape`：`np.reshape`

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

## 数学运算

基本跟 numpy 一致

```
[b, 3, 4] @ [b, 4 ,5] = [b, 3, 5]
# 相当于多个矩阵的乘法
```

## 前向传播

`tf.GradientType()` 只会跟踪 `tf.Variable`，不会跟踪那些 `tf.constant` 创建的

```python
# 手写 tensorflow 多层神经网络注意事项

w = w - lr * grad # 这是错误的，因为这样赋值后 w 变成了 tf.Constant，而不是 Variable
# 应该使用
w.assign_sub(lr * grad)
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

## 数据统计

```python
a = tf.ones((3, 4))
tf.reduce_sum(a) # ()
tf.reduce_sum(a, axis=0) # (1, 4) -> (4, )
tf.reduce_sum(a, axis=1) # (3, 1) -> (3, )
```

注：tensorflow 对于哪些会压缩计算的函数前面都加了 `reduce_`

## Tensor sort

- `tf.argsort`
- `tf.sort`
- `tf.topk`

这一部分，后面看看，指定的轴是什么看的

## 复制和填充

- `tf.pad`：填充

```python
a = tf.ones((1, 5, 5, 1))
b = tf.pad(a, [[0, 0], [1, 2], [3, 4], [0, 0]])
# (0 + 1 + 0, 1 + 5 + 2, 3 + 5 + 4, 0 + 1 + 0) = (1, 8, 12, 1)
```

- `tf.tile`

```python
a = tf.ones(3, 4)
b = tf.tile(a, [2, 5])
# (3 * 2, 4 * 5) = (6, 20)
```

## 张量限幅


