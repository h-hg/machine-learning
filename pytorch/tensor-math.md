# Tensor 数学运算

## 矩阵乘法

||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
|点积|`np.dot`|`torch.dot`||
||`@`|`@`, `torch.mm`|`@`|
||`np.dot`, `np.matmul`|`torch.matmul`|`tf.linalg.matmul`|
||`np.split`|`torch.split`|`tf.unstack`|

```python
a.shape             # (4, 5)
b.shape             # (5, 6)
a @ b               # (4, 6)

a.shape             # (4, 3, 5)
b.shape             # (4, 5, 6)
a @ b               # error
torch.matmul(a, b)  # (4, 3, 6)
```


||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.log`|`torch.log`|`tf.math.log`|
||`np.exp`|`torch.exp`|`tf.math.exp`|
||`np.pow`|`torch.pow`|`tf.math.pow`|
||`np.sqrt`|`torch.sqrt`|`tf.math.sqrt`|
||`np.ceil`|`torch.ceil`|`tf.math.ceil`|
||`np.floor`|`torch.floor`|`tf.math.floor`|
||`np.round`|`torch.round`|`tf.math.round`|
||`np.trunc`|`torch.trunc`|`tf.math.trunc`|
||`np.minimum`|`torch.minimum`|`tf.math.minimum`|
||`np.maxmum`|`torch.maxmum`|`tf.math.maxmum`|
||`np.clip`|`torch.clip`|`tf.clip_by_value`|
||`np.min`|`torch.min`|`tf.math.reduce_min`|
||`np.max`|`torch.max`|`tf.math.reduce_max`|
||`np.sum`|`torch.sum`|`tf.math.reduce_sum`|
||`np.prod`|`torch.prod`|`tf.reduce_prod`|
||`np.argmin`|`torch.argmin`|`tf.math.argmin`|
||`np.argmax`|`torch.argmax`|`tf.math.argmax`|
||`np.mean`|`torch.mean`|`tf.math.mean`|
||`np.linalg.norm`|`torch.linalg.norm`|`tf.linalg.norm`|
|||`torch.topk`|`tf.topk`|


||NumPy|PyTorch|TensorFlow|
|:-:|:-:|:-:|:-:|
||`np.where`|`torch.where`|`tf.math.where`|
