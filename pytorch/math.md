## 数据统计

```python
a = tf.ones((3, 4))
tf.reduce_sum(a) # ()
tf.reduce_sum(a, axis=0) # (1, 4) -> (4, )
tf.reduce_sum(a, axis=1) # (3, 1) -> (3, )
```

注：tensorflow 对于哪些会压缩计算的函数前面都加了 `reduce_`

