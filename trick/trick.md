# 技巧

## 提前中断训练

每次 epoch 都保存模型参数，同时计算相关 acc，所以 epoch 训练完毕后，根据 acc 记录选择最好的模型参数。

## K 折交叉验证

每一次 epoch 都重新划分 train 和 valid，然后再使用 train 训练，

这样的好处：

1. 整个数据集（除去 test），都有被利用到。

## Dropout

learning less to learn better

each connection has p = [0, 1] to loss

直接使用 torch.nn.Dropout

```python
import torch
net = torch.nn.Sequential(
    torch.nn.Linear(784, 200),
    torch.nn.Dropout(0.3) # 丢弃概率为 0.3
    # tf.nn.dropout() # 参数为保存概率
    torch.nn.ReLU(),
    torch.nn.Linear(200, 1) 
)
```

train 的时候有 dropout，但是 test 的时候不能是使用 droput

```python
for i in range(epochs):
    net_dropped.train() # 开启训练模式
    for x,y in dataloader:
        pass
    net_dropped.eval() # 开启测试模式
    test_loss = 0
    for x,y in test_loader:
        pass
```