weight_decay 表示二阶正则化的参数？

L1 需要人为去实现？

```python
import torch
regularization_loss = 0
for param in model.parameters():
    regularization_loss = torch.sum(torch.abs(param))
total_loss = loss + 0.01 * regularization_loss
# 这个代码好像没有考虑 requires_grad=True 这个东西
```

torch.optim.lr_scheduler.ReduceLRonPlateau 用来监听 loss，根据 loss 来调整 learning rate

还有其他类似的

