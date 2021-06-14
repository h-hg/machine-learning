# 修改

修改时为了避免维度记录，可以使用如下进行修改

```python
with torch.no_grad():
  w -= lr * w.grad
  b-= lr * b.grad
```

```python
w.data -= lr * w.grad
b.data -= lr * b.grad
```

# 梯度清零

```python
w.grad.zero_()
b.grad.zero_()
```

# 利用 torch.nn.Module 建立网络

```python
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(2,4))
        self.b1 = nn.Parameter(torch.zeros(1,4))
        self.w2 = nn.Parameter(torch.randn(4,8))
        self.b2 = nn.Parameter(torch.zeros(1,8))
        self.w3 = nn.Parameter(torch.randn(8,1))
        self.b3 = nn.Parameter(torch.zeros(1,1))

    # 正向传播
    def forward(self,x):
        x = torch.relu(x@self.w1 + self.b1)
        x = torch.relu(x@self.w2 + self.b2)
        y = torch.sigmoid(x@self.w3 + self.b3)
        return y
    
    # 损失函数(二元交叉熵)
    def loss_func(self,y_pred,y_true):  
        #将预测值限制在1e-7以上, 1- (1e-7)以下，避免log(0)错误
        eps = 1e-7
        y_pred = torch.clamp(y_pred,eps,1.0-eps)
        bce = - y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
        return torch.mean(bce)
    
    # 评估指标(准确率)
    def metric_func(self,y_pred,y_true):
        y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                          torch.zeros_like(y_pred,dtype = torch.float32))
        acc = torch.mean(1-torch.abs(y_true-y_pred))
        return acc

model = DNNModel()
```

# torchkeras

这个库是那个《20天吃完pytorch》的作者开发的，代码量不多，简单地说，就是在 pytroch 上面封装了一个类似 tf.keras.Model 的模块，对 pytorch 构建的模型进行 summary, compile, fit, evaluate, predict

这是因为 pytorch 没有官方的高级 API，一般需要用户自己实现训练循环、验证循环和预测循环。


# Optimizer

创建

```python
model = nn.Linear(2, 1)
model.loss_func = nn.MSELoss()
model.optimizer = torch.optim.SGD(model.parameters, lr=0.01)
```

训练

```python
def train_step(model, features, labels):
    
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    loss.backward()
    model.optimizer.step() # 这个是干什么的
    model.optimizer.zero_grad()
    return loss.item()

# 测试train_step效果
features,labels = next(iter(dl))
train_step(model,features,labels)
```

# pytorch 结构

pytorch 神经网络的部分都封装在 `torch.nn` 下面

```python
from torch.nn import functional as F
```

激活函数

- F.relu
- F.sigmoid
- F.tanh

模型层

- F.linear
- F.conv2d

损失函数

- F.binary_cross_entropy

**为了便于对参数管理**，一般通过继承 `torch.nn.Module` 转换为类的实现形式，并直接封装在 `torch.nn` 

激活函数

- torch.nn.ReLU
- torch.nn.Sigmoid
- torch.nn.Tanh

模型层

- torch.nn.Linear
- torch.nn.Conv2d

损失函数

- torch.nn.MSELoss

参数管理，为了方便管理参数，pytorch 一般使用 `torch.nn.Parameter` 来管理参数

```python
w = torch.nn.Parameter(torch.randn(2, 2))
params_list = nn.ParameterList([torch.randn(2, 2), torch.randn(2, 2)])
params_dict = torch.nn.OarameterDict({
    "w": torch.nn.Parameter(torch.rand(2,2)),
    "b": torch.nn.Parameter(torch.zeros(1))
})
```

```python
# 可以用Module将它们管理起来
# module.parameters()返回一个生成器，包括其结构下的所有parameters

module = nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict

num_param = 0
for param in module.parameters():
    print(param,"\n")
    num_param = num_param + 1
print("number of Parameters =",num_param)
```

```python
#实践当中，一般通过继承nn.Module来构建模块类，并将所有含有需要学习的参数的部分放在构造函数中。

#以下范例为Pytorch中nn.Linear的源码的简化版本
#可以看到它将需要学习的参数放在了__init__构造函数中，并在forward中调用F.linear函数来实现计算逻辑。

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
```

## Dataset, DataLoader

Dataset 定义数据集的内容，它相当于一个类似 list 的数据结构，具有确定的长度，能够索引数据集中的元素。

DataLoader 定义了按 batch 加载数据集的方法，它实现了一个 `__iter__` 方法的可迭代对象，每次迭代输出一个 batch 的数据。

drop_last=False 时保留最后一个小于 batch 的批次

5-3 介绍了优化器的使用方法

