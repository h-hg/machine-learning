# Tensor 类型出错

NumPy 默认 int64，而 PyTorch 默认 int32，在浮点数上，NumPy 默认 double（64），而 PyTorch 默认 float（32），有时候会出现问题。

```python
import numpy as np
import torch
a = np.array([1.2, 1.3])
x = torch.from_numpy(a)
model = torch.nn.Linear(2, 1)
model(x) # RuntimeError: expected scalar type Double but found Float
```

解决方法

1. 将输入改成 float

```python
import numpy as np
import torch
a = np.array([1.2, 1.3])
x = torch.from_numpy(a).float()
model = torch.nn.Linear(2, 1)
model(x)
```

2. 将模型改成 double

```python
import numpy as np
import torch
a = np.array([1.2, 1.3])
x = torch.from_numpy(a)
model = torch.nn.Linear(2, 1).double()
model(x)
```

## 额外

1. 貌似只有调用内置的层才会，自己写的层（`torch.nn.Module`）没有这个问题，可能是调用 C/C++ 库的缘故？

## 参考资料

- [How to fix RuntimeError “Expected object of scalar type Float but got scalar type Double for argument”?](https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ)
