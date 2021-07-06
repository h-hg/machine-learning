# 数据集

数据集可以分成如下两种

- 映射式：`torch.utils.data.Dataset`，给定某一个 key（索引），能够直接返回相应的 value（样本）
- 迭代式：`torch.utils.data.IterableDataset`，当随机读取代价较高或不可能实现时，以及批处理大小取决于所获取数据的情况时，使用这种数据集是相当合适的。

```python
class Dataset(object):
	# 强制所有的子类override getitem和len两个函数，否则就抛出错误；
	# 输入数据索引，输出为索引指向的数据以及标签；
	def __getitem__(self, index):
		raise NotImplementedError
	
	# 输出数据的长度
	def __len__(self):
		raise NotImplementedError
		
	def __add__(self, other):
		return ConcatDataset([self, other])
```

```python
class ConcatDataset(Dataset):
	@staticmethod
	def cumsum(sequence):
		# sequence是一个列表，e.g. [[1,2,3], [a,b], [4,h]]
		# return 一个数据大小列表，[3, 5, 7], 明显看的出来包含数据多少，第一个代表第一个数据的大小，第二个代表第一个+第二数据的大小，最后代表所有的数据大学；
	...
	def __getitem__(self, idx):
		# 主要是这个函数，通过bisect的类实现了任意索引数据的输出；
		dataset_idx = bisect.bisect_right(self.cumulative_size, idx)
		if dataset_idx == 0:
			sample_idx == idx
		else:
			sample_idx = idx - self.cumulative_sizes[dataset_idx -1]
		return self.datasets[dataset_idx][sample_idx]
```

```python
class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
	'''数据集包装张量。

    每个样本将通过沿第一维索引张量来检索。

    参数：
        *张量（张量）：具有与第一维相同大小的张量。'''
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
```

