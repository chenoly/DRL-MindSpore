from mindspore import Tensor

x = Tensor([[1, 2], [3, 6], [5, 4]])
print(x.max(axis=1))
