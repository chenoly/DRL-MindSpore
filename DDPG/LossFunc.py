import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops


class AGradMSELoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def construct(self, logits, labels):
        return -self.loss(logits, labels)


class DGradMSELoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def construct(self, logits, labels):
        return self.loss(logits, labels)
