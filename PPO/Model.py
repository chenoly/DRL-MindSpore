import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.probability.distribution import Categorical
from mindspore import Tensor
import numpy as np


class Critic(nn.Cell):
    """
    value-function: V(s)
    """

    def __init__(self, state_dim):
        """
        state size
        :param state_dim:
        """
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Dense(in_channels=state_dim, out_channels=32),
            nn.ReLU(),
            nn.Dense(in_channels=32, out_channels=1),
        ])
        self.Cat = ops.Concat(axis=1)

    def construct(self, state):
        out = self.fc(state)
        return out


class Actor(nn.Cell):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Dense(in_channels=state_dim, out_channels=32),
            nn.ReLU(),
            nn.Dense(in_channels=32, out_channels=action_dim),
            nn.Tanh()
        ])
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, state):
        out = self.fc(state)
        if out.sum() - 1. != 0.0:
            d = 1 - out.sum()
            out[0,0] += d
        return self.softmax(out)


if __name__ == '__main__':
    a = Actor(4, 2)
    x1 = Tensor(np.random.random(size=(10, 1))).clip(0, 1)
    x2 = 1 - x1
    cat = ops.Concat(axis=1)
    y = cat((x1, x2))
    ca1 = Categorical(probs=y)
    x = ca1.prob([0.2])
    print(x)
    for i in range(100):
        a = ca1.sample()
        print(ca1.prob([]))
