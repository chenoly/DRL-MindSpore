import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.nn.probability.distribution import Categorical


class Actor(nn.Cell):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Dense(in_channels=state_dim, out_channels=32),
            nn.ReLU(),
            nn.Dense(in_channels=32, out_channels=action_dim),
        ])
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, state):
        out = self.fc(state)
        return self.softmax(out)


actor = Actor(4, 2)
for i in range(1000):
    x = Tensor(np.random.random(size=(10, 4)),dtype=mindspore.float32)
    y = actor(x)
    print(y)
    cat = Categorical(y)
    index = cat.sample()
    print(index)