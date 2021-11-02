import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import save_checkpoint, load_checkpoint
from mindspore.common.initializer import initializer


class Critic(nn.Cell):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Dense(in_channels=action_dim + state_dim, out_channels=32),
            nn.ReLU(),
            nn.Dense(in_channels=32, out_channels=1),
        ])
        self.Cat = ops.Concat(axis=1)

    def construct(self, state, action):
        x = self.Cat((state, action))
        out = self.fc(x)
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

    def construct(self, state):
        out = self.fc(state)
        return out


if __name__ == '__main__':
    Cat = ops.Concat(axis=1)
    Critic = Critic(state_dim=4, action_dim=2)
    x = initializer(5, shape=(64, 4))
    y = initializer(4, shape=(64, 2))
    print(Critic(x, y))
