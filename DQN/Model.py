import mindspore.nn as nn


class Eval_Net(nn.Cell):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Dense(in_channels=state_dim, out_channels=16),
            nn.ReLU(),
            nn.Dense(in_channels=16, out_channels=8),
            nn.ReLU(),
            nn.Dense(in_channels=8, out_channels=action_dim),
        ])

    def construct(self, state):
        out = self.fc(state)
        return out

