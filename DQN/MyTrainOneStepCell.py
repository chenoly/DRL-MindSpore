import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple


class CriticWithLossCell(nn.Cell):
    def __init__(self, critic, loss_fn):
        super().__init__()
        self._critic = critic
        self._loss_fn = loss_fn
        self.gather = ops.Gather()

    def construct(self, state, action, td_error):
        q = self.gather(self._critic(state), action, 1)
        return self._loss_fn(q, td_error)


# 生成参数梯度
class CriticGradWrap(nn.Cell):
    """ CriticGradWrap definition """

    def __init__(self, loss_network, Critic_network):
        super().__init__(auto_prefix=False)
        self.loss_network = loss_network
        self.weights = ParameterTuple(Critic_network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True)

    # 注释2
    def construct(self, state, action, td_error):
        """
        actor gradient ascend
        :param state:
        :return:
        """
        weights = self.weights
        return self.grad(self.loss_network, weights)(state, action, td_error)
