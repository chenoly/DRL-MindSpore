import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ParameterTuple
from mindspore.nn.probability.distribution import Categorical


class ActorWithLossCell(nn.Cell):
    def __init__(self, actor, actor_):
        super().__init__()
        self.actor = actor
        self.actor_ = actor_
        # self.Cat = Categorical()
        self.ops = ops.Minimum()

    def construct(self, state, advantage_reward, epsilon):
        cat = Categorical(self.actor(state))
        index = cat.sample()
        log_probs = ops.exp(cat.log_prob(index))

        cat_ = Categorical(self.actor_(state))
        index_ = cat_.sample()
        log__probs = ops.exp(cat_.log_prob(index_))

        v1 = (log_probs - log__probs)*advantage_reward
        v2 = ops.clip_by_value(advantage_reward,clip_value_min=1-epsilon,clip_value_max=1+epsilon)
        J = -self.ops(v1,v2).mean()
        return J


class CriticWithLossCell(nn.Cell):
    def __init__(self, critic):
        super().__init__()
        self._critic = critic
        self._loss_fn = nn.MSELoss()

    def construct(self, state,advantage_reward):
        q = self._critic(state)
        return self._loss_fn(q, advantage_reward)


# 生成参数梯度
class ActorGradWrap(nn.Cell):
    """ ActorGradWrap definition """

    def __init__(self, loss_network, Actor_network):
        super().__init__(auto_prefix=False)
        self.loss_network = loss_network
        self.weights = ParameterTuple(Actor_network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True)

    # 注释2
    def construct(self, state, advantage_reward, epsilon):
        """
        actor gradient ascend
        :param state:
        :return:
        """
        weights = self.weights
        return self.grad(self.loss_network, weights)(state, advantage_reward, epsilon)


# 生成参数梯度
class CriticGradWrap(nn.Cell):
    """ CriticGradWrap definition """

    def __init__(self, loss_network, Critic_network):
        super().__init__(auto_prefix=False)
        self.loss_network = loss_network
        self.weights = ParameterTuple(Critic_network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True)

    # 注释2
    def construct(self, state, advantage_reward):
        """
        actor gradient ascend
        :param state:
        :return:
        """
        weights = self.weights
        return self.grad(self.loss_network, weights)(state, advantage_reward)
