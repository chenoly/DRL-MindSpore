import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import save_checkpoint, load_checkpoint
from mindspore import Tensor
from LossFunc import DGradMSELoss, AGradMSELoss
from Model import Critic, Actor
from ReplayBuffer import ReBuffer
import mindspore.ops.functional as F
from MyTrainOneStepCell import CriticGradWrap, ActorGradWrap, CriticWithLossCell, ActorWithLossCell
import copy


class Agent:
    '''
    agent
    '''

    def __init__(self, action_dim, state_dim, action_max=1.0, action_min=-1.0, batchsize=64, lr=0.03, tau=0.005, gamma=0.9, policy_delay=2):
        """
                agent
        :param action_dim: action's space size
        :param state_dim: state's space size
        :param batchsize: training batchsize
        :param lr: learning rate
        :param tau: softupdate param
        :param action_max:
        :param action_min:
        :param gamma:
        :param policy_delay:
        """
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_max = action_max
        self.action_min = action_min
        self.policy_delay = policy_delay
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.Buffer = ReBuffer(batchsize=batchsize, maxbuffersize=102400)
        self.Critic1 = Critic(action_dim=action_dim, state_dim=state_dim)
        self.Critic1_ = copy.deepcopy(self.Critic1)
        self.Critic2 = Critic(action_dim=action_dim, state_dim=state_dim)
        self.Critic2_ = copy.deepcopy(self.Critic2)
        self.Actor = Actor(action_dim=action_dim, state_dim=state_dim)
        self.Actor_ = copy.deepcopy(self.Actor)

        self.AGradCritican = AGradMSELoss()
        self.DGradCritican = DGradMSELoss()
        self.Critic1_Loss = CriticWithLossCell(critic=self.Critic1, loss_fn=self.DGradCritican)
        self.Critic2_Loss = CriticWithLossCell(critic=self.Critic2, loss_fn=self.DGradCritican)
        self.Actor_Loss = ActorWithLossCell(critic=self.Critic1, actor=self.Actor, loss_fn=self.AGradCritican)

        self.Critic1optimizer = nn.Adam(self.Critic1.trainable_params(), learning_rate=self.lr)
        self.Critic2optimizer = nn.Adam(self.Critic2.trainable_params(), learning_rate=self.lr)
        self.Actoroptimizer = nn.Adam(self.Actor.trainable_params(), learning_rate=self.lr)

        self.Critic1_train = CriticGradWrap(loss_network=self.Critic1_Loss, Critic_network=self.Critic1)
        self.Critic2_train = CriticGradWrap(loss_network=self.Critic2_Loss, Critic_network=self.Critic2)
        self.Actor_train = ActorGradWrap(loss_network=self.Actor_Loss, Actor_network=self.Actor)

    def choose(self, state):
        '''
        choose a action by using Actor network,math define a = policy(s)
        :param state:
        :return:
        '''
        c = 0.2
        epsilon = np.random.normal(loc=0, scale=0.5, size=(1, self.action_dim))
        epsilon = Tensor.from_numpy(epsilon).clip(xmin=-c, xmax=c)
        state = Tensor.from_numpy(state).reshape(1, self.state_dim)
        action = self.Actor(state)
        A = (action + epsilon).clip(xmin=self.action_min, xmax=self.action_max)
        return A.asnumpy().flatten()

    def choose_(self, state_):
        '''
        train
        choose a action by using Actor network,math define a = policy(s)
        :param state:
        :return:
        '''
        c = 0.2
        epsilon = np.random.normal(loc=0, scale=0.2, size=(state_.shape[0], self.action_dim))
        epsilon = Tensor(epsilon, dtype=mindspore.float32).clip(xmin=-c, xmax=c)
        action = self.Actor_(state_)
        A = (action + epsilon).clip(xmin=self.action_min, xmax=self.action_max)
        return A

    def store(self, state, action, state_, reward, done):
        '''
        store a transition to Buffer
        :param state: precent state
        :param action: a action by precent state
        :param state_: net state
        :param reward: reward by playing a action
        :param done: game over or not
        :return: none
        '''
        self.Buffer.store(state, action, state_, reward, done)

    def train(self, t_epoch):
        '''
        train critic and actor networks
        :return:none
        '''
        if self.Buffer.num >= self.Buffer.batchsize:
            self.Critic1_train.set_train()
            self.Critic2_train.set_train()
            self.Actor_train.set_train()
            state, action, state_, reward, done = self.Buffer.retract_()
            state = Tensor.from_numpy(state).astype(dtype=mindspore.float32)
            action = Tensor.from_numpy(action).astype(dtype=mindspore.float32)
            state_ = Tensor.from_numpy(state_).astype(dtype=mindspore.float32)
            reward = Tensor.from_numpy(reward).astype(dtype=mindspore.float32)
            done = Tensor.from_numpy(done).astype(dtype=mindspore.float32)
            action_ = self.choose_(state_)
            y = reward + (1 - done) * self.gamma * F.minimum(self.Critic1_(state_, action_),
                                                             self.Critic2_(state_, action_))

            critic1_grad = self.Critic1_train(state, action, y)
            self.Critic1optimizer(critic1_grad)
            critic2_grad = self.Critic2_train(state, action, y)
            self.Critic2optimizer(critic2_grad)

            if t_epoch % self.policy_delay == 0:
                actor_grad = self.Actor_train(state, Tensor(0.0, mindspore.float32))
                self.Actoroptimizer(actor_grad)

                Critic1_params = self.Critic1.parameters_dict()
                _Critic1_params = self.Critic1_.parameters_dict()
                Critic2_params = self.Critic2.parameters_dict()
                _Critic2_params = self.Critic2_.parameters_dict()
                Actor_params = self.Actor.parameters_dict()
                _Actor_params = self.Actor_.parameters_dict()
                for param, _param in zip(Critic1_params, _Critic1_params):
                    _Critic1_params[_param] = self.tau * Critic1_params[param].clone() + (1 - self.tau) * _Critic1_params[
                        _param].clone()
                for param, _param in zip(Critic2_params, _Critic2_params):
                    _Critic2_params[_param] = self.tau * Critic2_params[param].clone() + (1 - self.tau) * _Critic2_params[
                        _param].clone()
                for param, _param in zip(Actor_params, _Actor_params):
                    _Actor_params[_param] = self.tau * Actor_params[param].clone() + (1 - self.tau) * _Actor_params[
                        _param].clone()

    def save_model(self):
        print("=============save model==============")
        save_checkpoint(self.Critic1, "./checkpoints/Critic1_checkpoint.ckpt")
        save_checkpoint(self.Critic1_, "./checkpoints/_Critic1_checkpoint.ckpt")
        save_checkpoint(self.Critic2, "./checkpoints/Critic2_checkpoint.ckpt")
        save_checkpoint(self.Critic2_, "./checkpoints/_Critic2_checkpoint.ckpt")
        save_checkpoint(self.Actor, "./checkpoints/Actor_checkpoint.ckpt")
        save_checkpoint(self.Actor_, "./checkpoints/_Actor_checkpoint.ckpt")

    def load_model(self):
        print("=============load model==============")
        load_checkpoint("./checkpoints/Critic1_checkpoint.ckpt", self.Critic1)
        load_checkpoint("./checkpoints/_Critic1_checkpoint.ckpt", self.Critic1_)
        load_checkpoint("./checkpoints/Critic2_checkpoint.ckpt", self.Critic2)
        load_checkpoint("./checkpoints/_Critic2_checkpoint.ckpt", self.Critic2_)
        load_checkpoint("./checkpoints/Actor_checkpoint.ckpt", self.Actor)
        load_checkpoint("./checkpoints/_Actor_checkpoint.ckpt", self.Actor_)
