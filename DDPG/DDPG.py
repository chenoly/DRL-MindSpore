import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import save_checkpoint, load_checkpoint
from mindspore import Tensor
from LossFunc import DGradMSELoss, AGradMSELoss
from Model import Critic, Actor
from ReplayBuffer import ReBuffer
from MyTrainOneStepCell import CriticGradWrap, ActorGradWrap, CriticWithLossCell, ActorWithLossCell


class Agent:
    '''
    agent
    '''

    def __init__(self, action_dim, state_dim, gamma=0.9, batchsize=64, lr=0.002, tau=0.005, train_epoch=1):
        '''
        agent
        :param action_dim: action's space size
        :param state_dim: state's space size
        :param batchsize: training batchsize
        :param lr: learning rate
        :param tau: softupdate param
        '''
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.train_epoch = train_epoch
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.Buffer = ReBuffer(batchsize=batchsize, maxbuffersize=102400)
        self.Critic = Critic(action_dim=action_dim, state_dim=state_dim)
        self.Critic_ = Critic(action_dim=action_dim, state_dim=state_dim)
        self.Actor = Actor(action_dim=action_dim, state_dim=state_dim)
        self.Actor_ = Actor(action_dim=action_dim, state_dim=state_dim)
        self.AGradCritican = AGradMSELoss()
        self.DGradCritican = DGradMSELoss()
        self.Critic_Loss = CriticWithLossCell(critic=self.Critic, loss_fn=self.DGradCritican)
        self.Actor_Loss = ActorWithLossCell(critic=self.Critic, actor=self.Actor, loss_fn=self.AGradCritican)
        self.Criticoptimizer = nn.Adam(self.Critic.trainable_params(), learning_rate=self.lr)
        self.Actoroptimizer = nn.Adam(self.Actor.trainable_params(), learning_rate=self.lr)
        self.Critic_train = CriticGradWrap(loss_network=self.Critic_Loss, Critic_network=self.Critic)
        self.Actor_train = ActorGradWrap(loss_network=self.Actor_Loss, Actor_network=self.Actor)

    def choose(self, state):
        '''
        choose a action by using Actor network,math define a = policy(s)
        :param state:
        :return:
        '''
        epsilon = np.random.normal(loc=0, scale=0.5, size=(1, self.action_dim))
        epsilon = Tensor.from_numpy(epsilon)
        state = Tensor.from_numpy(state).reshape(1, self.state_dim)
        action = self.Actor(state)
        A = action + epsilon
        return A.asnumpy().flatten()

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

    def train(self):
        '''
        train critic and actor networks
        :return:none
        '''
        if self.Buffer.num >= self.Buffer.batchsize:
            for n in range(self.train_epoch):
                self.Critic_train.set_train()
                self.Actor_train.set_train()
                state, action, state_, reward, done = self.Buffer.retract_()
                state = Tensor.from_numpy(state).astype(dtype=mindspore.float32)
                action = Tensor.from_numpy(action).astype(dtype=mindspore.float32)
                state_ = Tensor.from_numpy(state_).astype(dtype=mindspore.float32)
                reward = Tensor.from_numpy(reward).astype(dtype=mindspore.float32)
                done = Tensor.from_numpy(done).astype(dtype=mindspore.float32)
                action_ = self.Actor_(state_)
                y = reward + (1 - done) * self.gamma * self.Critic_(state_, action_)

                critic_grad = self.Critic_train(state, action, y)
                self.Criticoptimizer(critic_grad)
                actor_grad = self.Actor_train(state, Tensor(0.0, mindspore.float32))
                self.Actoroptimizer(actor_grad)

                Critic_params = self.Critic.parameters_dict()
                _Critic_params = self.Critic_.parameters_dict()
                Actor_params = self.Actor.parameters_dict()
                _Actor_params = self.Actor_.parameters_dict()
                for param, _param in zip(Critic_params, _Critic_params):
                    _Critic_params[_param] = self.tau * Critic_params[param].clone() + (1 - self.tau) * _Critic_params[_param].clone()
                for param, _param in zip(Actor_params, _Actor_params):
                    _Actor_params[_param] = self.tau * Actor_params[param].clone() + (1 - self.tau) * _Actor_params[_param].clone()

    def save_model(self):
        print("=============save model==============")
        save_checkpoint(self.Critic, "./checkpoints/_Critic_checkpoint.ckpt")
        save_checkpoint(self.Critic_, "./checkpoints/_Critic_checkpoint.ckpt")
        save_checkpoint(self.Actor, "./checkpoints/_Actor_checkpoint.ckpt")
        save_checkpoint(self.Actor_, "./checkpoints/_Actor_checkpoint.ckpt")

    def load_model(self):
        print("=============load model==============")
        load_checkpoint("./checkpoints/Critic_checkpoint.ckpt", self.Critic)
        load_checkpoint("./checkpoints/_Critic_checkpoint.ckpt", self.Critic_)
        load_checkpoint("./checkpoints/Actor_checkpoint.ckpt", self.Actor)
        load_checkpoint("./checkpoints/_Actor_checkpoint.ckpt", self.Actor_)
