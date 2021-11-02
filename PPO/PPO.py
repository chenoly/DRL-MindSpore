import copy

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore.nn.probability.distribution import Categorical
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

    def __init__(self, action_dim, state_dim, gamma=0.9,epsilon=0.2, batchsize=64, lr=0.002, tau=0.005, train_epoch=1):
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
        self.epsilon = epsilon
        self.Buffer = ReBuffer(batchsize=batchsize, maxbuffersize=102400)
        self.Critic = Critic(state_dim=state_dim)
        self.Actor = Actor(action_dim=action_dim, state_dim=state_dim)
        self.Actor_ = copy.deepcopy(self.Actor)
        self.Critic_Loss = CriticWithLossCell(critic=self.Critic)
        self.Actor_Loss = ActorWithLossCell(actor=self.Actor,actor_=self.Actor)
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
        state_ = Tensor(state).reshape(1,-1)
        action = self.Actor(state_)
        cat = Categorical(action,dtype=mindspore.float32)
        index = cat.sample()
        return index.asnumpy().flatten()

    def store(self, state, reward, done):
        '''
        store a transition to Buffer
        :param state: precent state
        :param action: a action by precent state
        :param state_: net state
        :param reward: reward by playing a action
        :param done: game over or not
        :return: none
        '''
        self.Buffer.store(state, reward, done)

    def train(self):
        '''
        train critic and actor networks
        :return:none
        '''

        if self.Buffer.num >= self.Buffer.batchsize:

            self.Critic_train.set_train()
            self.Actor_train.set_train()
            state, rewards_, done = self.Buffer.retract_()
            state = Tensor.from_numpy(state).astype(dtype=mindspore.float32)

            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards_), reversed(done)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            # print(rewards)
            rewards = Tensor(rewards, dtype=mindspore.float32)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            advantages_reward = rewards - self.Critic(state)

            for n in range(self.train_epoch):
                actor_grad = self.Actor_train(state,advantages_reward,self.epsilon)
                self.Actoroptimizer(actor_grad)
                critic_grad = self.Critic_train(state,advantages_reward)
                self.Criticoptimizer(critic_grad)

            Actor_params = self.Actor.parameters_dict()
            _Actor_params = self.Actor_.parameters_dict()
            for param, _param in zip(Actor_params, _Actor_params):
                _Actor_params[_param] = Actor_params[param].clone()

    def save_model(self):
        print("=============save model==============")
        save_checkpoint(self.Critic, "./checkpoints/_Critic_checkpoint.ckpt")
        save_checkpoint(self.Actor, "./checkpoints/_Actor_checkpoint.ckpt")
        save_checkpoint(self.Actor_, "./checkpoints/_Actor_checkpoint.ckpt")

    def load_model(self):
        print("=============load model==============")
        load_checkpoint("./checkpoints/Critic_checkpoint.ckpt", self.Critic)
        load_checkpoint("./checkpoints/Actor_checkpoint.ckpt", self.Actor)
        load_checkpoint("./checkpoints/_Actor_checkpoint.ckpt", self.Actor_)
