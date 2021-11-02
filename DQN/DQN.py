import copy
import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore import save_checkpoint, load_checkpoint
from LossFunc import DGradMSELoss
from Model import Eval_Net
from MyTrainOneStepCell import CriticGradWrap, CriticWithLossCell
from ReplayBuffer import ReBuffer


class Agent:
    '''
    agent
    '''

    def __init__(self, action_dim, state_dim, gamma=0.9, batchsize=64, lr=0.002, epsilon=0.9, c_epoch=20):
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
        self.c_epoch = c_epoch
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.Buffer = ReBuffer(batchsize=batchsize, maxbuffersize=102400)
        self.Critic = Eval_Net(action_dim=action_dim, state_dim=state_dim)
        self.Critic_target = copy.deepcopy(self.Critic)
        self.DGradCritican = DGradMSELoss()
        self.Critic_Loss = CriticWithLossCell(critic=self.Critic, loss_fn=self.DGradCritican)
        self.Criticoptimizer = nn.SGD(self.Critic.trainable_params(), learning_rate=self.lr)
        self.Critic_train = CriticGradWrap(loss_network=self.Critic_Loss, Critic_network=self.Critic)

    def choose(self, state):
        '''
        choose a action
        :param state:
        :return:
        '''
        if np.random.rand() < self.epsilon:
            a = self.Critic(Tensor(state, mindspore.float32).reshape(1, self.state_dim))
            action = np.argmax(a.asnumpy())
        else:
            action = np.random.randint(0, self.action_dim)
        return action

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
            self.Critic_train.set_train()
            state, action, state_, reward, done = self.Buffer.retract_()
            state = Tensor.from_numpy(state).astype(dtype=mindspore.float32)
            action = Tensor(action).astype(int)
            state_ = Tensor.from_numpy(state_).astype(dtype=mindspore.float32)
            reward = Tensor.from_numpy(reward).astype(dtype=mindspore.float32)
            done = Tensor.from_numpy(done).astype(dtype=mindspore.float32)
            y = reward + (1 - done) * self.gamma * self.Critic_target(state_).max(axis=1)
            critic_grad = self.Critic_train(state, action, y)
            self.Criticoptimizer(critic_grad)

            if t_epoch % self.c_epoch == 0:
                Critic_params = self.Critic.parameters_dict()
                _Critic_params = self.Critic_target.parameters_dict()
                for param, _param in zip(Critic_params, _Critic_params):
                    _Critic_params[_param] = Critic_params[param].clone()

    def save_model(self):
        print("=============save model==============")
        save_checkpoint(self.Critic, "./checkpoints/Critic_checkpoint.ckpt")
        save_checkpoint(self.Critic_target, "./checkpoints/_Critic_checkpoint.ckpt")

    def load_model(self):
        print("=============load model==============")
        load_checkpoint("./checkpoints/Critic_checkpoint.ckpt", self.Critic)
        load_checkpoint("./checkpoints/_Critic_checkpoint.ckpt", self.Critic_target)
