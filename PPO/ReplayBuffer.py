import numpy as np
import random


class ReBuffer:
    def __init__(self, batchsize=64, maxbuffersize=124000):
        self.batchsize = batchsize
        self.maxbuffersize = maxbuffersize
        self.index_done = 0
        self.index_done_ = 0
        self.num = 0
        self.state = []
        self.reward = []
        self.done = []

    def store(self, state, reward, done):
        if self.num < self.maxbuffersize:
            self.state.append(state)
            self.reward.append(reward)
            self.done.append(done + 0)
            self.num += 1
            if done:
                self.index_done = self.index_done_
                self.index_done_ = self.num

    def retract_(self):
        if self.num == 0:
            return
        elif self.num <= self.batchsize:
            state = (np.array(self.state))
            reward = (np.array(self.reward))
            done = (np.array(self.done))
            return state, reward, done
        else:
            index = random.sample(range(0, self.num), self.batchsize)
            state = (np.array(self.state)[index])
            reward = (np.array(self.reward)[index])
            done = (np.array(self.done)[index])
            return state, reward, done

    def retract(self):
        state = (np.array(self.state)[self.index_done:])
        reward = (np.array(self.reward)[self.index_done:])
        done = (np.array(self.done)[self.index_done:])
        return state, reward, done