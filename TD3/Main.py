import gym
from mindspore import context
from TD3 import Agent
import numpy as np
import matplotlib.pyplot as plt

context.set_context(device_target="CPU", device_id=0)


def Main(epoch=10000):
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(2, env.observation_space.shape[0], batchsize=64, tau=0.005)
    # agent.save_model()
    allrewards = []
    for i in range(epoch):
        rewards = []
        done = False
        state = env.reset()
        while not done:
            action = agent.choose(state)
            state_, reward, done, info = env.step(action)
            agent.store(state, action, state_, reward, done)
            state = state_
            rewards.append(reward)
            agent.train((i + 1))
            env.render()
        allrewards.append(np.sum(np.array(rewards)))
        print("epoch:", (i + 1), "this epoch reward:", np.sum(np.array(rewards)), "ave reward:",
              np.average(allrewards))
        # if np.sum(np.array(rewards)) > np.average(allrewards):
        #     agent.save_model()
    agent.save_model()
    x = [(i + 1) for i in range(len(allrewards))]
    y = allrewards
    plt.plot(x, y)
    plt.title("Reward")
    plt.xlabel("epoch")
    plt.ylabel("aveReward")
    plt.savefig("p.png")


Main()
