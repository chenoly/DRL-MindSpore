import gym
from DQN import Agent


def Main(epoch=102400):
    env = gym.make('MountainCar-v0')
    agent = Agent(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0], batchsize=128)
    state = env.reset()
    for i in range(epoch):
        rewards = []
        action = agent.choose(state)
        state_, reward, done, info = env.step(action)
        agent.store(state, action, state_, reward, done)
        state = state_
        rewards.append(reward)
        agent.train((i + 1))
        env.render()
    agent.save_model()
Main()
