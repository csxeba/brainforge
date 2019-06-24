from collections import deque

import numpy as np
import gym

from matplotlib import pyplot

from brainforge.learner import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.optimizers import Momentum
from brainforge.reinforcement import PG, AgentConfig

env = gym.make("CartPole-v0")
nactions = env.action_space.n


def get_agent():
    brain = BackpropNetwork(input_shape=env.observation_space.shape, layerstack=[
        DenseLayer(nactions, activation="softmax")
    ], cost="xent", optimizer=Momentum(eta=0.001))
    return brain


def run(agent):

    episode = 1
    rewards = deque(maxlen=100)

    while 1:
        state = env.reset()
        done = False
        reward = None
        rwsum = 0.
        while not done:
            env.render()
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            rwsum += reward
        rewards.append(rwsum)
        cost = agent.accumulate(state, reward)
        meanrwd = np.mean(rewards)
        print(f"\rEpisode {episode:>6}, running reward: {meanrwd:.2f}," +
              f" Cost: {cost:>6.4f}, Epsilon: {agent.cfg.epsilon:>6.4f}",
              end="")
        episode += 1


def plotrun(agent):
    rewards = []
    rwmean = []
    EPISODES = 2000
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        reward_sum = 0.
        reward = 0.
        win = False
        for time in range(1, 151):
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
        else:
            win = True
        rewards.append(reward_sum)
        rwmean.append(np.mean(rewards[-10:]))
        cost = agent.accumulate(state, (-1. if not win else 1.))
        print(f"\r{episode / EPISODES:.1%}, Cost: {cost:8>.4f}, Epsilon: {agent.cfg.epsilon:8.6f}", end="")
    print()
    Xs = np.arange(len(rewards))
    pyplot.scatter(Xs, rewards, marker=".", s=3, c="r", alpha=0.5)
    pyplot.plot(Xs, rwmean, color="b", linewidth=2)
    pyplot.title(agent.type)
    pyplot.show()


if __name__ == '__main__':
    plotrun(PG(get_agent(), nactions, AgentConfig(
        epsilon=1.5, epsilon_decay_rate=0.9999, epsilon_min=0.,
        discount_factor=0.2, replay_memory_size=3600, training_batch_size=360,
    )))
