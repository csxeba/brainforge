from collections import deque

import numpy as np
import gym

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.optimization import SGD
from brainforge.reinforcement import PG, AgentConfig
from matplotlib import pyplot

env = gym.make("CartPole-v1")
nactions = env.action_space.n


def get_agent():
    brain = BackpropNetwork(input_shape=env.observation_space.shape, layerstack=[
        DenseLayer(nactions, activation="softmax")
    ], cost="xent", optimizer=SGD(eta=0.0001))
    return brain


def run(agent):

    episode = 1
    wins = 0
    rewards = deque(maxlen=100)

    while 1:
        state = env.reset()
        done = False
        steps = 1
        reward = None
        while not done:
            # env.render()
            # print(f"\rStep {steps:>4}", end="")
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            steps += 1

        rewards.append(steps)
        win = steps > 145
        cost = agent.accumulate(state, 10. if win else -1.)
        # agent.push_weights()
        meanrwd = np.mean(rewards)
        print(f"\rEpisode {episode:>6}, running reward: {meanrwd:.2f}," +
              f" Cost: {cost:>6.4f}, Epsilon: {agent.cfg.epsilon:>6.4f}",
              end="")
        # if episode % 100 == 0:
        #     print(" Pulled pork")
        #     agent.pull_weights()
        if win:
            print(" Win!")
            wins += 1
        else:
            wins = 0
        if wins >= 50:
            break
        episode += 1
    print("\n\n")
    print("-" * 50)
    print("Environment solved!")


def plotrun(agent):
    rewards = []
    rwmean = []
    EPISODES = 1000
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
    pyplot.show()


if __name__ == '__main__':
    plotrun(PG(get_agent(), nactions, AgentConfig(
        epsilon_greedy_rate=1.0, epsilon_decay_factor=0.9998, epsilon_min=0.0,
        discount_factor=0.6, replay_memory_size=1800, training_batch_size=180,
    )))
