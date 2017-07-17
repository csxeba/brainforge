from collections import deque

import numpy as np
import gym

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.reinforcement import DQN as AgentType, AgentConfig
from matplotlib import pyplot

env = gym.make("CartPole-v1")
nactions = env.action_space.n


def Qann():
    from optimization.gradient_descent import Momentum as Opt

    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(24, activation="relu"),
        DenseLayer(24, activation="relu"),
        DenseLayer(nactions, activation="linear")
    ])
    brain.finalize("mse", Opt(brain.nparams, eta=0.0001, mu=0.9))
    return brain


def PGann():
    from optimization.gradient_descent import SGD as Opt

    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(24, activation="tanh"),
        DenseLayer(nactions, activation="softmax")
    ])
    brain.finalize("xent", Opt(brain.nparams, eta=0.0001))
    return brain


def keras():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam as Optm
    brain = Sequential([
        Dense(24, input_dim=env.observation_space.shape[0], activation="relu"),
        Dense(24, activation="relu"),
        Dense(nactions, activation="linear")
    ])
    brain.compile(Optm(lr=0.001), "mse")
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
        agent.push_weights()
        meanrwd = np.mean(rewards)
        print(f"\rEpisode {episode:>6}, running reward: {meanrwd:.2f}," +
              f" Cost: {cost:>6.4f}, Epsilon: {agent.cfg.epsilon:>6.4f}",
              end="")
        if episode % 100 == 0:
            print(" Pulled pork")
            agent.pull_weights()
        if win:
            print(" Win!")
            wins += 1
        else:
            wins = 0
        if wins > 100:
            break
        episode += 1
    print("\n\n")
    print("-" * 50)
    print("Environment solved!")


def plotrun(agent):
    rewards = []
    for episode in range(1, 1001):
        state = env.reset()
        reward_sum = 0.
        reward = 0.
        win = False
        for time in range(146):
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
        else:
            win = True
        rewards.append(reward_sum)
        cost = agent.accumulate(state, (-1. if not win else 1.))
        agent.push_weights()
        if episode % 100 == 0:
            agent.pull_weights()
        print(f"\r{episode / 1000:.1%}, Cost: {cost:8>.4f}, Epsilon: {agent.cfg.epsilon:8.6f}", end="")
    print()
    pyplot.plot(range(len(rewards)), rewards)
    pyplot.show()


if __name__ == '__main__':
    run(AgentType(Qann(), nactions, AgentConfig(
        epsilon_greedy_rate=1., epsilon_decay_factor=0.999, epsilon_min=0.01,
        discount_factor=0.9, replay_memory_size=180, training_batch_size=180
    )))
