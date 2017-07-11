from collections import deque

import numpy as np
import gym

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.reinforcement import DQN as AgentType, AgentConfig
from brainforge.optimizers import Adam as Opt

env = gym.make("CartPole-v1")
nactions = env.action_space.n


def Qann():
    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(24, activation="relu"),
        DenseLayer(24, activation="relu"),
        DenseLayer(nactions, activation="linear")
    ])
    brain.finalize("mse", Opt(brain.nparams, eta=0.001))
    return brain


def PGann():
    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(60, activation="tanh"),
        DenseLayer(nactions, activation="softmax")
    ])
    brain.finalize("xent", Opt(brain.nparams, eta=0.01))
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

    while 1:
        state = env.reset()
        done = False
        steps = 1
        reward = None
        rewards = deque(maxlen=100)
        while not done:
            # env.render()
            # print(f"\rStep {steps:>4}", end="")
            action = agent.sample(state, 0.)
            state, reward, done, info = env.step(action)
            steps += 1

        rewards.append(steps)
        win = steps > 145
        cost = agent.accumulate(state, 10. if win else -1.)
        meanrwd = np.mean(rewards)
        print(f"\rEpisode {episode:>6}, running reward: {meanrwd:.2f}, Cost: {cost:>6.4f}",
              end="")
        if episode % 1000 == 0:
            print(" Pulled pork")
            # agent.pull_weights()
        episode += 1
        if win:
            print(" Win!")
            wins += 1
        else:
            wins = 0
        if wins >= 100:
            print("Environment solved!")
            env.render()
            break


if __name__ == '__main__':
    run(AgentType(Qann(), nactions, AgentConfig(epsilon_greedy_rate=0.01,
                                                discount_factor=0.95,
                                                replay_memory_size=10000,
                                                training_batch_size=1000)))
