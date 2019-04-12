from collections import deque

from brainforge import backend as xp
import gym

from brainforge import BackpropNetwork
from brainforge.layers import Linear, ClockworkLayer
from brainforge.optimization import RMSprop
from brainforge.reinforcement import AgentConfig, DDQN as AgentType
from matplotlib import pyplot

env = gym.make("CartPole-v1")
nactions = env.action_space.n


def QannRecurrent():
    brain = BackpropNetwork(env.observation_space.shape, layers=[
        ClockworkLayer(120, activaton="tanh"),
        Linear(60, activation="relu"),
        Linear(nactions, activation="linear")
    ], cost="mse", optimizer=RMSprop(eta=0.0001))
    return brain


def QannDense():
    brain = BackpropNetwork(input_shape=env.observation_space.shape, layerstack=[
        Linear(24, activation="tanh"),
        Linear(nactions, activation="linear")
    ], cost="mse", optimizer=RMSprop(eta=0.0001))
    return brain


def run(agent, **kw):
    del kw
    episode = 1
    rewards = deque(maxlen=100)

    while 1:
        state = env.reset()
        win = False
        step = 0
        reward = None
        for step in range(1, 201):
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            if done:
                break
        else:
            win = True

        rewards.append(step)
        cost = agent.accumulate(state, 10. if win else -1.)
        meanrwd = xp.mean(rewards)
        print(f"\rEpisode {episode:>6}, running reward: {meanrwd:.2f}, Cost: {cost:>6.4f}",
              end="")
        if win:
            print(" Win!")
        episode += 1


def plotrun(agent, episodes=1000):
    rewards = []
    rwmean = []
    for episode in range(1, episodes + 1):
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
        rwmean.append(xp.mean(rewards[-10:]))
        cost = agent.accumulate(state, (-1. if not win else 1.))
        print(f"\r{episode / episodes:.1%}, Cost: {cost:8>.4f}, Epsilon: {agent.cfg.epsilon:8.6f}", end="")
    print()
    Xs = xp.arange(len(rewards))
    pyplot.scatter(Xs, rewards, marker=".", s=3, c="r", alpha=0.5)
    pyplot.plot(Xs, rwmean, color="b", linewidth=2)
    pyplot.title(agent.type)
    pyplot.show()


if __name__ == '__main__':
    plotrun(AgentType(QannDense(), nactions, AgentConfig(
        epsilon_greedy_rate=1.0, epsilon_decay_factor=0.9998, epsilon_min=0.0,
        discount_factor=0.6, replay_memory_size=7200, training_batch_size=720,
    )), episodes=2000)
