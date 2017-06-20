import gym

from brainforge import Network
from brainforge.reinforcement import PolicyGradient

env = gym.make("CartPole-v1")
env.reset()

for _ in range(1000):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample())
