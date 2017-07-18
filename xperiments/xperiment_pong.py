from collections import deque

import gym
import numpy as np

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.reinforcement import PG, AgentConfig


def prepro_coroutine(I):

    def ds(image):
        downsmpl = image[35:195].astype(float)
        downsmpl = downsmpl[::2, ::2, 0]  # downsample by factor of 2
        downsmpl[downsmpl == 144] = 0  # erase background (background type 1)
        downsmpl[downsmpl == 109] = 0  # erase background (background type 2)
        downsmpl[downsmpl != 0] = 1.  # everything else (paddles, ball) just set to 1
        return downsmpl

    dsI = ds(I)
    pI = dsI.copy()  # type: np.ndarray
    while 1:
        I = yield (dsI - pI).ravel()
        pI = dsI
        dsI = ds(I)


RENDER = False

env = gym.make("Pong-v0")
nactions = env.action_space.n
stateshape = 6400
print("Pong stateshape =", stateshape)
brain = BackpropNetwork(stateshape, layers=[
    DenseLayer(200, activation="tanh"),
    DenseLayer(nactions, activation="softmax")
])
brain.finalize("xent", "momentum")
agent = PG(brain, nactions, AgentConfig(training_batch_size=3000, discount_factor=0.99,
                                        epsilon_greedy_rate=1., epsilon_decay=0.99,
                                        epsilon_min=0.01, replay_memory_size=3000))
rwds = deque(maxlen=100)
episode = 1
print(f"Episode {episode:>5}")
while 1:
    prepro = prepro_coroutine(env.reset())
    next(prepro)
    state = prepro.send(env.reset())
    done = False
    reward = None
    rwd_sum = 0.
    step = 1
    while not done:
        print(f"\rStep {step:>8}", end="")
        if RENDER:
            env.render()
        action = agent.sample(state, reward)
        state, reward, done, info = env.step(action)
        state = prepro.send(state)
        rwd_sum += reward
        step += 1
    rwds.append(rwd_sum)
    print()
    cost = agent.accumulate(state, reward)
    agent.push_weights()
    rwd_mean = sum(rwds) / len(rwds)
    if episode % 10 == 0:
        agent.pull_weights()
    episode += 1
    print(f"\nEpisode {episode:>5} Rwd: {rwd_mean:>5.2f} E: {agent.cfg.epsilon}")
