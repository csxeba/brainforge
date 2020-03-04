from collections import deque

from matplotlib import pyplot as plt

from grund import getout

from brainforge import Backpropagation
from brainforge.layers import Dense, Flatten
from brainforge.reinforcement import DQN, AgentConfig

env = getout.GetOut((10, 10))
agent = DQN(
    Backpropagation(input_shape=env.neurons_required[0], layerstack=[
        Flatten(),
        Dense(30, activation="tanh"),
        Dense(env.neurons_required[-1], activation="linear")
    ], cost="mse", optimizer="rmsprop"), len(env.actions),
    AgentConfig(epsilon=1., epsilon_decay=0.999, epsilon_min=0.1)
)

render = False

if render:
    frame = env.reset()
    plt.ion()
    obj = plt.imshow(frame, vmin=-1, vmax=1)

rwds = deque(maxlen=100)
episodes = 1
while 1:
    done = False
    reward = None
    state = env.reset()
    rwsum = 0.
    while not done:
        if render:
            obj.set_data(state)
            plt.pause(0.1)
        action = agent.sample(state, reward)
        state, reward, done = env.step(action)
        rwsum += reward
    rwds.append(rwsum)
    cost = agent.accumulate(state, reward)
    print(f"\rEpisode {episodes:>6}, mean_rwd: {sum(rwds)/len(rwds):.2f} cost: {cost:.4f},"
          + f" e: {agent.cfg.epsilon:.4f}", end="")
    episodes += 1
