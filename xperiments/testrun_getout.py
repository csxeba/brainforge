from collections import deque

from brainforge import Network
from brainforge.layers import DenseLayer, Flatten
from brainforge.reinforcement import DQN, PG, AgentConfig

from grund.getout import GetOut


EPSILON = 0.
RENDER = 0


def netbase(inshape):
    return Network(inshape, layers=[Flatten(), DenseLayer(60, "tanh")])


def get_qagent(environment):
    inshape, outshape = environment.neurons_required()
    net = netbase(inshape)
    net.add(DenseLayer(outshape))
    return DQN(net.finalize("mse", "adam"), outshape, AgentConfig(epsilon_greedy_rate=EPSILON))


def get_pgagent(environment):
    inshape, outshape = environment.neurons_required()
    net = netbase(inshape)
    net.add(DenseLayer(outshape, "softmax"))
    return PG(net.finalize("xent", "adam"), outshape, AgentConfig(epsilon_greedy_rate=EPSILON))


env = GetOut((5, 5))
agent = get_pgagent(env)

if RENDER:
    from matplotlib import pyplot as plt
    plt.ion()
    obj = plt.imshow(env.reset().T, vmin=-1, vmax=1)

episode = 1
rewards = deque(maxlen=100)
while 1:
    done = False
    reward = None
    state = env.reset()
    rsum = 0.
    while not done:
        if RENDER:
            obj.set_data(state.T)
            plt.pause(0.1)
        action = agent.sample(state, reward)
        state, reward, done = env.step(action)
        rsum += reward
    rewards.append(rsum)
    cost = agent.accumulate(state, reward)
    print(f"\rEpisode: {episode:>5}, MeanRWD: {sum(rewards)/len(rewards):.3f} Cost: {cost:.3f}", end="")
    if episode % 10 == 0:
        agent.update()
    episode += 1
