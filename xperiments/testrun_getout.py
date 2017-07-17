from collections import deque

from brainforge import Network
from brainforge.architecture import DenseLayer, Flatten
from brainforge.reinforcement import DQN, AgentConfig

from grund.getout import GetOut


RENDER = 1


def netbase(inshape):
    return Network(inshape, layers=[Flatten(), DenseLayer(60, "tanh")])


def get_qagent(environment):
    inshape, outshape = environment.neurons_required()
    net = netbase(inshape)
    net.add(DenseLayer(outshape))
    return DQN(net.finalize("mse", "momentum"), outshape, AgentConfig(
        training_batch_size=320, replay_memory_size=320,
        epsilon_decay_factor=0.999, epsilon_greedy_rate=1.0,
        discount_factor=0.65
    ))


env = GetOut((5, 5))
agent = get_qagent(env)

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
