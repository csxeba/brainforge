from collections import deque

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer, Flatten, ConvLayer, Activation, PoolLayer
from brainforge.reinforcement import DQN, AgentConfig
from brainforge.optimization import RMSprop as Opt

from grund.getout import GetOut


RENDER = 0


def netbase(inshape):
    return BackpropNetwork(
        inshape, layers=[Flatten(), DenseLayer(60, "tanh")]
    )


def convbase(inshape):
    return BackpropNetwork((1,) + inshape, layers=[
        ConvLayer(12, compiled=True), Activation("tanh"),  # 8x8x12
        ConvLayer(12, compiled=True), Activation("tanh"),  # 6x6x12
        Flatten()
    ])


def get_qagent(environment):
    inshape, outshape = environment.neurons_required()
    net = convbase(inshape)
    net.add(DenseLayer(outshape))
    return DQN(net.finalize("mse", Opt(net.nparams, eta=0.001)), outshape, AgentConfig(
        training_batch_size=180, replay_memory_size=1800,
        epsilon_decay_factor=0.999, epsilon_greedy_rate=1.0,
        epsilon_min=0.01, discount_factor=0.1
    ))


env = GetOut((10, 10))
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
            # noinspection PyUnboundLocalVariable
            obj.set_data(state.T)
            plt.pause(0.1)
        action = agent.sample(state[None, ...], reward)
        state, reward, done = env.step(action)
        rsum += reward
    rewards.append(rsum)
    cost = agent.accumulate(state, reward)
    print(f"\rEpisode: {episode:>5}, MeanRWD: {sum(rewards)/100.:.2f} " +
          f"Cost: {cost:.3f} E: {agent.cfg.epsilon:.3f}",
          end="")
    episode += 1
