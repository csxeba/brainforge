import gym

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.reinforcement import DQN, PG
from brainforge.optimizers import Adam

env = gym.make("CartPole-v0")
nactions = env.action_space.n


def Qann():
    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(nactions, activation="linear")
    ])
    brain.finalize("mse", Adam(brain.nparams, eta=0.001))
    return brain


def PGann():
    brain = Network(env.observation_space.shape, layers=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(nactions, activation="softmax")
    ])
    brain.finalize("xent", Adam(brain.nparams, eta=0.001))
    return brain


def keras():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    brain = Sequential([
        Dense(30, input_dim=env.observation_space.shape[0], activation="tanh"),
        Dense(nactions, activation="linear")
    ])
    brain.compile(Adam(), "mse")
    return brain


def run(agent):

    reward_running = None
    episode = 1
    wins = 0

    while 1:
        state = env.reset()
        done = False
        steps = 1
        reward = None
        reward_sum = 0.
        while not done:
            # env.render()
            # print(f"\rStep {steps:>4}", end="")
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            steps += 1

        cost = agent.accumulate(state, -10.)
        reward_running = reward_sum if reward_running is None else \
            (0.1 * reward_sum + 0.9 * reward_running)
        # print()
        print(f"\rEpisode {episode:>6}, running reward: {reward_running:.2f}, Cost: {cost:>6.4f}",
              end="")
        if episode % 10 == 0:
            agent.update()
        if episode % 1000 == 0:
            print()
            # print()
            # print("Pulled weights!")
            # agent.pull_weights()
        episode += 1
        if reward_running >= 145:
            print(" Win!")
            wins += 1
        else:
            wins = 0
        if wins >= 100:
            print("Environment solved!")
            env.render()
            break


if __name__ == '__main__':
    run(PG(PGann(), nactions))
