import gym

from brainforge import Network
from brainforge.layers import DenseLayer, Activation
from brainforge.reinforcement import DeepQLearning, HillClimbing

env = gym.make("CartPole-v1")


def ann():
    brain = Network(env.observation_space.shape, layers=[
        Activation("tanh"),
        DenseLayer(30, activation="tanh"),
        DenseLayer(10, activation="tanh"),
        DenseLayer(env.action_space.n, activation="softmax")
    ])
    brain.finalize("xent", "adam")
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
            # print(f"\rStep {steps:>4}: rwd_sum: {reward_sum:.2f}", end="")
            action = agent.sample(state, reward)
            state, reward, done, info = env.step(action)
            reward_sum += reward
            steps += 1
        # print()
        agent.accumulate(-10)
        reward_running = reward_sum if reward_running is None else \
            (0.1 * reward_sum + 0.9 * reward_running)
        print(f"\rEpisode {episode:>6}, running reward: {reward_running:.2f}", end="")
        if episode % 100 == 0:
            # print(" Performed knowledge transfer!")
            agent.update()
        episode += 1
        if reward_running >= 145:
            env.render()
            print(" Win!")
            wins += 1
        else:
            wins = 0
        if wins >= 100:
            print("Environment solved!")
            env.render()


if __name__ == '__main__':
    run(HillClimbing(ann()))
