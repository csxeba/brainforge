import gym

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.reinforcement import DeepQLearning

env = gym.make("CartPole-v1")

brain = Network(env.observation_space.shape, layers=[
    DenseLayer(300, activation="tanh"),
    DenseLayer(env.action_space.n, activation="softmax")
])
brain.finalize("xent", "adam")
agent = DeepQLearning(brain, env.action_space.n, discount_factor=0.)

reward_running = None
episode = 1

while 1:
    state = env.reset()
    done = False
    steps = 1
    reward = None
    reward_sum = 0.
    while not done:
        env.render()
        print(f"\rStep {steps:>4}: rwd_sum: {reward_sum:.2f}", end="")
        action = agent.sample(state, reward)
        state, reward, done, info = env.step(action)
        reward_sum += reward
        steps += 1
    print()
    agent.accumulate(reward)
    reward_running = reward_sum if reward_running is None else \
        (0.1 * reward_sum + 0.9 * reward_running)
    print(f"Episode {episode:>2}, running reward: {reward_running:.2f}")
    episode += 1
    if episode % 50 == 0:
        print("Performed knowledge transfer!")
        agent.update()
