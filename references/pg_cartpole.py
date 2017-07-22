# coding: utf-8

# This code works using a vanilla version of the policy-gradient method.

import numpy as np
import gym

env = gym.make('CartPole-v0')


# Play with policy gradient agent, with given parameter vector
# - num_episodes: the number of episodes to run the agent
# - theta: the parameter to use for the policy
# - max_episode_length: the maximum length of an episode
def policy_gradient_agent(num_episodes, theta, max_episode_length, render=False):
    for i_episode in range(num_episodes):
        episode_rewards, _, _ = run_episode(theta, max_episode_length, render)
        print("TEST: Reward for episode:", sum(episode_rewards))


# Train an agent using policy gradients. Each episode, we sample a trajectory,
# and then estimate the gradient of the expected reward with respect to theta.
# We then update theta in the direction of the gradient.
# - num_episodes: the number of episodes to train for
# - max_episode_length: the maximum length of an episode
# - initial_step_size: the initial step size. We decrease the step size
# proportional to 1/n, where n is the episode number
def train_policy_gradient_agent(num_episodes, max_episode_length,
                                initial_step_size):
    # Initialise theta
    theta = np.random.randn(1, 4)

    for i_episode in range(num_episodes):
        # Run an episode with our current policy
        episode_rewards, episode_actions, episode_observations = \
            run_episode(theta, max_episode_length)
        print("TRAIN: Reward for episode:", sum(episode_rewards))

        # Compute the policy gradient for this trajectory
        policy_gradient = compute_policy_gradient(episode_rewards,
                                                  episode_actions, episode_observations, theta)

        # Vanilla gradient ascent
        # We decrease the step size proportional to 1/i_episode
        step_size = initial_step_size / (1 + i_episode)
        theta = theta + step_size * policy_gradient

    # Return our trained theta
    return theta


# observation and theta are both row vectors.
# We want to find theta such that observation . theta > 0 is a good predictor
# for the 'move right' action.
def compute_policy(observation, theta):
    # We compute the dot product of our observation with theta, and then apply
    # the sigmoid function, to get the probability of moving right, denoted
    # prob_right. The probability of moving left is then 1 - prob_right.

    prob_right = sigmoid(np.dot(observation, np.transpose(theta)))
    return [1 - prob_right, prob_right]


# Samples an action from the policy
# observation: an observation from the environment
# theta: the parameter vector theta
# Returns: a sample from the policy distribution. The distribution is: move
# right with probability sigma(x dot theta), and otherwise move left.
def sample_action(observation, theta):
    prob_right = compute_policy(observation, theta)[1]
    r = np.random.rand()
    if r < prob_right:
        return 1
    else:
        return 0


# Computes the sigmoid function
# u: a real number
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


# This function computes the gradient of the policy with respect to theta for
# the specified trajectory.
# - episode_rewards: the rewards of the episode
# - episode_actions: the actions of the episode
# - episode_observations: the observations of the episode
# - theta: the parameter for the policy that ran the episode
def compute_policy_gradient(episode_rewards, episode_actions,
                            episode_observations, theta):
    # The gradient computation is explained at https://cgnicholls.github.io

    # Compute the grad_theta log P(tau | pi; theta), i.e. the gradient with
    # respect to theta. This is the sum of grad_theta log pi(a_t | x_t; theta) *
    # R, for each timestep t in the episode, where R is the total reward for the
    # episode and x_t is the observation at time t.

    # One can show that grad_theta log pi(a_L | x_t; theta) = - pi(a_R | x_t;
    # theta) x_t, and grad_theta log pi(a_R | x_t; theta) = pi(a_L | x_t; theta)
    # x_t.
    gradient = 0
    episode_length = len(episode_rewards)
    for t in range(episode_length):
        # Compute the policy on the observation and theta.
        pi = compute_policy(episode_observations[t], theta)
        a_t = episode_actions[t]
        v_t = sum(episode_rewards[t::])
        if a_t == 0:
            grad_theta_log_pi = - pi[1] * episode_observations[t] * v_t
        else:
            grad_theta_log_pi = pi[0] * episode_observations[t] * v_t

        gradient = gradient + grad_theta_log_pi
    return gradient


# Run an episode with the policy parametrised by theta.
# - theta: the parameter to use for the policy
# - max_episode_length: the maximum length of an episode
# - render: whether or not to show the episode
# Returns the episode rewards, episode actions and episode observations
def run_episode(theta, max_episode_length, render=False):
    # Reset the environment
    observation = env.reset()
    episode_rewards = []
    episode_actions = []
    episode_observations = []
    episode_observations.append(observation)
    for t in range(max_episode_length):
        # If rendering, draw the environment
        if render:
            env.render()
        a_t = sample_action(observation, theta)
        observation, reward, done, info = env.step(a_t)
        episode_rewards.append(reward)
        episode_observations.append(observation)
        episode_actions.append(a_t)
        if done:
            break
    return episode_rewards, episode_actions, episode_observations


# Train the agent
num_episodes = 1000
max_episode_length = 200
initial_step_size = 0.1

# Start the monitor
# env.monitor.start('/tmp/cartpole-policy-gradient')
theta = train_policy_gradient_agent(num_episodes, max_episode_length,
                                    initial_step_size)
# End the monitor
# end.monitor.close()

# Run the agent for 10 episodes
policy_gradient_agent(10, theta, max_episode_length)
