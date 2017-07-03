import random as random
from collections import deque
import argparse
import logging

import tensorflow as tf
import numpy as np
import gym

DEFAULT_EPISODES = 2000
DEFAULT_STEPS = 500
DEFAULT_ENVIRONMENT = 'CartPole-v0'

DEFAULT_MEMORY_CAPACITY = 10000
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_MINI_BATCH_SIZE = 10

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_REGULARIZATION = 0.001
DEFAULT_NUM_HIDDEN = 2  # not used in tensorflow implementation
DEFAULT_HIDDEN_SIZE = 20

DEFAULT_ID = 0


# noinspection PyAttributeOutsideInit
class CNN:
    """Convolutional Neural Network model."""

    def __init__(self, num_actions, observation_shape, params={}, verbose=False):
        """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """

        self.verbose = verbose
        self.num_actions = num_actions

        # observation shape will be a tuple
        self.observation_shape = observation_shape[0]
        logging.info('Initialized with params: {}'.format(params))

        self.lr = params['lr']
        self.reg = params['reg']
        self.num_hidden = params['num_hidden']
        self.hidden_size = params['hidden_size']

        self.session = self.create_model()

    def add_placeholders(self):
        input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
        labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
        actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

        return input_placeholder, labels_placeholder, actions_placeholder

    def nn(self, input_obs):
        with tf.name_scope("Layer1"):
            W1shape = [self.observation_shape, self.hidden_size]
            W1 = tf.get_variable("W1", shape=W1shape, )
            bshape = [1, self.hidden_size]
            b1 = tf.get_variable("b1", shape=bshape, initializer=tf.constant_initializer(0.0))

        with tf.name_scope("Layer2"):
            W2shape = [self.hidden_size, self.hidden_size]
            W2 = tf.get_variable("W2", shape=W2shape, )
            bshape = [1, self.hidden_size]
            b2 = tf.get_variable("b2", shape=bshape, initializer=tf.constant_initializer(0.0))

        with tf.name_scope("OutputLayer"):
            Ushape = [self.hidden_size, self.num_actions]
            U = tf.get_variable("U", shape=Ushape)
            b3shape = [1, self.num_actions]
            b3 = tf.get_variable("b3", shape=b3shape, initializer=tf.constant_initializer(0.0))

        xW = tf.matmul(input_obs, W1)
        h = tf.tanh(tf.add(xW, b1))

        xW = tf.matmul(h, W2)
        h = tf.tanh(tf.add(xW, b2))

        hU = tf.matmul(h, U)
        out = tf.add(hU, b3)

        reg = self.reg * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(U)))
        return out, reg

    def create_model(self):
        """
        The model definition.
        """
        self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
        outputs, reg = self.nn(self.input_placeholder)
        self.predictions = outputs

        self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)

        self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        self.train_op = optimizer.minimize(self.loss)
        init = tf.initialize_all_variables()
        session = tf.Session()
        session.run(init)

        return session

    def train_step(self, Xs, ys, actions):
        """
    Updates the CNN model with a mini batch of training examples.
    """

        loss, _, prediction_probs, q_values = self.session.run(
            [self.loss, self.train_op, self.predictions, self.q_vals],
            feed_dict={self.input_placeholder: Xs,
                       self.labels_placeholder: ys,
                       self.actions_placeholder: actions
                       })

    def predict(self, observation):
        """
    Predicts the rewards for an input observation state.
    Args:
      observation: a numpy array of a single observation state
    """

        loss, prediction_probs = self.session.run(
            [self.loss, self.predictions],
            feed_dict={self.input_placeholder: observation,
                       self.labels_placeholder: np.zeros(len(observation)),
                       self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                       })

        return prediction_probs


class DQN:
    def __init__(self, num_actions, observation_shape, dqn_params, cnn_params):
        self.num_actions = num_actions
        self.epsilon = dqn_params['epsilon']
        self.gamma = dqn_params['gamma']
        self.mini_batch_size = dqn_params['mini_batch_size']

        # memory
        self.memory = deque(maxlen=dqn_params['memory_capacity'])

        # initialize network
        self.model = CNN(num_actions, observation_shape, cnn_params)
        print("model initialized")

    def select_action(self, observation):
        """
    Selects the next action to take based on the current state and learned Q.
    Args:
      observation: the current state
    """

        if random.random() < self.epsilon:
            # with epsilon probability select a random action
            action = np.random.randint(0, self.num_actions)
        else:
            # select the action a which maximizes the Q value
            obs = np.array([observation])
            q_values = self.model.predict(obs)
            action = np.argmax(q_values)

        return action

    def update_state(self, action, observation, new_observation, reward, done):
        """
    Stores the most recent action in the replay memory.
    Args:
      action: the action taken
      observation: the state before the action was taken
      new_observation: the state after the action is taken
      reward: the reward from the action
      done: a boolean for when the episode has terminated
    """
        transition = {'action': action,
                      'observation': observation,
                      'new_observation': new_observation,
                      'reward': reward,
                      'is_done': done}
        self.memory.append(transition)

    def get_random_mini_batch(self):
        """
    Gets a random sample of transitions from the replay memory.
    """
        rand_idxs = random.sample(range(len(self.memory)), self.mini_batch_size)
        mini_batch = []
        for idx in rand_idxs:
            mini_batch.append(self.memory[idx])

        return mini_batch

    def train_step(self):
        """
    Updates the model based on the mini batch
    """
        if len(self.memory) > self.mini_batch_size:
            mini_batch = self.get_random_mini_batch()

            Xs = []
            ys = []
            actions = []

            for sample in mini_batch:
                y_j = sample['reward']

                # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term to y_j
                if not sample['is_done']:
                    new_observation = sample['new_observation']
                    new_obs = np.array([new_observation])
                    q_new_values = self.model.predict(new_obs)
                    action = np.max(q_new_values)
                    y_j += self.gamma * action

                action = np.zeros(self.num_actions)
                action[sample['action']] = 1

                observation = sample['observation']

                Xs.append(observation.copy())
                ys.append(y_j)
                actions.append(action.copy())

            Xs = np.array(Xs)
            ys = np.array(ys)
            actions = np.array(actions)

            self.model.train_step(Xs, ys, actions)


def parse_args():
    """
  Parses the command line input.
  """
    parser = argparse.ArgumentParser()
    parser.add_argument('-episodes', default=DEFAULT_EPISODES, help='number of episodes', type=int)
    parser.add_argument('-steps', default=DEFAULT_STEPS, help='number of steps', type=int)
    parser.add_argument('-env', default=DEFAULT_ENVIRONMENT, help='environment name', type=str)
    parser.add_argument('-id', default=DEFAULT_ID, help='id number of run to append to output file name', type=str)

    parser.add_argument('-capacity', default=DEFAULT_MEMORY_CAPACITY, help='memory capacity', type=int)
    parser.add_argument('-epsilon', default=DEFAULT_EPSILON,
                        help='epsilon value for the probability of taking a random action', type=float)
    parser.add_argument('-gamma', default=DEFAULT_GAMMA,
                        help='gamma value for the contribution of the Q function in learning', type=float)
    parser.add_argument('-minibatch_size', default=DEFAULT_MINI_BATCH_SIZE, help='mini batch size for training',
                        type=int)

    parser.add_argument('-l', default=DEFAULT_LEARNING_RATE, help='learning rate', type=float)
    parser.add_argument('-r', default=DEFAULT_REGULARIZATION, help='regularization', type=float)
    parser.add_argument('-num_hidden', default=DEFAULT_NUM_HIDDEN,
                        help='the number of hidden layers in the deep network', type=int)
    parser.add_argument('-hidden_size', default=DEFAULT_HIDDEN_SIZE,
                        help='the hidden size of all layers in the network', type=int)

    args = parser.parse_args()

    run_id = "lr_" + str(args.l) + "_reg_" + str(args.r) + "_h_" + str(args.hidden_size) + "_m_" + str(
        args.minibatch_size) + "_c_" + str(args.capacity) + "_id_" + str(args.id)

    agent_params = {'episodes': args.episodes, 'steps': args.steps, 'environment': args.env, 'run_id': run_id}
    dqn_params = {'memory_capacity': args.capacity, 'epsilon': args.epsilon, 'gamma': args.gamma,
                  'mini_batch_size': args.minibatch_size}
    cnn_params = {'lr': args.l, 'reg': args.r, 'num_hidden': args.num_hidden, 'hidden_size': args.hidden_size,
                  'mini_batch_size': args.minibatch_size}

    return agent_params, dqn_params, cnn_params


def run_dqn():
    # get command line arguments, defaults set in utils.py
    agent_params, dqn_params, cnn_params = parse_args()

    env = gym.make(agent_params['environment'])
    episodes = agent_params['episodes']
    steps = agent_params['steps']
    num_actions = env.action_space.n
    observation_shape = env.observation_space.shape

    # initialize dqn learning
    dqn = DQN(num_actions, observation_shape, dqn_params, cnn_params)

    last_100 = deque(maxlen=100)

    for i_episode in range(episodes):
        observation = env.reset()
        reward_sum = 0

        if np.mean(last_100) > 200:
            break

        for t in range(steps):
            # env.render()
            # print observation

            # select action based on the model
            action = dqn.select_action(observation)
            # execute actin in emulator
            new_observation, reward, done, _ = env.step(action)
            # update the state
            dqn.update_state(action, observation, new_observation, reward, done)
            observation = new_observation

            # train the model
            dqn.train_step()

            reward_sum += reward
            if done:
                print("Episode ", i_episode)
                print("Finished after {} timesteps".format(t + 1))
                print("Reward for this episode: ", reward_sum)
                last_100.append(reward_sum)
                print("Average reward for last 100 episodes: ", np.mean(last_100))
                break


if __name__ == '__main__':
    run_dqn()
