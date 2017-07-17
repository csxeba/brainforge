import abc

import numpy as np

from ..util.rl_util import Experience


class AgentConfig:

    def __init__(self, training_batch_size=300,
                 discount_factor=0.99,
                 knowledge_transfer_rate=0.1,
                 epsilon_greedy_rate=0.9,
                 epsilon_min=0.01,
                 epsilon_decay_factor=1.0,
                 replay_memory_size=9000):
        self.bsize = training_batch_size
        self.gamma = discount_factor
        self.tau = knowledge_transfer_rate
        self._epsilon = epsilon_greedy_rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay_factor
        self.xpsize = replay_memory_size

    @staticmethod
    def alias(item):
        return {"training_batch_size": "bsize",
                "discount_factor": "gamma",
                "knowledge_transfer_rate": "tau",
                "epsilon_greedy_rate": "epsilon",
                "replay_memory_size": "xpsize",
                "bsize": "bsize", "gamma": "gamma",
                "tau": "tau", "xpsize": "xpsize",
                "epsilon": "epsilon"}[item]

    @property
    def epsilon(self):
        if self._epsilon > self.epsilon_min:
            self._epsilon *= self.epsilon_decay
            return self._epsilon
        else:
            return self.epsilon_min

    def __getitem__(self, item):
        return self.__dict__[self.alias(item)]

    def __setitem__(self, key, value):
        self.__dict__[self.alias(key)] = value


class AgentBase(abc.ABC):

    type = ""

    def __init__(self, network, agentconfig, **kw):
        if agentconfig is None:
            agentconfig = AgentConfig(**kw)
        self.net = network
        self.shadow_net = network.get_weights()
        self.xp = Experience(agentconfig.xpsize)
        self.cfg = agentconfig

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, state, reward):
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self, state, reward):
        raise NotImplementedError

    def learn_batch(self):
        X, Y = self.xp.replay(self.xp.limit)
        N = len(X)
        if N < self.xp.limit:
            return 0.
        costs = self.net.fit(X, Y, verbose=0)
        # return np.mean(cost.history["loss"])
        return np.mean(costs)

    def push_weights(self):
        W = self.net.get_weights(unfold=True)
        D = np.linalg.norm(self.shadow_net - W)
        self.shadow_net *= (1. - self.cfg.tau)
        self.shadow_net += self.cfg.tau * self.net.get_weights(unfold=True)
        return D / len(W)

    def pull_weights(self):
        self.net.set_weights(self.shadow_net, fold=True)

    def update(self):
        pass


class PG(AgentBase):

    """Policy Gradient"""

    type = "PG"

    def __init__(self, network, nactions, agentconfig=None, **kw):
        super().__init__(network, agentconfig, **kw)
        self.actions = np.arange(nactions)
        self.action_labels = np.eye(nactions)
        self.X = []
        self.Y = []
        self.rewards = []
        self.grad = np.zeros_like(network.get_gradients(unfold=True))

    def reset(self):
        self.X = []
        self.Y = []
        self.rewards = []

    def sample(self, state, reward):
        self.X.append(state)
        self.rewards.append(reward)
        probabilities = self.net.predict(state[None, ...])[0]
        action = np.random.choice(self.actions, p=probabilities)
        self.Y.append(self.action_labels[action])
        return action

    def accumulate(self, state, reward):
        # R = np.array(self.rewards[1:] + [reward])
        # if self.cfg.gamma > 0.:
        #     R = discount_rewards(R, self.cfg.gamma)
        #     # R -= R.mean()
        #     # R /= (R.std() + 1e-7)
        X = np.stack(self.X[1:], axis=0)
        Y = np.stack(self.Y[1:], axis=0)
        w = np.ones((len(X))) * reward
        costs = self.net.fit(X, Y, w=w, epochs=1, batch_size=50)
        self.reset()
        return np.mean(costs)


class DQN(AgentBase):

    """Deep Q Network"""

    type = "DeepQLearning"

    def __init__(self, network, nactions, agentconfig=None, **kw):
        super().__init__(network, agentconfig, **kw)
        self.X = []
        self.Q = []
        self.R = []
        self.A = []
        self.nactions = nactions

    def reset(self):
        self.X = []
        self.Q = []
        self.R = []
        self.A = []

    def sample(self, state, reward):
        self.X.append(state)
        self.R.append(reward)
        Q = self.net.predict(state[None, ...])[0]
        self.Q.append(Q)
        action = (np.argmax(Q) if np.random.uniform() > self.cfg.epsilon
                  else np.random.randint(0, self.nactions))
        self.A.append(action)
        return action

    def accumulate(self, state, reward):
        q = self.net.predict(state[None, ...])[0]
        X = np.stack(self.X, axis=0)
        Q = np.stack(self.Q[1:] + [q], axis=0)
        R = np.array(self.R[1:] + [reward])
        ix = tuple(self.A)
        Y = Q.copy()
        Ym = Y.max(axis=1) * self.cfg.gamma
        Y[range(len(Y)), ix] = -(R + Ym)
        Y[-1, ix[-1]] = -reward
        self.xp.remember(X, Y)
        self.reset()
        cost = self.learn_batch()
        return cost


class HillClimbing(AgentBase):

    def __init__(self, network, nactions, agentconfig=None, **kw):
        super().__init__(network, agentconfig, **kw)
        self.rewards = 0
        self.bestreward = 0

    def reset(self):
        self.rewards = 0

    def sample(self, state, reward):
        self.rewards += reward if reward is not None else 0
        pred = self.net.predict(state[None, :])[0]
        return pred.argmax()

    def accumulate(self, state, reward):
        W = self.net.get_weights(unfold=True)
        if self.rewards > self.bestreward:
            # print(" Improved by", self.rewards - self.bestreward)
            self.bestreward = self.rewards
            self.shadow_net = W
        self.net.set_weights(W + np.random.randn(*W.shape)*0.1)
        self.reset()
        self.bestreward *= 0.1
        return self.bestreward


class DDPG(AgentBase):

    """Deep Deterministic Policy Gradient"""

    def accumulate(self, state, reward):
        pass

    def sample(self, state, reward):
        pass

    def reset(self):
        pass
