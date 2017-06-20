import abc

import numpy as np

from ..util.rl_util import discount_rewards


class AgentBase(abc.ABC):

    type = ""

    def __init__(self, network, gamma=0.99):
        self.rewards = []
        self.net = network
        self.dcr = lambda R: discount_rewards(R, gamma)

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, state, reward):
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self, Ys, reward):
        raise NotImplementedError


class PolicyGradientAgent(AgentBase):

    type = "PolicyGradientAgent"

    def __init__(self, network, nactions, gamma=0.99):
        super().__init__(network, gamma)
        self.actions = np.eye(nactions)
        self.X = []
        self.Y = []
        self.rewards = []

    def reset(self):
        self.X = []
        self.rewards = []

    def sample(self, state, reward):
        self.X.append(state)
        self.rewards.append(reward)
        preds = self.net.predict(state[None, ...])[0]
        pred = np.random.choice()

    def accumulate(self, Ys, reward):
        R = self.dcr(self.rewards[1:] + [reward])
