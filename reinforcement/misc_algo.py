import numpy as np

from .abstract_agent import AgentBase


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
        W = self.net.layers.get_weights(unfold=True)
        if self.rewards > self.bestreward:
            # print(" Improved by", self.rewards - self.bestreward)
            self.bestreward = self.rewards
            self.shadow_net = W
        self.net.layers.set_weights(W + np.random.randn(*W.shape)*0.1)
        self.reset()
        self.bestreward *= 0.1
        return self.bestreward
