import numpy as np

from .experience import xp_factory
from .abstract_agent import AgentBase


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
        Q = self.net.feedforward(state[None, ...])[0]
        self.Q.append(Q)
        action = (np.argmax(Q) if np.random.uniform() > self.cfg.decaying_epsilon
                  else np.random.randint(0, self.nactions))
        self.A.append(action)
        return action

    def accumulate(self, state, reward):
        q = self.net.feedforward(state[None, ...])[0]
        X = np.stack(self.X, axis=0)
        Q = np.stack(self.Q[1:] + [q], axis=0)
        R = np.array(self.R[1:] + [reward])
        ix = tuple(self.A)
        Y = Q.copy()
        Y[range(len(Y)), ix] = -(R + Y.max(axis=1) * self.cfg.gamma)
        Y[-1, ix[-1]] = -reward
        self.xp.remember(X, Y)
        self.reset()
        cost = self.learn_batch()
        return cost


class DDQN(DQN):

    def __init__(self, network, nactions, agentconfig, **kw):
        from pickle import loads, dumps
        super().__init__(network, nactions, agentconfig, **kw)
        self.double = [network, loads(dumps(network))]
        self.doublexp = [xp_factory(agentconfig.xpsize, "drop", agentconfig.time) for _ in range(2)]
        self.xp = self.doublexp[0]
        self.ix_actor = False

    def _swap_actor(self):
        self.ix_actor = not self.ix_actor
        self.net = self.double[self.ix_actor]
        self.xp = self.doublexp[self.ix_actor]

    @property
    def critic(self):
        return self.double[not self.ix_actor]

    def accumulate(self, state, reward):
        X = np.stack(self.X + [state], axis=0)
        R = np.array(self.R[1:] + [reward])

        Y = self.critic.feedforward(X[1:])
        Y[range(len(Y)), (tuple(self.A))] = -(R + Y.max(axis=1) * self.cfg.gamma)
        Y[-1, self.A[-1]] = -reward

        self.xp.remember(X[1:], Y)
        cost = self.learn_batch()
        self._swap_actor()

        self.reset()
        return cost
