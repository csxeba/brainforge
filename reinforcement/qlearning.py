import numpy as np

from .experience import replay_memory_factory
from .abstract_agent import AgentBase


class DQN(AgentBase):

    """Deep Q Network"""

    type = "DQN"

    def __init__(self, network, num_actions, agentconfig=None, **kw):
        super().__init__(network, agentconfig, **kw)
        self.inputs = []
        self.predictions = []
        self.rewards = []
        self.actions = []
        self.num_actions = num_actions

    def reset(self):
        self.inputs = []
        self.predictions = []
        self.rewards = []
        self.actions = []

    def sample(self, state, reward):
        self.inputs.append(state)
        self.rewards.append(reward)
        Q = self.net.predict(state[None, ...])[0]
        self.predictions.append(Q)
        action = (np.argmax(Q) if np.random.uniform() > self.cfg.decaying_epsilon
                  else np.random.randint(0, self.num_actions))
        self.actions.append(action)
        return action

    def accumulate(self, state, reward):
        q = self.net.predict(state[None, ...])[0]
        X = np.stack(self.inputs, axis=0)
        Q = np.stack(self.predictions[1:] + [q], axis=0)
        R = np.array(self.rewards[1:] + [reward])
        ix = tuple(self.actions)
        Y = Q
        Y[range(len(Y)), ix] = -(R + Y.max(axis=1) * self.cfg.gamma)
        Y[-1, ix[-1]] = -reward
        self.xp.remember(X, Y)
        self.reset()


class DDQN(DQN):

    type = "DDQN"

    def __init__(self, network, num_actions, agentconfig, **kw):
        from pickle import loads, dumps
        super().__init__(network, num_actions, agentconfig, **kw)
        self.double = [network, loads(dumps(network))]
        self.doublexp = [replay_memory_factory(agentconfig.xpsize, "drop", agentconfig.time) for _ in range(2)]
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
        X = np.stack(self.inputs + [state], axis=0)
        R = np.array(self.rewards[1:] + [reward])

        Y = self.critic.predict(X[1:])
        Y[range(len(Y)), (tuple(self.actions))] = -(R + Y.max(axis=1) * self.cfg.gamma)
        Y[-1, self.actions[-1]] = -reward

        self.xp.remember(X[1:], Y)
        cost = self.learn_batch()
        self._swap_actor()

        self.reset()
        return cost
