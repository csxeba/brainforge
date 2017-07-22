import numpy as np
from .abstract_agent import AgentBase
from ..util.rl_util import discount_rewards


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
        self.grad = np.zeros((network.nparams,))

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
        R = np.array(self.rewards[1:] + [reward])
        if self.cfg.gamma > 0.:
            R = discount_rewards(R, self.cfg.gamma)
            # R -= R.mean()
            # R /= (R.std() + 1e-7)
        X, Y = np.stack(self.X, axis=0), np.stack(self.Y, axis=0)
        N = len(X)

        cost = 0.
        m = self.cfg.bsize
        for start in range(0, N, m):
            y = Y[start:start+m]
            pred = self.net.predict(X[start:start + m])
            cost += self.net.cost(pred, y)
            delta = self.net.cost.derivative(pred, y)
            self.grad += self.net.backpropagate(delta * R[start:start+m, None])

        W = self.net.optimizer.optimize(
            self.net.layers.get_weights(unfold=True), self.grad, N
        )
        self.net.layers.set_weights(W)
        self.reset()
        return cost / N
