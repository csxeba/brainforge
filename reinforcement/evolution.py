from .agentconfig import AgentConfig
from .abstract_agent import AgentBase


class EvolutionStrategies(AgentBase):

    def __init__(self, network, agentconfig: AgentConfig, **kw):
        super().__init__(network, agentconfig, **kw)


    def reset(self):
        pass

    def sample(self, state, reward):
        pass

    def accumulate(self, state, reward):
        pass