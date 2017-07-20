def _parameter_alias(item):
    return {"training_batch_size": "bsize",
            "discount_factor": "gamma",
            "knowledge_transfer_rate": "tau",
            "epsilon_greedy_rate": "epsilon",
            "epsilon_decay": "epsilon_decay",
            "epsilon_decay_factor": "epsilon_decay",
            "replay_memory_size": "xpsize"}.get(item, item)


class AgentConfig:

    def __init__(self, **kw):

        self.bsize = 300
        self.gamma = 0.99
        self.tau = 0.1
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 1.0
        self.xpsize = 9000
        self.time = 1
        self.__dict__.update({_parameter_alias(k): v for k, v in kw.items() if k != "self"})

    @property
    def decaying_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            return self.epsilon

        self.epsilon = self.epsilon_min
        return self.epsilon_min

    def __getitem__(self, item):
        return self.__dict__[_parameter_alias(item)]

    def __setitem__(self, key, value):
        self.__dict__[_parameter_alias(key)] = value
