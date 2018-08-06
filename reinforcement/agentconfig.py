from .experience import replay_memory_factory, Experience


class AgentConfig:

    def __init__(self, batch_size=128,
                 discount_factor=0.99,
                 knowledge_transfer_rate=0.1,
                 epsilon_greedy_rate=0.9,
                 epsilon_decay=1.0,
                 epsilon_min=0.01,
                 replay_memory=None,
                 timestep=1):

        self.bsize = batch_size
        self.gamma = discount_factor
        self.tau = knowledge_transfer_rate
        self.epsilon = epsilon_greedy_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_memory = replay_memory if isinstance(replay_memory, Experience) else replay_memory_factory(replay_memory)
        self.time = timestep

    @property
    def decaying_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            return self.epsilon
        self.epsilon = self.epsilon_min
        return self.epsilon_min
