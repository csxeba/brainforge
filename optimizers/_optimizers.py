import numpy as np


class SGD:

    def __init__(self, layer, eta=0.01):
        self.layer = layer
        self.eta = eta

    def __call__(self, m):
        eta = self.eta / m
        self.layer.weights -= self.layer.nabla_w * eta
        self.layer.biases -= self.layer.nabla_b * eta

    def __str__(self):
        return "SGD"

    def capsule(self):
        return [self.eta]


class Momentum(SGD):

    def __init__(self, layer, eta=0.1, mu=0.9, nesterov=False, *args):
        SGD.__init__(layer, eta)
        self.mu = mu
        self.nesterov = nesterov
        if not args:
            self.vW = np.zeros_like(layer.weights)
            self.vb = np.zeros_like(layer.biases)
        else:
            if len(args) != 2:
                raise RuntimeError("Invalid number of params for Adagrad!")
            self.vW, self.vb = args

    def __call__(self, m):
        eta = self.eta / m
        self.vW *= self.mu
        self.vb *= self.mu
        deltaW = self.layer.weights + self.vW if self.nesterov else self.layer.nabla_w
        deltab = self.layer.biases + self.vb if self.nesterov else self.layer.nabla_b
        self.vW += deltaW * eta
        self.vb += deltab * eta
        self.layer.weights -= self.vW
        self.layer.biases -= self.vb

    def __str__(self):
        return ("Nesterov " if self.nesterov else "") +"Momentum"

    def capsule(self):
        return [self.eta, self.mu, self.nesterov]  # , self.vW, self.vb]


class Adagrad(SGD):

    def __init__(self, layer, eta=0.01, epsilon=1e-8, *args):
        SGD.__init__(self, layer, eta)
        self.epsilon = epsilon
        if not args:
            self.mW = np.zeros_like(layer.weights)
            self.mb = np.zeros_like(layer.biases)
        else:
            if len(args) != 2:
                raise RuntimeError("Invalid number of params for Adagrad!")
            self.mW, self.mb = args

    def __call__(self, m):
        eta = self.eta / m
        self.mW += self.layer.nabla_w ** 2
        self.mb += self.layer.nabla_b ** 2
        self.layer.weights -= (eta / np.sqrt(self.mW + self.epsilon)) * self.layer.nabla_w
        self.layer.biases -= (eta / np.sqrt(self.mb + self.epsilon)) * self.layer.nabla_b

    def __str__(self):
        return "Adagrad"

    def capsule(self):
        return [self.eta, self.epsilon]  # , self.mW, self.mb]


class RMSprop(Adagrad):

    def __init__(self, layer, eta=0.1, decay=0.9, epsilon=1e-8, *args):
        Adagrad.__init__(self, layer, eta, epsilon)
        self.decay = decay
        if args:
            if len(args) != 2:
                raise RuntimeError("Invalid number of params for Adagrad!")
            self.mW, self.mb = args

    def __call__(self, m):
        eta = self.eta / m
        self.mW = self.decay * self.mW + (1 - self.decay) * self.layer.nabla_w**2
        self.mb = self.decay * self.mb + (1 - self.decay) * self.layer.nabla_b**2
        self.layer.weights -= eta * self.layer.nabla_w / (np.sqrt(self.mW) + self.epsilon)
        self.layer.biases -= eta * self.layer.nabla_b / (np.sqrt(self.mb) + self.epsilon)

    def __str__(self):
        return "RMSprop"

    def capsule(self):
        return [self.eta, self.decay, self.epsilon]  # , self.mW, self.mb]


class Adam(SGD):

    def __init__(self, layer, eta=0.1, decay_memory=0.9, decay_velocity=0.999, epsilon=1e-8, *args):
        SGD.__init__(self, layer, eta)
        self.decay_memory = decay_memory
        self.decay_velocity = decay_velocity
        self.epsilon = epsilon

        if not args:
            self.mW = np.zeros_like(layer.weights)
            self.mb = np.zeros_like(layer.biases)
            self.vW = np.zeros_like(layer.weights)
            self.vb = np.zeros_like(layer.biases)
        else:
            if len(args) != 4:
                raise RuntimeError("Invalid number of params for ADAM!")
            self.mW, self.mb, self.vW, self.vb = args

    def __call__(self, m):
        eta = self.eta / m
        self.mW = self.decay_memory*self.mW + (1-self.decay_memory)*self.layer.nabla_w
        self.mb = self.decay_memory*self.mb + (1-self.decay_memory)*self.layer.nabla_b
        self.vW = self.decay_velocity*self.vW + (1-self.decay_velocity)*(self.layer.nabla_w**2)
        self.vb = self.decay_velocity*self.vb + (1-self.decay_velocity)*(self.layer.nabla_b**2)

        self.layer.weights -= eta * self.mW / (np.sqrt(self.vW) + self.epsilon)
        self.layer.biases -= eta * self.mb / (np.sqrt(self.vb) + self.epsilon)

    def __str__(self):
        return "Adam"

    def capsule(self):
        param = [self.eta, self.decay_memory, self.decay_velocity, self.epsilon]
        # param += [self.mW, self.mb, self.vW, self.vb]
        return param


optimizer = {key.lower(): cls for key, cls in locals().items() if key != "np"}
