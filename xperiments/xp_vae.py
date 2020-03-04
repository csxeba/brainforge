import numpy as np
from matplotlib import pyplot as plt
from verres.data import inmemory

from brainforge import LayerStack, Backpropagation
from brainforge.layers.abstract_layer import LayerBase, NoParamMixin
from brainforge.layers import Dense
from brainforge.optimizers import Adam
from brainforge.metrics import costs


class Sampler(NoParamMixin, LayerBase):

    def __init__(self):
        super().__init__()
        self._outshape = None
        self.epsilon = None
        self.cached_variance = None

    def connect(self, brain):
        self._outshape = brain.outshape
        super().connect(brain)

    def feedforward(self, X):
        """O = M + V * E"""
        mean, log_variance = X
        self.epsilon = np.random.randn(*mean.shape)
        self.cached_variance = np.exp(log_variance)
        return mean + self.cached_variance * self.epsilon

    @property
    def outshape(self):
        return self._outshape

    def backpropagate(self, delta):
        return delta, delta * self.cached_variance * self.epsilon


class KLD(costs.CostFunction):

    def __call__(self, mean, log_variance):
        return 0.5 * (np.exp(log_variance) + mean**2 - log_variance - 1).sum() / len(mean)

    def derivative(self, mean, log_variance):
        """
        d_KL/d_mean = mean
        d_KL/d_var = exp(log_var) - 1

        """
        m = len(mean)
        return mean / m, (np.exp(log_variance) - 1) / m


class VAE:

    def __init__(self, Z):
        self.encoder = Backpropagation([Dense(60, activation="relu"),
                                        Dense(30, activation="relu")], optimizer=Adam(1e-4))

        self.mean_z = Backpropagation(input_shape=30, layerstack=[Dense(Z)], optimizer=Adam(1e-4))
        self.stdev_z = Backpropagation(input_shape=30, layerstack=[Dense(Z)], optimizer=Adam(1e-4))
        self.sampler = Backpropagation(input_shape=Z, layerstack=[Sampler()])
        self.decoder = Backpropagation(input_shape=Z, layerstack=[Dense(30, activation="relu"),
                                                                  Dense(60, activation="relu"),
                                                                  Dense(784, activation="linear")],
                                       optimizer=Adam(1e-4))
        self.kld = KLD()
        self.mse = costs.mean_squared_error

        self.optimizers = [module.optimizer for module in [
            self.encoder, self.mean_z, self.stdev_z, self.sampler, self.decoder
        ]]

    def reconstruct(self, x, return_loss=True):
        x = x.reshape(len(x), -1)

        z = self.encoder.predict(x)
        s = self.stdev_z.predict(z)
        z = self.mean_z.predict(z)
        r = self.decoder.predict(z)

        result = [r]
        if return_loss:
            result.append(self.mse(x, r))
            result.append(self.kld(z, s))

        return result

    def train_on_batch(self, x):
        m = len(x)
        x = x.reshape(m, -1)
        enc = self.encoder.predict(x)
        mean = self.mean_z.predict(enc)
        log_variance = self.stdev_z.predict(enc)
        smpl = self.sampler.predict([mean, log_variance])
        dcd = self.decoder.predict(smpl)

        delta_reconstruction = self.mse.derivative(dcd, x)
        delta = self.decoder.backpropagate(delta_reconstruction)
        reconstruction_grandient = self.sampler.backpropagate(delta)
        variational_gradient = self.kld.derivative(mean, log_variance)
        delta = self.mean_z.backpropagate(reconstruction_grandient[0] + variational_gradient[0])
        delta += self.stdev_z.backpropagate(reconstruction_grandient[1] + variational_gradient[1])
        self.encoder.backpropagate(delta)

        self.encoder.update(m)
        self.mean_z.update(m)
        self.stdev_z.update(m)
        self.decoder.update(m)

        return self.mse(dcd, x), self.kld(mean, log_variance)


def main():
    vae = VAE(Z=32)

    data = inmemory.MNIST()
    steps_per_epoch = data.steps_per_epoch(32, "train")
    loss_history = {"recon": [], "kld": []}
    for epoch in range(1, 31):
        print("\n\nEpoch", epoch)
        for i, (x, y) in enumerate(data.stream(batch_size=32, subset="train")):
            recon, kld = vae.train_on_batch(x)
            loss_history["recon"].append(recon)
            loss_history["kld"].append(kld)
            print("\rDone: {:.2%} Reconstruction loss: {:.4f} KL-divergence: {:.4f}"
                  .format(i / steps_per_epoch,
                          np.mean(loss_history["recon"][-100:]),
                          np.mean(loss_history["kld"][-100:])), end="")
            if i == steps_per_epoch:
                break
        if epoch == 20:
            for optimizer in vae.optimizers:
                print("DROPPED!!!")
                optimizer.eta *= 0.1

    for x, y in data.stream(subset="val", batch_size=1):
        r, mse, kld = vae.reconstruct(x, return_loss=True)
        r = r.reshape(x.shape)[0]

        r, x = data.deprocess(r).squeeze(), data.deprocess(x[0]).squeeze()

        fig, (left, right) = plt.subplots(1, 2, sharex="all", sharey="all", figsize=(16, 9))
        left.imshow(x)
        right.imshow(r)
        left.set_title("original")
        right.set_title("reconstruction")
        plt.suptitle("MSE: {:.4f} KLD: {:.4f}".format(mse, kld))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
