from .abstract_optimizer import Optimizer as _Optimizer


class DifferentialEvolution(_Optimizer):

    """
    Wrapper for brainforge.evolution.Population.
    Coordinates the differential evolution of weight learning.
    """

    def __init__(self, nparams=0, population=None, optimize_accuracy=False):

        super().__init__()
        if population is None:
            from ..evolution import Population
            if not nparams:
                raise RuntimeError("Please supply the number of weights to be optimized!")
            population = Population(nparams, self.default_fitness, fitness_weights=[1], limit=50)

        self.population = population
        self.optimize_accuracy = optimize_accuracy

    def optimize(self, net, x, y, epochs=1, survival_rate=0.1, mutation_rate=0.1,
                 force_update_at_every=0, verbosity=0):
        self.population.run(epochs, survival_rate, mutation_rate,
                            force_update_at_every, verbosity,
                            net=net, x=x, y=y, opt_acc=self.optimize_accuracy)
        best = self.population.best * 10.
        grade = self.population.grades.min()
        return best, grade

    def capsule(self, nosave=()):
        caps = {"optimize_accuracy": self.optimize_accuracy}
        caps.update(self.population.capsule())
        return caps

    def __str__(self):
        return "Evolution"

    @staticmethod
    def default_fitness(ind, net, x, y, opt_acc):
        net.set_weights(ind)
        result = net.evaluate(x, y, classify=opt_acc)
        return (1. - result[-1]) if opt_acc else result
