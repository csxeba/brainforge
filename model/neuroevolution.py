from .abstract_learner import LearnerBase
from ..evolution import Population


class NeuroEvolution(LearnerBase):

    def __init__(self, layerstack, cost="mse", population_size=100, name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        ff = kw.pop("fitness_function", self.fitness)
        fw = kw.pop("fitness_weights", [1.])
        oa = kw.pop("on_accuracy", False)
        self.population = Population(
            loci=self.layers.nparams,
            fitness_function=ff,
            fitness_weights=fw,
            limit=population_size, **kw
        )
        self.on_accuracy = oa

    @staticmethod
    def as_weights(genome):
        return (genome - 0.5) * 20.

    def learn_batch(self, X, Y, **kw):
        evolepoch = kw.get("epochs", 1)
        survrate = kw.get("survival_rate", 0.8)
        mutrate = kw.get("mutation_rate", 0.1)
        evolverbose = kw.get("verbosity", 0)
        self.population.run(epochs=evolepoch, survival_rate=survrate, mutation_rate=mutrate,
                            verbosity=evolverbose, X=X, Y=Y)
        self.layers.set_weights(self.as_weights(self.population.best))
        return self.population.mean_grade()

    def fitness(self, genome, X, Y):
        self.layers.set_weights(self.as_weights(genome))
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result
