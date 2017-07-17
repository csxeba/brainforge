from .abstract_learner import Learner
from ..evolution import Population


class NeuroEvolution(Learner):

    population = None  # type: Population
    on_accuracy = False

    def finalize(self, cost, population_size=100, on_accuracy=False, **kw):
        super().finalize(cost)
        self.population = Population(loci=self.nparams,
                                     fitness_function=kw.get("fitness_function", self.fitness),
                                     fitness_weights=kw.get("fitness_weights", [1.]),
                                     limit=population_size, **kw)
        self.on_accuracy = on_accuracy
        self._finalized = True

    @staticmethod
    def as_weights(genome):
        return (genome - 0.5) * 20.

    def learn_batch(self, X, Y):
        self.population.run(epochs=1, survival_rate=0.8, mutation_rate=0.1,
                            verbosity=0, X=X, Y=Y)
        self.set_weights(self.as_weights(self.population.best))
        return self.population.mean_grade()

    def fitness(self, genome, X, Y):
        self.set_weights(self.as_weights(genome))
        cost = self.cost(self.predict(X), Y)
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result
