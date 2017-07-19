from .abstract_learner import LearnerBase
from ..evolution import Population


class NeuroEvolution(LearnerBase):

    population = None  # type: Population
    on_accuracy = False

    def finalize(self, cost="mse", population_size=100, on_accuracy=False, **kw):
        super().finalize(cost)
        ff = kw.pop("fitness_function", self.fitness)
        fw = kw.pop("fitness_weights", [1.])
        self.population = Population(
            loci=self.layers.nparams,
            fitness_function=ff,
            fitness_weights=fw,
            limit=population_size, **kw
        )
        self.on_accuracy = on_accuracy

    @staticmethod
    def as_weights(genome):
        return (genome - 0.5) * 20.

    def learn_batch(self, X, Y, **kw):
        self.population.run(epochs=1, survival_rate=0.8, mutation_rate=0.1,
                            verbosity=0, X=X, Y=Y)
        self.layers.set_weights(self.as_weights(self.population.best))
        return self.population.mean_grade()

    def fitness(self, genome, X, Y):
        self.layers.set_weights(self.as_weights(genome))
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result
