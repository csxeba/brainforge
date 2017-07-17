from .abstract_learner import Learner
from ..evolution import Population


class NeuroEvolution(Learner):

    population = None  # type: Population
    on_accuracy = False

    def finalize(self, population_size=100, on_accuracy=False, **kw):
        self.population = Population(loci=self.nparams, fitness_function=self.fitness,
                                     fitness_weights=[1.], limit=population_size)
        self.on_accuracy = on_accuracy

    def learn_batch(self, X, Y):
        self.population.run(epochs=1, survival_rate=0.3, verbosity=0, X=X, Y=Y)
        self.set_weights(self.population.best * 10.)
        return self.population.mean_grade()

    def fitness(self, genome, X, Y):
        self.set_weights(genome)
        result = self.evaluate(X, Y, classify=self.on_accuracy)
        return (1. - result[-1]) if self.on_accuracy else result
