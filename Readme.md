[![codebeat badge](https://codebeat.co/badges/f72db301-fd66-4c05-b1ca-9b8c8196f06e)](https://codebeat.co/projects/github-com-csxeba-brainforge-dev)

# Brainforge documentation

Brainforge is an Artificial Neural Networking library implemented in **Python**, which only depends on **NumPy** (and optionally **Numba**, a jit compiler for Python and NumPy)

## Installation

Sadly, the project is not packaged at the moment. This is because I don't know how it is done, but I intend to do so in
the near future, so it's downloading and unzipping for now :(

## Models

Brainforge treats neural networks as layer stacks with some learning logic applied to them. These concepts are separated in the API, so the building of an ANN is a multi-stage process.
1. A **LayersStack** instance should be created and populated with layers.
2. Layers are instances of **Layer** subclasses, which group together the forward and backward logic of various architectures.
2. The LayerStack should be wrapped with some learning logic, these are instances of Learner subclasses.

Currently the following learning wrappers are implemented:

- **BackpropNetwork**, which uses a selected variant of Gradient Descent to optimize the weights (see section **Optimizers** for more information)
- **NeuroEvolution**, which uses differential evolution for weight optimization

### LayerStack - Specifiing the architecture

#### Constructor parameters

- *input_shape*: tuple or int, specifying the dimensionality of the data. An **InputLayer** will be automatically instanciated based on this information.
- *layers*: some iterable, holding Layer instances.
- *name*: string, specifying a name for the network, used in the **describe()** and **save()** methods.

#### Methods

- *add*: expects a Layer instance, which is added to the top of the layer stack.
- *pop*: deletes the last layer from the layer stack.
- *feedforward*: performs a forward-propagation through the layer stack.
- *get_weights* and *set_weights*. They accept a bool parameter (unfold and fold respectively) for convenience.
- *reset*: reinitializes the layers randomly.

#### Properties

- *nparams*: returns the total number of trainable parameters

#### Other info

LayerStack can be iterated with a for loop, which yields the layers of the stack.

### Learner - Learning Logic Interface
Every learning logic wrapper is derived from the abstract base class, Learner, which specifies an interface for fitting.

#### Constructor parameters

- *layerstack*: either a LayerStack instance or an iterable containing layer instances (in the latter case, **input_shape** has to be passed as a keyword argument.
- *cost*: eiter a string, specifiing a cost function or the instance of a ConstFunction subclass. See section **Costs** for more information.
- *name*: optional string, used in persistance.

#### Methods

- *fit*: fits the model to the given data. Calls *fit_generator* in the background.
  - *X*: numpy array containing the training Xs (the independent variables)
  - *Y*: numpy array containing the training Ys (the dependent variables)
  - *batch_size*: the batch size of stochastic training. 1 for on-line (per-lesson) training. Defaults to 20.
  - *epochs*: the number of epochs to train for. Defaults to 30.
  - *classify*: boolean parameter, which determines whether to calculate classification accuracy along with cost during evaluation.
  - *validation*: tuple, containing separate X and Y tensors used to validate the network performance. Defualts to an
empty tuple.
  - *verbose*: integer, specifying the level of verbosity. Defaults to 1.
  - *shuffle*: bool, if set, the Xs and Ys will be reshuffled before every epoch. Defaults to True.
supported for monitoring the classification accuracy.
- *fit_generator*: fits the model to an infinite generator, which spits out batches of learning data. Calls *epoch* in the background
- *epoch*: runs a single epoch on the supplied data. Calls *learn_batch* in the background.
- *learn_batch*: abstract method, derivatives must implement it.
- *predict*: performs a forward-propagation through the layer stack.
- *evaluate*: evaluates the network's performance on some X and Y. Calculates the cost and can optionally return classification accuracy as well, if the parameter *classify* is set to True.
- *cost*: an instance method, basically a function reference to the cost function.

#### Properties
- *layers*: a LayerStack instance. Can be iterated with a for loop.
- *age*: 0 if the network is untrained. Incremented by 1 after every epoch.

### BackpropNetwork - Gradient Descent learning

Gradient Descent and derivatives are implemented as Optimizer subclasses.

#### Constructor parameters

- *layerstack*, *cost* and *name* are the same as in **Learner**
- *optimizer*: either a string specifiing an optimizer algorithm with default hyperparameters or the instance of an Optimizer subclass. See section **Optimizers** for more information.

#### Methods
- Methods from Learner are inherited.
- *learn_batch*: fits the weights to a batch of data.
- *backpropagate*: performs a backwards pass and returns all gradients concatenated to vector. Expects a matrix, *error*, which contains the output errors. Output errors can be obtained by calling instance.cost.derivative(prediction, target).
- *get_gradients*: returns the gradients currently stored (from the last call of *backpropagate* or *fit_batch*). The parameter *unfold* can be set to False if the grads should be reshaped to the weights' shapes.

#### Properties
- *layers* and *age* are inherited from Learner.
- *optimizer*: can be used to directly call into the optimizer.

### NeuroEvolution - Differential Evolution

A simple genetic algorith is implemented in the submodule *evolution*. See more in section **Evolution**.

#### Constructor parameters
- *layerstack*, *cost* and *name* are the same as in **Learner**.
- *population_size*: how many sets of weights to keep in memory as individuals in a population.
- *kw*: optional keywors arguments are passed down to an implicitly instanciated Population constructor. See section **Evolution** for more information. **on_accuracy** is one such argument which specifies whether to minimize cost or the classification accuracy directly.

#### Methods
- *learn_batch*: kwargs are supported for Population.run. Calling **fit()** or **fit_generator()**, these args are passed down to **epoch** and finally **learn_batch**.
- *fitness*: the default fitness function, return the cost or the classification accuracy (depending on the value of the **on_accuracy** constructor parameter.
- *as_weights*: upscales an individual (0-1) to the range (-10, 10). I know its lame, but meh... I'll look into this later.

#### Properties
- *population*: a Population instance with unfolded weight matrices as individuals.

## Layers

### Core layers
These layers are the ones used most regularily when working with ANNs.
- **(InputLayer)**: passes the input tensor forward, unmodified.
  - No parameters required
  - This layer gets instantiated automatically
- **Flatten**: performs flattening on a *dims* dimensional tensor. Returns a matrix of shape
(*batch_size*, prod(*dims*))
  - No paramters required
- **Reshape**: reshapes the input tensor to a given shape. Keeps the leading dimension,
which corresponds to *batch_size*
  - *shape*: determines the output shape.
- **DenseLayer**: just your regular densely connected layer.
  - *neurons*: integer, specifying the output dimensions
  - *activation*: string or ActivationFunction instance, specifying the activation function. Defaults to linear.
- **Activation**: applies a (possibly) nonlinear activation function elementwise on the
input tensor.
  - *activation*: string, specifying the function to be used. Can be one of see section **Activation** for more info.

### Fancy layers
Some fancy feedforward layers from various scientific articles
- **HighwayLayer**: See Srivastava et al., 2015. This layer applies a gating mechanism to its inputs. This consist of
a forget gate and an input gate, determining the amount of information discarded and kept. Finally, it applies an
output gate, selecting the information to be passed on to the next layer. This operation doesn't change the
dimensionality of the inputs, but it does perform the usual neural projection by appliing a weight matrix.
  - activation: string or ActivationFunction instance, specifying the activation function to be applied
- **DropOut**: Discards certain neurons in the training phase, thus improving generalization by regularization.
See Srivastava et al., 2014

### Recurrent
Recurrent layers working with multidimensional (time-series) data

- **RLayer**: simple recurrence, output is simply fed back in each timestep.
  - *neurons*: integer, specifying the output (and the inner state's) shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.
- **LSTM**: Long-Short Term Memory, see Hochreiter et al., 1997
  - *neurons*: integer, specifying the output (and the inner state's) shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.
- **GRU**: Gated Recurrent Unit, see Chung et al., 2014
  - *neurons*: integer, specifying the output (and the inner state's) shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.
- **ClockworkLayer**: see Koutník et al., 2014: a simple recurrent layer with time-dependent masking. The weight matrix
is partitioned into blocks. Each block is assigned a tick. At timestep t, only those blocks are activated, whose tick
value is a factor of t (e.g. t *mod* tick = 0, where *mod* is the modulo operation).
  - *neurons*: integer, specifying the output shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *blocksizes*: iterable, specifying the size and number of blocks in the layer. Sizes have to sum up to *neurons*.
Defaults to 5 blocks, with equal blocksizes. The remainder of the division-by-five is added to the first block.
  - *ticktimes*: iterable, specifying the tick time of the blocks. Has to contain as many > 0 elements as *blocksizes*.
Defaults to 2^i, where i is the (one-based) index of the block in question.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.
- **Reservoir**: an untrainable recurrent layer used in Echo State Networks. Has a sparse weight matrix normalized to
unit spectral radius. See Reservoir computing and Jaeger, 2007 (Scholarpaedia)
  - *neurons*: integer, specifying the output (and the inner state's) shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.

### Tensor
Feedforward layers working with multidimensional data

- **PoolLayer**: untrainable layer performing the max-pooling operation
  - *fdim*: integer, specifying the filter dimension. This value will also be used as the stride of the pooling operation.
  - *compiled*: bool, specifying whether to jit compile the layer with **numba** (highly recommended).
- **ConvLayer**: performs convolution/cross-correlation on a batch of images by learnable kernels of a given shape.
  - *nfilters*: integer, specifying the number of filters (neurons) to be learned
  - *filterx*: integer, specifying the X dimension of the filter
  - *filtery*: integer, specifying the Y dimension of the filter
  - *compiled*: bool, specifying whether to jit compile the layer with **numba** (highly recommended).

## Optimizers
Currently the following optimizers are implemented:

- **SGD**: the classic stochastic gradient descent
  - *eta*: the learning rate, defaults to 0.01
- **Momentum**: Sums up all previous gradients and updates parameters according to this accumulated term (velocity, in a
physics sense), instead of the raw gradients.
  - *eta*: the learning rate, defaults to 0.1
  - *mu*: decay term for summed gradients, defaults to 0.9. Friction, from a physics standpoint.
- **Nesterov**: Nesterov-accelerated gradient, very similar to Momentum, only this method "looks ahead" before updating
the position. See Nesterov, 1983.
- **Adagrad**: Adaptive Subgradient Method for stochastic optimization, see Duchi et al. 2011. An internal memory of all
prevoius squared gradients and the general learning rate is modulated by its RMS on a per-parameter basis.
  - *eta*: general learning rate, defaults to 0.01
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.
- **RMSprop**: the unpublished method of Geoffrey Hinton. See Lecture 6e of his Coursera Class. Keeps a decaying average
of past squared gradients and uses it to adjust updates on a per-parameter basis.
  - *eta*: general learning rate, defaults to 0.1
  - *decay*: the memory decay term / update rate. Defaults to 0.9
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.
- **Adam**: Adaptive Moment Estimation, see Kingma et al., 2015. Keeps a decaying average of past gradients (velocity)
and of past squared gradients (memory). Basically the combination of Adagrad and RMSprop.
  - *eta*: general learning rate, defaults to 0.1
  - *decay_memory*: the memory decay term / update rate. Defaults to 0.9
  - *decay_velocity*: the velocity decay term / update rate. Defaults to 0.999
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.

Adam becomes unstable when the gradients are small :(

## Costs
The following cost functions are supported:

- **Mean Squared Error (MSE)**: can be used with any output activation, but can be slow to converge with saturated
sigmoids. For classification and regression tasks.
- **Categorical Cross-Entropy (Xent)**: can be used with *sigmoid* or *softmax* (0-1) output activations. It is intended to
use for classification tasks (softmax for multiclass, sigmoid for multilabel-multiclass).
- **Hinge loss (Hinge)**: for maximal margin convergence.

## Evolution
A simple genetic algorithm is included. This technique can be used either to evolve optimal hyperparameters for an ANN
or to evolve the parameters (weights and biases) themselves (or both) (or for other purposes).
The feature can be accessed via the **brainforge.evolution** module, which defines the following class:
- **Population**: abstraction of a population of individuals
  - *loci*: integer, specifying the number of loci on the individuals' genome (the length of the genome).
  - *fitness_function*: a function reference, which will be called on every individual. This function has to return
a tuple/list/array of fitness values.
  - *fitness_weights*: an array, specifying the weights used when calculating the weighted sum of the fitness values,
thus creating a **grade** value representing the individuals' overall value. Defaults to (1,).
  - *limit*: integer, specifying the size of the population (the number of individuals). Defaults to 100.
  - *grade_function*: reference to a function, which, if supplied, will be used instead of weighted sum to determine the
grade (overall value) of individuals. This function has to return a scalar given a tuple of fitness values. Defaults to
weighted sum, if not set.
  - *mate_function*: reference to a function, which, if supplied, will be used to produce an offspring, given two
individuals (genomes). Defaults to per-locus random selection, if not set.

Populations an be run for a set amount of epochs by using the **Population.run()** method. An epoch consists of the
following evolutionary operations:
1. *selection*: determines which individuals survive, based on their grades
2. *mating*: generate candidate individuals in place of the ones deleted in the previous step.
3. *mutation*: mutation is applied on a per-locus basis.
4. *update*: updates the fitnesses and grades where it is needed.

The **run** method accepts the following parameters:
- *epochs*: the number of epochs to run for
- *survival_rate*: the rate of individuals kept in the *selection* step. Defaults to 0.5.
- *mutation_rate*: the probability of a mutation happening on a per-individual basis. Defaults to 0.1. Note that
mutation is applied per-locus, so this term gets divided by the length of the chromosome.
- *force_update_at_every*: integer, specifying whether it is needed to force-update every individual, not just the
ones, which are not up-to-date (e.g. offsprings and mutants).
- *verbosity*: integer, specifying the level of verbosity. 1 prints out run dynamics after each epoch. > 1 prints out
verbosity-1 number of the top individuals' genomes and fitnesses as grades.
- Miscellaneous keyword arguments can also be specified, which get passed down to the fitness function.

## Reinforcement Learning
Reinforcement learning agents are accessible from **brainforge.reinforcement**. Currently the following angents are
implemented:
- DQN: Deep Q Learning, see Mnih et al., 2013.
- PG: A vanilla policy gradient method.
- DDQN: Double Deep Q Learning, see Hasselt et al., 2015.

## Examples
### Fit shallow backprop net to the XOR problem
```python
import numpy as np

from brainforge import LayerStack, Backpropagation
from brainforge.layers import DenseLayer

def input_stream(batchsize=20):
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(Xs), size=batchsize)
        yield Xs[arg], Ys[arg]

layers = LayerStack(input_shape=2, layers=[
    DenseLayer(3, activation="sigmoid"),
    DenseLayer(2, activation="softmax")
])
net = BackpropNetwork(layers, cost="xent", optimizer="momentum")

datastream = input_stream(1000)
validation = next(input_stream(100))

net.fit_generator(datastream, lessons_per_epoch=1000000, epochs=10, classify=True,
                  validation=validation, verbose=1)
```

For more complicated tasks, the use of the dataframe library csxdata is suggested.

### Fit LeNet-like ConvNet to images
```python
from csxdata import CData

from brainforge import Backpropagation
from brainforge.layers import (
    DenseLayer, DropOut, Activation,
    PoolLayer, ConvLayer, Flatten
)

dataroot = "path/to/pickled/gzipped/tuple/of/X/Y/ndarrays.pkl.gz"
images = CData(dataroot, indeps=0, headers=None, cross_val=10000)

inshape, outshape = images.neurons_required

model = BackpropNetwork(input_shape=inshape, layerstack=(
    ConvLayer(nfilters=10, filterx=3, filtery=3, compiled=True),
    PoolLayer(filter_size=2, compiled=True),
    Activation("relu"),
    ConvLayer(nfilters=10, filterx=5, filtery=5, compiled=True),
    PoolLayer(filter_size=3, compiled=True),
    Activation("relu"),
    Flatten(),
    DenseLayer(120, activation="tanh"),
    DropOut(0.5),
    DenseLayer(outshape, activation="softmax")
), cost="xent", optimizer="rmsprop")
X, Y = images.table("learning")
valid = images.table("testing")
model.fit(X, Y, batch_size=20, epochs=30, validation=valid,
          monitor=["acc"])
```

### Fit LSTM to text data
```python
from csxdata import Sequence

from brainforge import Backpropagation
from brainforge.layers import DenseLayer, LSTM

datapath = "path/to/text/file.txt"
# Chop up the data into three-character 'ngrams'
# (a character-level RNN would use n_gram=1)
# Sequence data is ordered. In this case, the
# timestep parameter specifies how many ngrams
# are in X before Y is set. Consider the following case:
# [Mar][y h][ad ][a l][itt][le ][lam]
# [     THESE NGRAMS ALL GO TO X    ]
# The next ngram, [b ,] is the Y corresponding to the
# above X. Thus X is 3-dimensional, conventionally:
# (timestep, batch_size, k),
# Where k either comes from the one-hot representation
# of every individual ngram, or some kind of embedding
# into k-dimensional continous space.
seq = Sequence(datapath, n_gram=3, timestep=7)
inshape, outshape = seq.neurons_required
model = BackpropNetwork(input_shape=inshape, layerstack=(
    LSTM(120, activation="relu", return_seq=True),
    LSTM(60, activation="relu", return_seq=False),
    DenseLayer(120, activation="tanh"),
    DenseLayer(outshape, activation="softmax")
), cost="xent", optimizer="adam")
X, Y = seq.table("learning")
valid = seq.table("testing")
model.fit(X, Y, batch_size=120, epochs=100,
          monitor=["acc"], validation=valid)
```

### Evolve network hyperparameters.
In this script, we use the evolutionary algorithm to optimize the number of neurons in two hidden layers, along with
dropout rates. The fitness values are constructed so, that besides the classification error rate, we try to minimize
the time required to train a net for 30 epochs.
This approach creates a separate neural network for every individual during update time, so depending on the input
dimensions, this can be quite a RAM-hog.
```python
import time

import numpy as np
from matplotlib import pyplot as plt

from brainforge import Backpropagation
from brainforge.layers import DenseLayer, DropOut
from brainforge.evolution import Population, to_phenotype

from csxdata import CData

dataroot = "/path/to/csv"
frame = CData(dataroot, headers=1, indeps=3, feature="FeatureName")

inshape, outshape = frame.neurons_required

# Genome will be the number of hidden neurons at two network DenseLayers.
ranges = ((10, 300), (0., 0.75), (10, 300), (0., 0.75))
# We determine 3 fitness values: the network's classification error, the time
# required to run the net and the L2 norm of the weights. These values will be
# minimized and the accuracy will be considered with a 20x higher weight.
fweights = (200., 1., 1.)


def phenotype_to_ann(phenotype):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(int(phenotype[0]), activation="tanh"),
        DropOut(dropchance=phenotype[1]),
        DenseLayer(int(phenotype[2]), activation="tanh"),
        DropOut(dropchance=phenotype[3]),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="momentum")
    return net


# Define the fitness function
def fitness(genotype):
    start = time.time()
    net = phenotype_to_ann(to_phenotype(genotype, ranges))
    net.fit(*frame.table("learning"), batch_size=20, epochs=30, verbose=0)
    cost, acc = net.evaluate(*frame.table("testing", m=10), classify=True)
    error_rate = 1. - acc
    time_req = time.time() - start
    l2 = np.linalg.norm(net.layers.get_weights(unfold=True), ord=2)
    return error_rate, time_req, l2

# Build a population of 15 individuals. grade_function and mate_function are
# left to defaults.
pop = Population(loci=4, limit=15,
                 fitness_function=fitness,
                 fitness_weights=fweights)
# The population is optimized for 30 rounds.
# At every 3 rounds, we force a complete re-update of fitnesses because of the
# many stochastic aspects of ANNs (weight initialization, minibatch-randomizing, etc.)
means, stds, bests = pop.run(epochs=30,
                             survival_rate=0.3,
                             mutation_rate=0.05,
                             force_update_at_every=3)

Xs = np.arange(1, len(means)+1)
plt.title("Population grade dynamics of\nevolutionary hyperparameter optimization")
plt.plot(Xs, means, color="blue")
plt.plot(Xs, means+stds, color="green", linestyle="--")
plt.plot(Xs, means-stds, color="green", linestyle="--")
plt.plot(Xs, bests, color="red")
plt.show()
```

### Evolve network weights and biases
```python
from matplotlib import pyplot as plt

from csxdata import CData

from brainforge import NeuroEvolution
from brainforge.layers import DenseLayer

datapath = "path/to/data.csv"
data = CData(datapath, indeps=1, headers=1)
data.transformation = "std"

net = NeuroEvolution(input_shape=data.neurons_required[0], layerstack=(
    DenseLayer(60, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="softmax")
), cost="xent", population_size=30, on_accuracy=False)

print("Initial acc:", net.evaluate(*data.table("testing"))[1])
costs = net.fit(*data.table("learning"), epochs=50, validation=data.table("testing"), classify=True)

plt.plot(range(len(costs)), costs, c="red")
plt.show()
```

### Deep Q Learning on CartPole-v0
```python
from collections import deque

import gym

from brainforge import Backpropagation
from brainforge.layers import DenseLayer
from brainforge.reinforcement import DQN, agentconfig
from brainforge.optimization import RMSprop

env = gym.make("CartPole-v0")
inshape, outshape = env.observation_space.shape, env.action_space.n

net = BackpropNetwork(input_shape=inshape, layerstack=[
    DenseLayer(24, activation="tanh"),
    DenseLayer(outshape, activation="linear")
], cost="mse", optimizer=RMSprop(eta=0.0001))
agent = DQN(net, nactions=outshape, agentconfig=agentconfig.AgentConfig(
    epsilon_greedy_rate=1.0, epsilon_decay_factor=0.9998, epsilon_min=0.0,
    discount_factor=0.6, replay_memory_size=7200, training_batch_size=720
))

episode = 1
rewards = deque(maxlen=100)

while 1:
    state = env.reset()
    win = False
    reward = None
    for step in range(1, 201):
        action = agent.sample(state, reward)
        state, reward, done, info = env.step(action)
        if done:
            break
    else:
        win = True

    rewards.append(step)
    cost = agent.accumulate(state, 10. if win else -1.)
    meanrwd = sum(rewards) / len(rewards)
    if episode % 10 == 0:
        print(f"Episode {episode:>6}, running reward: {meanrwd:.2f}, Cost: {cost:>6.4f}")
        if win:
            print(" Win!")
    episode += 1
```
