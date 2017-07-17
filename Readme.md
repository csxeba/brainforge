[![codebeat badge](https://codebeat.co/badges/f72db301-fd66-4c05-b1ca-9b8c8196f06e)](https://codebeat.co/projects/github-com-csxeba-brainforge-dev)

# Brainforge documentation
Brainforge is an Artificial Neural Networking library implemented in **Python**, which only depends on **NumPy** (and optionally **Numba**, a jit compiler for Python and NumPy)

## Models
A Neural Network can be considered as a stack of Layer instances. There are two possible configuration of models currently
available in Brainforge:

- BackpropNetwork, which uses a variant of Gradient Descent to optimize the weights
- NeuroEvolution, which uses differential evolution for optimization.

These classes expect the following parameters at initialization time:

- *input_shape*: tuple, specifying the dimensionality of the data. An InputLayer will be automatically instanciated
based on this information.
- *layers*: some iterable, holding Layer instances. Optional, layers can also be added to the network via the **add()**
method.
- *name*: string, specifying a name for the network, used in the **describe()** and **save()** methods.

#### Methods
For building the architecture:
- *add*: expects a Layer instance, which is added to the top of the layer stack.
- *pop*: deletes the last layer from the layer stack.
- *finalize*: finalizes the model, making it ready to fit the data.
  - *cost*: string or CostFunction inste, specifying the cost (or loss) function used to evaluate the network's
performance. See section: **Costs**

For model fitting:
- *fit*: fits the model to the given data
  - *X*: numpy array containing the training Xs (the independent variables)
  - *Y*: numpy array containing the training Ys (the dependent variables)
  - *batch_size*: the batch size of stochastic training. 1 for on-line (per-lesson) training. Defaults to 20.
  - *epochs*: the number of epochs to train for. Defaults to 30.
  - *monitor*: iterable, containing strings, specifying what runtime scores to monitor. Currently, only "acc" is supported.
Defaults to nothingness (aka void or nihil).
  - *validation*: tuple, containing separate X and Y tensors used to validate the network performance. Defualts to an
empty tuple.
  - *verbose*: integer, specifying the level of verbosity. Defaults to 1.
  - *shuffle*: bool, if set, the Xs and Ys will be reshuffled before every epoch. Defaults to True.
supported for monitoring the classification accuracy.
- *fit_generator*: fits the model to an infinite generator, which spits out batches of learning data.
- *epoch*: runs a single epoch on the supplied data.
- *learn_batch*: runs a forward and backward propagation and performs a single parameter update.
- *backpropagate*: only available for BackpropNetwork. Backpropagates a supplied delta tensor and calculates gradients.

For prediction and forward propagation:
- *prediction*: forward propagates a stimulus (X) throught the network, returning raw predictions.
  - *X*: the stimulus
- *classify*: wraps prediction by adding a final argmax to the computation, thus predicting a class from raw
probabilities.
- *evaluate*: evaluates the network's performance on the supplied tensors.
  - *X*: the input tensor
  - *Y*: the target tensor
  - *classify*: bool, determining whether a classification accuracy should be used or only cost determination.

Some utilities:
- *reset*: reshuffle the weights of the layers, effectively resetting the net
- *get_weights*: extract the paramters from the layers. If unfold is set, the parameters are returned as one big vector.
- *set_weights*: expects an iterable, ws. Sets the layers' weights according to **ws**. **ws** can be a flattened vector
if this method's **fold** argument is set.
- *get_gradients*: extracts gradients from the layers. Best to use together with *.backpropagate()*.
- *gradient_check*: performs numerical gradient check on the supplied X and Y tensors.
- *output*: property, returning the network's last output.
- *nparams*: property, returning the total number of parameters in the model.

### Autoencoder
TBD

## Layers
Neural network operations are implemented as *Layer* subclasses and an ANN can be thought of as a stack of *Layer*
instances.

### Core layers
These layers are the ones used most regularily when working with ANNs.
- **InputLayer**: passes the input tensor forward, unmodified.
  - No parameters required
  - This layer gets instantiated automatically
- **Flatten**: performs flattening on a *dims* dimensional tensor. Returns a matrix of shape
(*batch_size*, prod(*dims*))
  - No paramters required
- **Reshape**: reshapes the input tensor to a given shape. Keeps the leading dimension,
which corresponds to *batch_size*
  - No parameters required
- **DenseLayer**: just your regular densely connected layer.
  - *neurons*: integer, specifying the output dimensions
  - *activation*: string or ActivationFunction instance, specifying the activation function. Defaults to Linear.
- **Activation**: applies a (possibly) nonlinear activation function elementwise on the
input tensor.
  - *activation*: string, specifying the function to be used. Can be one of *sigmoid, tanh, relu* or *softmax*.

*Softmax* is only available with *categorycal crossentropy (xent)*, because the raw derivatives I worked out for this
function won't pass the numerical gradient test, so it's implemented the simplified form of *(a - y)*.

### Fancy layers
Some fancy feedforward layers from various scientific articles
- **HighwayLayer**: See Srivastava et al., 2015. This layer applies a gating mechanism to its inputs. This consist of
a forget gate and an input gate, determining the amount of information discarded and kept. Finally, it applies an
output gate, selecting the information to be passed on to the next layer. This operation doesn't change the
dimensionality of the inputs.
  - activation: string or ActivationFunction instance, specifying the activation function to be applied
- **DropOut**: Discards certain neurons in the training phase, thus improving generalization.
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
is partition into blocks. Each block is assigned a tick. At timestep t, only those blocks are activated, whose tick
value is a factor of t (e.g. t *mod* tick = 0, where *mod* is the modulo operation).
  - *neurons*: integer, specifying the output shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *blocksizes*: iterable, specifying the size and number of blocks in the layer. Sizes have to sum up to *neurons*.
Defaults to 5 blocks, with equal blocksizes. The remainder of the division-by-five is added to the first block.
  - *ticktimes*: iterable, specifying the tick time of the blocks. Has to contain as many > 0 elements as *blocksizes*.
Defaults to 2^i, where i is the (one-based) index of the block in question.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.
- **Reservoir**: an untrainable, specially initialized recurrent layer used in Echo State Networks.
See Reservoir computing and Jaeger, 2007 (Scholarpaedia)
  - *neurons*: integer, specifying the output (and the inner state's) shape of the layer.
  - *activation*: string or ActivationFunction instance, specifying the activation function.
  - *return_seq*: bool, determining whether to return every output (or inner state) generated (True) or to only return
the result of the last iteration (False). Defaults to False.


### Tensor
Feedforward layers working with multidimensional data

- **PoolLayer**: untrainable layer performing the max-pooling operation
  - *fdim*: integer, specifying the filter dimension. This value will also be used as the stride of the pooling operation.
  - *compiled*: bool, specifying whether to jit compile and optimize the layer with **numba** (highly recommended).
- **ConvLayer**: performs convolution/cross-correlation on a batch of images by learnable kernels of a given shape.
  - *nfilters*: integer, specifying the number of filters (neurons) to be learned
  - *filterx*: integer, specifying the X dimension of the filter
  - *filtery*: integer, specifying the X dimension of the filter
  - *compiled*: bool, specifying whether to jit compile and optimize the layer with **numba** (highly recommended).

These tensor processing layers are operating very slowly at the moment, especially PoolLayer.

## Optimizers
Currently the following optimizers are implemented:

- **SGD**: the classic stochastic gradient descent
  - *eta*: the learning rate, defaults to 0.01
- **Momentum**: momentum-accelerated gradient descent. Sums up all previous gradients and updates parameters according
to this accumulated term (velocity, in a physics sense), instead of the raw gradients.
  - *eta*: the learning rate, defaults to 0.1
  - *mu*: decay term for summed gradients, defaults to 0.9. Friction, from a physics standpoint.
  - *nesterov*: boolean argument, if set, the implemetation of the Nesterov Accelerated Gradient will be used.
It is very similar to normal Momentum, only this method "looks ahead" before updating the position. See Nesterov, 1983
for further details. Defaults to False.
- **Adagrad**: Adaptive Subgradient Method for stochastic optimization, see Duchi et al. 2011. An internal memory of all
prevoius squared gradients is kept and used to modify the general learning rate on a per-parameter basis.
  - *eta*: general learning rate, defaults to 0.01
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.
- **RMSprop**: the unpublished method of Geoffrey Hinton. See Lecture 6e of his Coursera Class. Keeps a decaying average
of past squared gradients and uses it to adjust a per-parameter learning rate.
  - *eta*: general learning rate, defaults to 0.1
  - *decay*: the memory decay term / update rate. Defaults to 0.9
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.
- **Adam**: Adaptive Moment Estimation, see Kingma et al., 2015. Keeps a decaying average of past gradients (velocity)
and of past squared gradients (memory).
  - *eta*: general learning rate, defaults to 0.1
  - *decay_memory*: the memory decay term / update rate. Defaults to 0.9
  - *decay_velocity*: the velocity decay term / update rate. Defaults to 0.999
  - *epsilon*: stabilizing term for the square root operation, defaults to 1e-8.

Adagrad and Adadelta are untested and seem to be faulty at the moment.

## Costs
The following cost functions are supported:

- **Mean Squared Error (MSE)**: can be used with any output activation, but can be slow to converge with saturated
sigmoids. For classification and regression tasks.
- **Categorical Cross-Entropy (Xent)**: can be used with *sigmoid* or *softmax* (0-1) output activations. It is intended to
use for classification tasks (softmax for multiclass, sigmoid for multilabel-multiclass).
- **Hinge loss (Hinge)**: for maximal margin convergence.

## Evolution
Support is available for evolutionary optimization. This technique can be used to either evolve optimal
hyperparameters for an ANN or to evolve the parameters (weights and biases) themselves (or both) (or for other purposes).
The feature can be accessed via the **brainforge.evolution** module, which defines the following class:
- **Population**: abstraction of a population of individuals
  - *loci*: integer, specifying the number of loci on the individuals' genome (the length of the genome).
  - *fitness_function*: a function reference, which will be called on every individual. This function has to return
a tuple/list/array of fitness values.
  - *fitness_weights*: an array, specifying the weights used when calculating the weighted sum of the fitness values,
thus creating a **grade** value representing the individuals' overall value.
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

## Examples
### Fit shallow net to the XOR problem
```python
import numpy as np

from brainforge import Network
from brainforge.layers import DenseLayer

def input_stream(m=20):
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(Xs), size=m)
        yield Xs[arg], Ys[arg]

net = Network(input_shape=(2,), layers=[
    DenseLayer(3, activation="sigmoid"),
    DenseLayer(2, activation="softmax")
])
net.finalize(cost="xent", optimizer="momentum")

datagen = input_stream(1000)
validation = next(input_stream(100))

net.fit_generator(datagen, 5000000, epochs=2, monitor=["acc"],
                  validation=validation, verbose=1)
```

For more complicated tasks, the use of the dataframe library csxdata is suggested.

### Fit LeNet-like ConvNet to images
```python
from csxdata import CData

from brainforge import Network
from brainforge.layers import (DenseLayer, DropOut, Activation,
                               PoolLayer, ConvLayer, Flatten)

dataroot = "path/to/pickled/gzipped/tuple/of/X/Y/ndarrays.pkl.gz"
images = CData(dataroot, indeps=0, headers=None)

inshape, outshape = images.neurons_required

model = Network(inshape, layers=(
    ConvLayer(nfilters=10, filterx=3, filtery=3, compiled=True),
    PoolLayer(fdim=2),
    Activation("relu"),
    ConvLayer(nfilters=10, filterx=5, filtery=5, compiled=True),
    PoolLayer(fdim=3),
    Activation("relu"),
    Flatten(),
    DenseLayer(120, activation="tanh"),
    DropOut(0.5),
    DenseLayer(outshape, activation="softmax")
))
model.finalize(cost="xent", optimizer="rmsprop")
X, Y = images.table("learning")
valid = images.table("testing")
model.fit(X, Y, batch_size=20, epochs=30, validation=valid,
          monitor=["acc"])
```

### Fit LSTM to text data
```python
from csxdata import Sequence

from brainforge import Network
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
model = Network(inshape, layers=(
    LSTM(120, activation="relu", return_seq=True),
    LSTM(60, activation="relu", return_seq=False),
    DenseLayer(120, activation="tanh"),
    DenseLayer(outshape, activation="softmax")
))
model.finalize(cost="xent", optimizer="rmsprop")
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

from brainforge import Network
from brainforge.layers import DenseLayer, DropOut
from brainforge.evolution import Population, to_phenotype

from csxdata import CData

dataroot = "/path/to/csv"
frame = CData(dataroot, headers=1, indeps=3, feature="FeatureName")

inshape, outshape = frame.neurons_required

# Genome will be the number of hidden neurons at two network DenseLayers.
ranges = ((10, 300), (0, 0.75), (10, 300), (0, 0.75))
# We determine 2 fitness values: the network's classification error and
# the time required to run the net. These two values will both be minimized
# and the accuracy will be considered with a 20x higher weight.
fweights = (200, 1)


def phenotype_to_ann(phenotype):
    net = Network(inshape, layers=[
        DenseLayer(int(phenotype[0]), activation="tanh"),
        DropOut(dropchance=phenotype[1]),
        DenseLayer(int(phenotype[2]), activation="tanh"),
        DropOut(dropchance=phenotype[3]),
        DenseLayer(outshape, activation="softmax")
    ])
    net.finalize(cost="xent", optimizer="momentum")
    return net


# Define the fitness function
def fitness(genotype):
    start = time.time()
    net = phenotype_to_ann(to_phenotype(genotype, ranges))
    net.fit_csxdata(frame, batch_size=20, epochs=30, verbose=0)
    score = net.evaluate(*frame.table("testing", m=10), classify=True)[-1]
    error_rate = 1. - score
    time_req = time.time() - start
    return error_rate, time_req


# Build a population of 12 individuals. grade_function and mate_function are
# left to defaults.
pop = Population(loci=4, limit=15,
                 fitness_function=fitness,
                 fitness_weights=fweights)
# The population is optimized for 12 rounds with the hyperparameters below.
# at every 3 rounds, we force a complete-reupdate of fitnesses, because the
# neural networks utilize randomness due to initialization, random batches, etc.
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
from csxdata import CData

from brainforge import NeuroEvolution
from brainforge.layers import DenseLayer

datapath = "path/to/data.csv"
data = CData(datapath, indeps=1, headers=1)
data.transformation = "std"

net = NeuroEvolution(data.neurons_required[0], layers=(
    DenseLayer(60, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="softmax")
))
net.finalize("xent", population_size=30, on_accuracy=False)

print("Initial acc:", net.evaluate(*data.table("testing"))[1])
net.fit(*data.table("learning"), batch_size=500, validation=data.table("testing"), monitor=["acc"])
```