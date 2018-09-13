# Intuitive Agency

Author: Dustin Fast, CSCE A470, Fall 2018.

## Inspiration

`To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes.`  

-David Chalmers, *Facing Up to The Problem of Consciousness*

## Objective

General intelligence of the type we possess exists exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our consciouses experience may be required if an artificial general intelligence is to one day be realized.

One defining aspect of human intelligence is our ability to subconsciously form new connections between abstract concepts, which then seem to "bubble up" to the forefront of our attention. This phenomenon, commonly called intuition, is responsible not only for our most startling "Aha!" moments, but also for the seemingly arbitrary changes in our awareness of, say, the ticking of a clock on the wall.

That we experience these changes of awareness unwillingly provides powerful clues to the underlying mechanisms of intuition and consciousness. With that in mind, this project aims to develop an ensemble learning system capable of rudimentary intuition by modeling an agent who's "attention" switches contexts according to its "intuition". If successful, the agent may then be used in a larger network of such agents, thus simulating ever-more advanced intuition.

## Implementation

The agent shall be implemented in Python using readily available machine learning and evolutionary programming libraries.

## The Agent

The agent exists as three layers, labeled *Conceptual*, *Intuitive*, and *Attentive*, as described below and given by Diagram 1 (attached). Data is mostly feed-forward, with recurrent feedback channels existing for the back-propagation of information representing the agent's current state and fitness.

## The Conceptual, Intuitive, and Attentive Layers

### Conceptual Layer

The conceptual layer **represents our ability to form abstract symbolic concepts**, from patterns in the environment via sensory input. It consists of a set `A` of artificial neural networks (ANNs) with the following properties -


Each ANN at this level has a set `X` of input nodes, defined as:

| X         | Channel        |  Description |
|-------------|-------------| -------------|
| `x_0` - `x_m-1` | Feed-forward | Environmental sensory input, identical for every `a` in `A` |
| `x_m` - `x_n`   | Feedback    | Each `x_m` to `x_n` maps onto a corresponding attentive-layer input node. This represent feedback of the agent's current context. |

Each ANN is pre-trained (offline) to classify a different class of objects (ex: `a_0` classifies digits, `a_1` classifies letters, etc). During this pre-training, each `x_0` to `x_m` should be randomized (or possibly non-existent).

Each ANN's output nodes are subsequently used as intuitive-layer input.

### Intuitive Layer

The intuitive layer is a set of data pipes, one for each conceptual-layer ANN, connecting the conceptual layer to the attentive layer. On each state change, each pipe is weighted according to some fitness function that evolves in an online manner according to some genetic algorithm.

These weights are subsequently used as a bias (possibly binary, logarithmic, etc.) by the attentive layer. In this way, the intuitive layer learns how to best allocate the agent's "attention" while simultaneously allowing "mistakes" to enter its awareness. These mistakes **represent our ability to subconsciously make new connections between existing abstract concepts**.

### Attentive Layer

The attentive layer is a singular ANN **representing the current context of our attention**. Context may be defined here as the agent's level of "awareness" of each symbolic concept present in the environment.

The attentive-layer ANN has a set `M` of input nodes, where `size(M) = size(A) * size(X)`. Each `m` is biased according to the weight assigned at the intuitive level.

The attentive-layer ANN is not pre-trained. Learning at this level is done in an online fashion specific to the problem at hand (TDB).

