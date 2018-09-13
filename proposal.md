# Modeling Inuitive Agency

`To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes.`  
-David Chalmers, *Facing Up to The Problem of Consciousness*

## Introduction

General intelligence of the type we possess seems to reside exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our consciouses experience may be required if an artificial general intelligence is to one day be realized.

One defining aspect of human-level intelligence is our ability to form conceptual connections subconsciously which then "bubble up" to the forefront of our attention. This process, commonly called intuition, is responsible not only for our most startling "Aha!" moments, but also for our seemingly arbitrary and sudden recognition of, say, the ticking of a clock on the wall. That we experience these two states unwillingly provides powerful clues to the underlying mechanisms of intuition and consciousness.

With these things in mind, this project aims to build an intuitive agent who's "attention" is focused according to its "intuition".

## The Agent

The agent exists holistically as a recurrent feedback loop consisting of three subsystems, labeled *Conceptual*, *Intuitive*, and *Analytical*. It is to be implemented as a state machine according to the following diagram -

DIAGRAM

## The Conceptual, Intuitive, and Analytical Subsystems

### Conceptual

The conceptual subsystem represents the human ability to recognize patterns in the environment via sensory input. It consists of a set of n artificial neural networks (ANNs), each having k inputs.

Each ANN receives identical input to its k-1 inputs, representing the agent's current sensory input.  

Each ANN's k-1'th input is received as feedback from the analytical subsystem and denotes whether or not that ANNs output was part of the agent's current "context". In this way, Each ANNs output is dependent on current sensory input as well as the agent's current context.

Each ANN's output is fed to the intuitive subsystem as input.

### Intuitive

The intuitive subsystem "decides" which context the analytical subsystem receives. It is composed of 3 layers, labeled *input*, *filter*, and *output*:

1. **Input**: A set of n FIFO queues of depth d. Each queue, q_i, receives input from ANN a_i. The queue implementation represents the limited working memory of a human !!!!!!as each element in the queue is used as input to each ANN.

2. **Filter**: The filter pops an element from each input-layer pipe, assigns it a weight, and the weighted elements to the output layer. The weight assigned is dictated by an evolutionary function implemented via a genetic algorithm. HEURISTIC??

In this way, the filter is responsible for allocating the agents "attention-time" in a way that may be analogous to a modern CPU allocating processor-time among multiple programs. Unlike a CPU however, the genetic algorithm allows "mistakes" to enter into our consciousness and facilitate new connections between conceptual objects.



### Analytical

The agent is modeled as a state machine due to our apparent lack of ability to focus on more than one "context" at a time.  and the realization that moving from one context to the next is often (if not always) completely subconscious.

Analyst - Accepts highest weighted input. What does it do with the input? Can it override the weights? We can't choose to focus on one context or another, rather, the environment dictates


## 


## Objectives
Conclude consciouscness may be an illusion
Provide an ensemble learning system that may be helpful in understanding the underlying mechanisms associated with human consciousness.
Gain experience with PyTorch, Numpy, Keras, … 

