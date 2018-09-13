# Modeling Intuition

`To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes.`
-David Chalmers, *Facing Up to The Problem of Consciousness*

## Introduction

### Assumptions

Assumes consciousness to be an emergent property of the brain.
Assumes that conscious agents do not exert control over their current context, rather the current context is dictated by the current brain-body state according to current and previous sensory input and selected context.

### Intuition and Generalized Intelligence

General intelligence of the type we possess seems to reside exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our consciousness may be required if an artificial general intelligence is to one day be realized.

One defining aspect of human-level intelligence is our ability to form conceptual connections subconciously which then "bubble up" to the forefront of our attention. This process, commonly called intuition, is responsible not only for our most startling "Aha!" moments, but also for our seemingly arbitrary and sudden recognition of, say, the ticking of a clock on the wall. That we experience these two states unwillingly provides powerful clues to the underlying mechanisms of intuition and consciousness.

With these things in mind, this project will build a model of an intuituve agent who's "attention" is focused according to its "intuition".

## The Model

The model consists of three subsystems, each existing as part of a larger recurrent feedback loop.

according to some evolutionary fitness function, modeled by a genetic algorithm. our apparent lack of ability to focus on more than one "context" at a time, and the realization that moving from one context to the next is often (if not always) completely subconscious. With this in mind, I have developed the following abstraction of a "conscious" system. It consists of three modules, each existing as part of a single recurrent feedback loop.

1. Sensory-Input - A collection of pattern recognizers, implemented as recurrent neural networks (RNNs). Each RNN represents one of our five senses.
2. Intuition - The genetic algorithm. Assigns weights to each sensory input and enques (7 deep, +-2). Responsible for allocating our "attention-time" in a way analogous to a modern CPU allocating processor-time among multiple programs. This system likely allocates each attention "clock-tick" according to some evolutionary fitness function.In this way, 
3. Analyst - Accepts highest weighted input. What does it do with the input? Can it override the weights? We can't choose to focus on one context or another, rather, the environment dictates

DIAGRAM


## Objectives
Conclude consciouscness may be an illusion
Provide an ensemble learning system that may be helpful in understanding the underlying mechanisms associated with human consciousness.
Gain experience with PyTorch, Numpy, Keras, … 

