# Intuitive Agency

`"To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes."`  

-David Chalmers, *Facing Up to The Problem of Consciousness*

## Introduction

General intelligence of the type we possess exists exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our conscious experience may be required if an artificial general intelligence is to one day be realized.

One defining aspect of our intelligence is the ability to subconsciously form new connections between abstract concepts which seem to “bubble up” into our awareness. This phenomenon, commonly called intuition, is responsible, not only for our most profound "Aha!" moments, but also for the sudden, seemingly-arbitrary changes in awareness we routinely experience of, say, the ticking of a clock on the wall.

An ensemble learning system was developed to explore mechanisms through which this intuitive behavior might emerge, with a system agent applied to the task of classifying known search-space symbols and stochastically optimizing a combinatorial generator in order to quickly find the logical connections among them according to its predefined context.

The idea here is not to provide these connections as output for human analysis, rather it is to demonstrate an agent with the ability to adapt to new environments in an intuitive way. I.e., the conclusions reached by the agent are irrelevant as long as they're conducive to its survival.

In its current form, the agent is attempting to learn the Python program language with the eventual goal of dynamically modifying itself via Python's ability for reflection.

## Design Paradigm

Although intuition exists inside us as a “black box” (we cannot consciously observe its decision-making processes), evolutionary programming techniques implemented according to observations of our behavior and biology may allow us to converge on an approximate solution. Towards this end, a model of intuition was conceived, based on the following assumptions -

* Some system, operating at the sub-conscious level, exists for selectively serving information into awareness.
* Awareness and intuition exist in a feedback loop, each influencing the other.
* Intuition is not perfect.
* Mistakes have evolutionary utility.
* The subconscious is likely optimized by processes that are Darwinian in nature.
* An agent possessing an intuition may naturally act to explore, and seek to understand, its environment.

In this context, intuition can be thought of as a sixth sensory organ (no supernatural connotations intended), different from the first five in that the information it serves is pre-processed by the sub-conscience and carries with it contextual meaning and symbolic comprehension: ideas composed by filtering sensory information through the sieve of one’s accumulated life experiences.

## The Intuitive Model

The agent operates from an intuitive model composed of three layers, labeled *Classification*, *Evolutionary*, and *Logical*. Data is mostly feed-forward, with recurrent feedback signaling contextual fitness.

![The Intuitive Model](https://github.com/dustinfast/intuitive_agent/raw/master/static/img/model.png "The Intuitive Model")

### Scalability

The agent was designed to scale from a single agent to a node in a hierarchy of agents in order to bootstrap an increasingly advanced intuition. In this way, a sufficiently complex agent might come to write its own programs and/or re-write itself according to search-heuristics in real-time via Python’s capability for reflection.

![Agent Hierarchy](https://github.com/dustinfast/intuitive_agent/raw/master/static/img/scalable.png "Agent Hierarchy")


## Usage

From the command line, run the agent with `./agent.py`. Performance metrics are displayed graphically as the agent runs. Try `./agent.py --help` for more options.

For more information on the agent, its layers, usage, problem domain, and data campaigns, see the primary documentation at <https://github.com/dustinfast/intuitive_agent/raw/master/docs/pdf/documentation.pdf>.

Additionally, the code-base contains extensive inline documentation.

## Technologies

The application was developed in Python (3.6). The 3rd party libraries KarooGP and PyTorch were used for their genetic programming and machine learning functionalities, respectively.

### Dependencies

| Dependency    | Installation                              |
|---------------|-------------------------------------------|
|KarooGP        | N/A (lib/karoo_gp)                        |
|Matplotlib     | pip install matplotlib                    |
|Numpy          | pip install numpy                         |
|Pandas         | pip install pandas                        |
|PyTorch        | see https://pytorch.org                   |
|Requests       | pip install requests                      |
|Scikit-learn   | pip install scikit-learn                  |
|Sympy          | pip install sympy                         |
|TensorFlow     | see https://www.tensorflow.org/install    |
|Scipy          | pip install scipy                         |

## File Hierarchy

```
/
|   agent.py            - The top-level agent module
|   classifier.py       - Aritificial neural network module
|   connector.py        - Context mode / environmental feedback functions
|   genetic.py          - Genetic programming module
|   LICENSE             - GPLv3License
|   README.md           - This document
|   sharedlib.py        - Shared classes and functions
|
+---docs - Application documentation
|
+---lib - 3rd party libraries
|
+---static - Static files (e.g. datasets, images)
|
+---var - Output files (logs, models, etc.)
```

#### Author: Dustin Fast, 2018
