# Intuitive Agency

`"To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes."`  

-David Chalmers, *Facing Up to The Problem of Consciousness*

## Introduction

General intelligence of the type we possess exists exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our conscious experience may be required if an artificial general intelligence is to one day be realized.

One defining aspect of human intelligence is our ability to subconsciously form new connections between abstract concepts which then seem to "bubble up" to the forefront of our attention. This phenomenon, commonly called intuition, is responsible not only for our most startling and profound "Aha!" moments, but also for the seemingly arbitrary changes in our awareness of, say, the ticking of a clock on the wall.

That we experience these shifts of attention unwillingly and "out of the blue" provides powerful clues to their underlying mechanisms. With that in mind, this project aims to develop an ensemble learning system capable of rudimentary intuition by modeling an agent who's "attention" is directed according to some optimization function (an intuition) that seeks to recognize the symbols in its seach-space. If successful, the agent may then be used in a larger network of such agents, thus bootstrapping an increasingly advanced intuition.

## Problem Domain

Currently the agent is attempting to learn the Python program language with the eventual goal of dynamically modifying itself via Pytyhon's ability for reflection.

## The Agent

The Agent was developed in Python (3.6). The 3rd party libraries KarooGP and PyTorch  were used for their genetic programming and machine learning capabilities, respectively.

### Usage

From the command line, run the agent with `./agent.py`. Performance metrics are displayed graphically as the agent runs. Try `./agent.py --help` for more options.

### Description

The agent operates from an "intuitive model" (image below) composed of three layers, labeled *Classification*, *Evolutionary*, and *Logical*. Data is mostly feed-forward, with recurrent feedback signaling the agent's current state and contextual fitness.
Each layer may persist to file and handle its own logging, depending on configurable options.

![The Intuitive Model](https://github.com/dustinfast/intuitive_agent/raw/master/docs/intutitive_model.png "The Intuitive Model")

### Documentation

For more information on the agent, its layers, example usage, problem domain, and current results, see the extensive inline code-level documentation and/or the primary documentation at `docs/documentation.docx`.

## File Structure

```
/
|   agent.py            - The top-level agent module
|   classifier.py       - An Aritificial Neural Net Classifier (e.g. level-one node)
|   connector.py        - The agent's environmental feedback functions / context-modes
|   genetic.py          - The genetic programming module
|   LICENSE             - GPLv3License
|   README.md           - This document
|   sharedlib.py        - Shared application-level library
|
+---docs - Application-level documentation
|      documentation.docx   - Application documentation
|
+---lib - 3rd party libraries
|
+---static - Static files (e.g. datasets)
|
+---var - Output files (logs, models, etc.)
```

## Design Philosophy

(Excerpt from `docs/documentation.docx`)

The intuitive model was conceived to represent a "sixth sense" interpretation of our intuition (no supernatural connotation intended), because (at least subjectively, from the perspective of our awareness) experiencing intuition “feels” no different than experiencing input from any of our other garden-variety five senses, except for at least one notable difference: intuitive input carries with it contextual meaning and symbolic comprehension about our environment - ideas composed by filtering sensory input through the sieve of one’s accumulated life experience. In this way it can be thought of as a sixth sensory organ, different from the first five in that it serves information that has been pre-processed by our sub-conscience.

Along with this sixth-sense interpretation, the agent was also designed with the following observations of human behavior in mind -

#### Observation 1

* We can react to events faster than we have time to logically determine a rational course of action (Klapproth, 2008). Regardless, we may still make very good “in-the-moment” approximations that seem to involve no conscious thought.

* We can articulate the rules we use to solve a given problem but are generally unable to explain why we chose to consider one specific set of rules over another (Pitrat, 2010).

* Shifts in changes to our awareness are often (if not always) autonomic (Shulman & Corbetta, 2002). For example, we cannot dictate when a song will become stuck in our head, or which clock we can suddenly hear ticking. 

* __CONCLUSION 1__  
Some system operating below our level of consciousness exists for selectively serving information into our awareness.

#### Observation 2

* Seemingly trivial and/or unproductive “mistakes” often come into our awareness. For example, songs DO get stuck in our head and we DO become suddenly aware of a ticking clock for no apparent reason. Contrapositively, we’re often oblivious to important environmental queues, especially when sufficiently distracted (a trait exploited by magicians and pick pockets alike).

* __CONCLUSION 2__  
Intuition is not perfect. However, "mistakes” have evolutionary value (e.g. genetic mutation compels biological adaptation). Therefore, as a biological system itself, it is reasonable to assume that the mechanism by which the subconscious learns to optimally processes and serve information is Darwinian in nature.

#### Observation 3

* The state of our awareness affects our intuition. Contrarywise, the state of our intuition affects our awareness. To demonstrate this, consider what occurs when focusing one’s awareness on a particular topic; we become acutely aware of it and much less aware of everything else.

* __CONCLUSION 3__  
Awareness and intuition exist as a feedback loop, each guiding each other in lockstep.

#### Observation 4

* We possess the ability hold, and operate on, a limited set of symbols in our head at one time. Summing three numbers in one’s head, for example.

* __CONCLUSION__ 4  
Humans possess a short-term “working memory” of symbols.

### Observation 5

* It is human nature to meticulously explore our environment for personal gain. As evidence of this, I offer humankind’s domination of its natural habitat through technological innovation.

* __CONCLUSION 5__  
An environmentally-aware agent motivated by evolutionary forces and possessing an intuition may naturally act to explore its environment and actively develop structure from it.

### Scalability 

Noting the hierarchal nature of information (including that of our problem domain) the model was designed to scale from a single agent to a collection of agent-nodes in a network of agents, thereby bootstrapping an increasingly advanced intuition. It is in this way that the intutive agent might eventually come to learn to write programs, including other versions and extensions of iteslf, in Python (image below).

![The Intuitive Model](https://github.com/dustinfast/intuitive_agent/raw/master/docs/scalable.png "The Intuitive Model")

#### Author: Dustin Fast, 2018
