# Intuitive Agency

## Inspiration

`"To explain the integration of information, we need only exhibit mechanisms by which information is brought together and exploited by later processes."`  

-David Chalmers, *Facing Up to The Problem of Consciousness*

## Objective

General intelligence of the type we possess exists exclusively, for now, in the domain of the conscious human. Therefore, an understanding of the mechanisms leading to our conscious experience may be required if an artificial general intelligence is to one day be realized.

One defining aspect of human intelligence is our ability to subconsciously form new connections between abstract concepts which then seem to "bubble up" to the forefront of our attention. This phenomenon, commonly called intuition, is responsible not only for our most startling and profound "Aha!" moments, but also for the seemingly arbitrary changes in our awareness of, say, the ticking of a clock on the wall.

That we experience these shifts of attention unwillingly and "out of the blue" provides powerful clues to their underlying mechanisms. With that in mind, this project aims to develop an ensemble learning system capable of rudimentary intuition by modeling an agent who's "attention" is directed by its "intuition". If successful, the agent may then be used in a larger network of such agents, thus bootstrapping an increasingly advanced intuition.

The agent is developed in Python using the PyTorch and KarooGP machine learning and genetic programming libraries, respectively.

## Problem Domain

Currently, the agent is attempting to learn the Python program language, with the eventual goal of dynamically modifying itself via Pytyhon's ability for reflection.

## The Agent

The agent is composed of three layers, labeled *Classifier*, *Evolutionary*, and *Logical*. Data is mostly feed-forward, with recurrent feedback signaling the agent's current state and contextual fitness.
Each layer may persist to file and handle its own logging, depending on configurable options.

For more infomration on the agent, its layers, example usage, problem domain , and current results, see the extensive inline code-level documentation, example module uses (found in each modules `if __main__`), and the main documentation file at `docs/documentation.docx`.

__Author__: Dustin Fast, 2018
