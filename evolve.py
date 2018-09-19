#!/usr/bin/env python
""" A genetic algorithm to evolve a tensor using the Karoo GP library.

    Conventions:
        t = Some arbitrary tensor
        p = pop = Population
        indiv = Individual

    # TODO: 


    Author: Dustin Fast, 2018
"""

import random
import operator
# import matplotlib.pyplot as plt


class TensorEvolve(object):
    """ A genetic algorithm for tensor evolution.
    """
    def __init__(self, shape):
        raise NotImplementedError

    def _eval_fitness(self, p, f):
        """ Evaluates the given tensor for fitness, based on heuristic func f.
        """
        raise NotImplementedError

    def _gen_first_pop(self):
        """ Generates a random first population of appropriate shape.
        """
        raise NotImplementedError

    def _gen_next_pop(self, population):
        """ Generates a genetically evolved population from the given population.
        """
        raise NotImplementedError

    def _mutate_indiv(self):
        """ Mutates and returns the given individual.
        """
        raise NotImplementedError

    def _mutate_pop(self, chance_of_mutation):
        """ Mutates the given tensor according to ...
        """
        raise NotImplementedError

    def _select_indiv(self):
        raise NotImplementedError

    def _avg_fitness(self):
        raise NotImplementedError

    def _best_fitness(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
        