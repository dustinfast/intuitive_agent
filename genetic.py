#!/usr/bin/env python
""" A module for genetically evolving a population of expression trees using 
    the Karoo GP library.

    Module Structure:
        GPMask is the main interface.
        Persistence and output handled by classlib.ModelHandler.
        
    Karoo GP Version Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version from https://github.com/kstaats/karoo_gp/pull/9.
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py') 

    # TODO: 
        forward() results should denote which inputs were in prev context?
        train(trainfile, epochs=5, verbose=True)
        validate(valfile)

    Author: Dustin Fast, 2018
"""

import re
from random import randint
import sys; sys.path.append('lib')

from numpy import array
import karoo_gp.karoo_gp_base_class as karoo_gp

from classlib import ModelHandler

MODEL_EXT = '.ev'


class GPMask(karoo_gp.Base_GP):
    """ An evolving population of expressions used by the intutitve agent
        to "mask" data between layers two and three. Learns in an online, or
        can be pre-trained.
    """
    def __init__(self, ID, max_pop, max_depth, input_sz, console_out, persist):
        """ ID (str)                : This object's unique ID number
            max_pop (int)           : Max number of expression trees
            max_depth (int)         : Max tree mutate depth
            input_sz (int)          : Max number of inputs to expect
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
        """
        super(GPMask, self).__init__()
        self.ID = ID
        self.persist = persist
        self.tree_pop_max = max_pop
        self.tree_depth_max = max_depth
        self.input_sz = input_sz

        # Application specific KarooGP params
        self.display = 's'                      # Silence KarooGP output
        self.tree_type = 'r'                    # Allow full and sparse trees
        self.fitness_type = 'max'               # "Maximizing" fitness kernel
        self.tourn_size = int(max_pop / 3)      # Fitness tourny size
        self.functions = array([['+', '2']])    # Expression operators/arity
        self.precision = 6                      # Fitness floating points

        # Terminal symbols - one ucase letter for each input, plus label(s)
        self.terminals = [chr(i) for i in range(65, min(91, 65 + input_sz))]
        self.terminals += ['s']
        
        # Mutation ratios
        self.evolve_repro = int(0.1 * self.tree_pop_max)
        self.evolve_point = int(0.1 * self.tree_pop_max)
        self.evolve_branch = int(0.1 * self.tree_pop_max)
        self.evolve_cross = int(0.7 * self.tree_pop_max)

        # Init the load, save, log, and console output handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, console_out, persist,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        # Init first generation if not already loaded by ModelHandler
        try:
            self.population_a
            self.pop_tree_type = 'g'
        except AttributeError:
            self.generation_id = 1
            self.population_a = ['Generation 1']
            self.fx_karoo_construct(self.tree_type, 3)
            for treeID in range(1, len(self.population_a)):
                self.population_a[treeID][12][1] = 0
    
    def __str__(self):
        str_out = 'ID = ' + self.ID + '\nSize = (\n  '
        str_out += 'max_depth = ' + str(self.tree_depth_max) + '\n  '
        str_out += 'max_pop = ' + str(self.tree_pop_max) + '\n  '
        str_out += 'inputs = ' + str(self.input_sz) + '\n)'
        return str_out

    def _save(self, filename):
        """ Saves a model of the current population. For use by ModelHandler.
        """
        with open(filename, 'w') as f:
            # Write model params to file in dict form
            f.write("{'operands': " + str(self.terminals))
            f.write(", 'tree_depth_max': " + str(self.tree_depth_max))
            f.write(", 'tourn_size': " + str(self.tourn_size))
            f.write(", 'tree_pop_max': " + str(self.tree_pop_max))
            f.write(", 'generation_id': " + str(self.generation_id))
            f.write("}\n")
            
            # Write current population to file
            f.write(str(self.population_a))
            
    def _load(self, filename):
        """ Loads model & population from from file. For use by ModelHandler.
        """
        # Build params and population strings from the given file
        params = ''
        population = ''
        param_flag = True
        with open(filename, 'r') as f:
            for line in f:
                for ch in line:
                    if param_flag:
                        params += ch
                        if ch == '}': param_flag = False
                    else:
                        population += ch

        # Restore population
        self.population_a = eval(population)

        # Restore params (note: not all params written need restored)
        for k, v in eval(params).items():
            if k == 'operands':
                self.terminals = v
            elif k == 'generation_id':
                self.generation_id = v

    def _trees_byfitness(self):
        """ Returns a list of the current population's tree ID's, sorted by
            fitness (L to R).
        """
        rev = {'min': False, 'max': True}.get(self.fitness_type)
        trees = [t for t in range(1, len(self.population_a))]
        trees = sorted(
            trees, key=lambda x: self.population_a[x][12][1], reverse=rev)
        return trees

    def _symp_expr(self, treeID):
        """ Returns the sympified expression of the tree with the given ID
        """
        self.fx_eval_poly(self.population_a[treeID])  # Updates self.algo_sym
        return self.algo_sym

    def _raw_expr(self, treeID):
        """ Returns the raw expression of the tree with the given ID
        """
        self.fx_eval_poly(self.population_a[treeID])  # Updates self.algo_raw
        return self.algo_raw

    def _expr_strings(self, symp_expr=False, w_fit=False):
        """ Returns the current population's expressions as one string. 
            Accepts:
                w_fit (bool)     : Also includes each tree's fitness in string
                symp_expr (bool) : Uses sympyfied expression, vs raw expr
        """
        if symp_expr:
            f_expr = self._symp_expr
        else:
            f_expr = self._raw_expr

        results = ''
        for treeID in range(1, len(self.population_a)):
            expr = str(f_expr(treeID))
            results += 'Tree ' + str(treeID) + ': ' + expr + '\n'

            if w_fit:
                fit = str(self.population_a[treeID][12][1])
                results += ' (fitness: ' + fit + ')\n'

        return results

    # def train(self, fname, epochs=10, verbose=False):
    #     """ Evolves an initial population trained from the given file, where
    #         "fname" is a csv file with col headers "A, B, C, ..., s", where
    #         col 's' and denotes row "solution".
    #         Accepts:
    #             fname (str)         : The datafile path/fname
    #             type (str)          : Tree type - (g)grow, (f)ull, or (r)amp
    #             start_depth (int)   : Each tree's initial depth
    #             epochs (int)        : Number of training iterations
    #             verbose (bool)      : Denotes verbose output
    #     """
    #     # Output log stmt
    #     info_str = 'population_sz=%d, ' % self.tree_pop_max
    #     info_str += 'treetype=%s, ' % self.tree_type
    #     info_str += 'treedepth_max=%d, ' % self.tree_depth_max
    #     info_str += 'treedepth_min=%d, ' % self.tree_depth_min
    #     info_str += 'epochs=%d, ' % epochs
    #     info_str += 'file=%s.' % fname
    #     self.model.log('Training started: ' + info_str)

    #     for i in range(epochs):
    #         # TODO train from file
            
    #         if verbose:
    #             t = self._expr_strings()
    #             self.model.log('Training epoch %d generated:\n%s' % (i, t))

    #     t = self._expr_strings()
    #     self.model.log('Training complete. Final population:\n%s' % t)
       
    #     if self.persist:
    #         self.model.save()

    def forward(self, inputs, max_results=0, split=.8, ordered=False):
        """ Peforms each tree's expression on the given inputs and returns 
            the results as a dict denoting the source tree ID
            Accepts:
                inputs (list)     : A list of lists, one for each input "row"
                max_results (int) : Max results to return (0=population size)
                split (float)     : Ratio of fittest expressions used to
                                      randomly chosen expressions used
                ordered (bool)    : Denotes if the inputs considered sequential
            Returns:
                dict: { treeID: [result1, result2, ... ], ... }
        """
        try:
            trees = self._trees_byfitness()  # leftmost = most fit
        except AttributeError:
            raise Exception('Forward attempted on an uninitialized model.')

        # If ordered specified, use raw expression, else use sympified
        if ordered:
            f_expr = self._symp_expr
        else:
            f_expr = self._raw_expr

        # If strings in "input", rm exprs w/neg operators - they're nonsensical
        for inp in inputs:
            if not [i for i in inp if type(i) is str]: 
                break
        else:
            trees = [t for t in trees if '-' not in str(f_expr(t))]

        # Filter trees having duplicate expressions
        added = set()
        trees = [t for t in trees if str(f_expr(t)) not in added and
                 (added.add(str(f_expr(t))) or True)]

        # At this point, every t in trees is useable, so do fit/random split
        # Determine max results
        if not max_results: max_results = self.tree_pop_max
        fit_count = int(max_results * split)
        split_trees = trees[:fit_count]
        rand_pool = trees[fit_count:]

        while len(split_trees) < max_results and rand_pool:
            idx = randint(0, len(rand_pool) - 1)
            split_trees.append(rand_pool.pop(idx))

        # Iterate every tree that hasn't been filtered out
        results = {}
        for treeID in trees:
            results[treeID] = []
            expr = str(f_expr(treeID))

            # Reform the expr by mapping each operand to an input index
            #   Ex: expr 'A + 2*D + B' -> 'row[0] + 2*row[3] + row[1]'
            new_expr = ''
            for ch in expr:
                if ch in self.terminals:
                    new_expr += 'row[' + str(self.terminals.index(ch)) + ']'
                else:
                    new_expr += ch
            expr = re.split("[+\-]+", expr)

            # debug
            # print('Processing tree ' + str(treeID) + ': ' + str(expr))
            # print(new_expr)

            # Eval reformed expr against each input, noting the source tree ID
            for row in inputs:
                results[treeID].append(eval(new_expr))

        # Filter out trees w/no results and return
        return {k: v for k, v in results.items() if v}

    def update(self, fitness):
        """ Evolves a new population of trees after updating the fitness of 
            each existing tree's expression according to "fitness". 
            Accepts:
                fitness (dict)    : Each key (int) is a tree ID and each value
                                    denotes it's new fitness value
        """
        # Give each tree a baseline fitness score
        for treeID in range(1, len(self.population_a)):
                self.population_a[treeID][12][1] = 0
                
        # Update tree's fitness as given by "fitness" arg
        for k, v in fitness.items():
            self.population_a[k][12][1] = v

        # print(self._expr_strings(w_fit=True))  # debug

        # Build the new gene pool
        self.gene_pool = [t for t in range(1, len(self.population_a))
                          if self._symp_expr(t)]

        # Evolve a new population
        self.population_b = []
        self.fx_karoo_reproduce()
        self.fx_karoo_point_mutate()
        self.fx_karoo_branch_mutate()
        self.fx_karoo_crossover()
        self.generation_id += 1
        self.population_a = self.fx_evolve_pop_copy(
            self.population_b, 'Generation ' + str(self.generation_id))

        if self.persist:
            self.model.save()


if __name__ == '__main__':
    # Define the training/validation files
    # trainfile = 'static/datasets/words_sum.csv'
    # valfile = 'static/datasets/words.dat'

    # Init the genetically evolving expression trees
    ev = GPMask('test_gp', 25, 15, 4, console_out=True, persist=False)

    # Example inputs
    inputs = [['A', 'B', 'C', 'D'],
              ['D', 'A', 'C', 'B']]

    ordered = False     # Denote inputs should not be considered sequential
    epochs = 30         # Learning epochs

    # Example "online" learning - forward(), update fitness, and update()
    for z in range(0, epochs):
        print('*** Epoch %d ***' % z)
        # Get results, according to current population
        results = ev.forward(inputs, ordered=ordered)

        # Output resulting trees and results to console
        print(ev._expr_strings(symp_expr=ordered))
        print(results)

        # Update fitness of each expression, depending on results
        fitness = {k: 0 for k in results.keys()}
        for k, v in results.items():
            for j in v:
                # If first two chars are 'DC' and len < 5, note as desirable
                if j[0] == 'D' and j[1] == 'C' and len(j) < 5:
                    fitness[k] += 1

        # Evolve a new population with the new fitness values
        ev.update(fitness)
