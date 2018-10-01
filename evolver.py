#!/usr/bin/env python
""" A module for genetically evolving a tensor using the Karoo GP library.

    Module Structure:
        Evolver is the main interface, with KarooEvolver as the helper
        interface to Karoo GP.
        Evolver persistence and output is handled by classlib.ModelHandler.
        
    Usage: 
        See "__main__" for example usage.

    Karoo GP Version Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version from https://github.com/kstaats/karoo_gp/pull/9.
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py') 

    # TODO: 
        Load/save model
        evolver.forward()

    Author: Dustin Fast, 2018
"""
import time  # debug
import json
import sympy
import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp

# from classlib import ModelHandler

MODEL_EXT = '.ev'

class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP wrapper class.
        Based on https://github.com/kstaats/karoo_gp.py.
    """

    def __init__(self,
                 # (c)lassifier, (r)egression, or (m)atching
                 kernel='r',
                 # (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug
                 display='m',
                 tree_pop_max=10,       # Maximum population size
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=10,     # Max tree depth
                 generation_max=10,     # Max generations to evolve
                 tourn_size=10,         # Individuals in each "tournament"
                 precision=6,           # Float points for fx_fitness_eval
                 menu=True              # Denotes Karoo GP menu enabled
                 ):
        """"""
        super(KarooEvolve, self).__init__()
        self.kernel = kernel
        self.display = display
        self.tree_pop_max = tree_pop_max
        self.tree_depth_min = tree_depth_min
        self.tree_depth_max = tree_depth_max
        self.generation_max = generation_max
        self.tourn_size = tourn_size
        self.precision = precision
        self.menu = menu

        # Init ratio of mutation types to be applied
        self.evolve_repro = int(0.1 * self.tree_pop_max)    # Reproductive
        self.evolve_point = int(0.1 * self.tree_pop_max)    # Point
        self.evolve_branch = int(0.1 * self.tree_pop_max)   # Branch
        self.evolve_cross = int(0.7 * self.tree_pop_max)    # Crossover

    def _show_menu(self):
        """ Displays the KarooGP "pause" menu, iff enabled.
        """
        if self.menu:
            self.fx_karoo_eol()

    def gen_first_pop(self, datafile, tree_type='f', tree_depth_base=5):
        """ Generates the initial population tree and fitness function.
            tree_type (char)        : (f)ull, (g)row, or (r)amped 50/50
            tree_depth_base (int)   : Initial population tree's depth
            data (str)              : CSV filename
        """
        # Load training data
        self.fx_karoo_data_load(tree_type, tree_depth_base, filename=datafile)

        # Construct the first generation of population trees
        self.generation_id = 1
        self.population_a = ['Generation ' + str(self.generation_id)]
        self.fx_karoo_construct(tree_type, tree_depth_base)

        # Setup fitness kernel and eval fitness of first population
        self.fx_fitness_gym(self.population_a)

        # Generate successive populations as specified
        if self.generation_max > 1:
            self._gen_next_pop(self.generation_max)

        # Show KarooGP menu (iff enabled)
        self._show_menu()

    def _evolve(self):
        """ Sets self.population_b to a newly evolved generation.
        """
        self.population_b = []          # The evolving/next generation
        self.fx_fitness_gene_pool()     # Init gene pool consttraints
        self.fx_karoo_reproduce()       # Do reproduction
        self.fx_karoo_point_mutate()    # Do point mutation
        self.fx_karoo_branch_mutate()   # Do branch mutation
        self.fx_karoo_crossover()       # Do crossover reproduction

    def _gen_next_pop(self, num_generations):
        """ Evolves the current population over "num_generations" successive 
            evolved generations.
            Accepts:
                num_generations (int)   : Number of generations to evolve
        """
        # Evolve population trees for each generation specified
        r_start = self.generation_id + 1
        r_end = self.generation_id + num_generations + 1
        for self.generation_id in range(r_start, r_end):
            self._evolve()                  # Evolve self.population_b
            self.fx_eval_generation()       # Eval all trees for fitness

            # Set curr population to the newly evolved population
            self.population_a = self.fx_evolve_pop_copy(self.population_b, [])

    def new_pop(self):
        """ Returns a new population bred from the current population. The
            current population is left intact/unmodified.
        """
        self._evolve()  # Evolve self.population_b
        return self.population_b


class AttentiveEvolver(object):
    """ An evolving population of expression trees used by the intutitve 
        agent's layer two to "mask" the input it receives before outputting
        iy to the agent's layer three.
    """
    def __init__(self, ID, console_out, persist, gp_args):
        """ ID (str)                : This Evolver's unique ID number
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
        """
        # Generic object params
        self.ID = ID
        self.persist = persist
        self.operands = None        # Sympy var labels, set on self.train()
        self.populaton = None       # The current expression tree population

        # The karoo_gp interface  # TODO: Tuneable
        self.evolver = KarooEvolve(**gp_args)

    def __str__(self):
        return 'ID = ' + self.ID

    def _save(self, filename):
        """ Saves a model of the expression tree. For use by ModelHandler.
        """
        print('Saving...')
        with open(filename, 'w') as f:
            f.write(self.operands + '\n')
            self._iter_trees(lambda x: f.write(x))
            
    def _load(self, filename):
        """ Loads the expression tree from file. For use by ModelHandler.
        """
        print('Loading...')

    def _iter_trees(self, f):
        """ Does f(tree) for every tree in the current population and returns
            the results, if any, as a list.
        """
        results = []
        # Iterate each tree in current population by ID (tree IDs start at 1)
        for treeID in range(1, len(self.evolver.population_a)):
            results.append(f(self.evolver.population_a[treeID]))
        return results

    def _get_sym_expr(self, tree):
        """ Returns the sympified expression of the given population tree.
        """
        self.evolver.fx_eval_poly(tree)
        return self.evolver.algo_sym

    def train(self, fname, epochs=10, ttype='r', start_depth=5, verbose=True):
        """ Evolves an initial population trained from the given file, where
            "fname" is a csv file with col headers "A, B, C, ..., s", where
            col 's' and denotes row "solution".
            Note: Many tuneable training parameters are set via the constructor
            Accepts:
                fname (str)         : The datafile path/fname
                type (str)          : Tree type - (g)grow, (f)ull, or (r)amp
                start_depth (int)   : Each tree's initial depth
                epochs (int)        : Number of training iterations
                verbose (bool)      : Denotes verbose output
        """
        info_str = 'population_sz=%d, ' % self.evolver.tree_pop_max
        info_str += 'treedepth_max=%d, ' % self.evolver.tree_depth_max
        info_str += 'treedepth_min=%d, ' % self.evolver.tree_depth_min
        info_str += 'tree_start_depth=%d, ' % start_depth
        info_str += 'epochs=%d, ' % epochs
        info_str += 'file=%s.' % fname
        print('Training started: ' + info_str)

        for i in range(epochs):
            if i == 0:
                # On epoch 1, generate initial population
                self.evolver.generation_max = 1
                self.evolver.gen_first_pop(datafile=fname,
                                           tree_type=ttype,
                                           tree_depth_base=start_depth)
            else:
                # Else generate successive generations
                self.evolver._gen_next_pop(1)
            
            # TODO: if verbose:
                # self.model.log()

        # Denote operands from csv col headers (excluding solutions)
        self.operands = [t for t in self.evolver.terminals if t != 's']

        print('Training complete. Final population:\n')
        self._iter_trees(lambda x: print(str(self._get_sym_expr(x))))

        if self.persist:
            self._save(self.ID + MODEL_EXT)

    def forward(self, inputs):
        """ For each tree in the population, peforms that tree's expression on
            the given list of inputs. 
            Returns: a list of lists, one for each expression result.
        """
        
        # debug - print the next 2 populations
        # for i in range(0, 2):
        #     print('\n*** Next population: ' + str(i))
        #     population = self.evolver.new_pop()
        #     for treeID in range(1, len(population)):
        #         print(self._get_sym_expr(population[treeID]))
        #         expr = str(self._get_sym_expr(population[treeID]))
        #         if 'A + B + C + D + E' in expr:
        #             print('Found: ' + expr)
        # print('ORIGINAL POP:')
        # self._iter_trees(lambda x: print(str(self._get_sym_expr(x))))

        # results = []    # Results container
        # processed = []  # Contains evaluated expressions, to avoid duplicates

        # # Advance the population
        # population = self.evolver.new_pop()

        # # Iterate each expression in the population
        # for tree in range(1, len(population)):
        #     self.evolver.fx_eval_poly(population[tree])

        #     # Ensure unique expression
        #     if self.evolver.algo_sym in processed:
        #         continue
        #     processed.append(self.evolver.algo_sym)
            
        #     # alg_sym will be an expr str. Ex: -C - 3*D + 2*s - E
        #     expr = str(self.evolver.algo_sym).replace('', '')
        #     print(expr)

        # tree = ast.parse(expr, mode='eval').body

    def update(self, fitness):
        """ Evolves a new population based on the given fitness metrics.
        """
        # Set each trees fitness param, then do gen_next_pop
        if self.persist():
            # do save..
            pass


if __name__ == '__main__':
    # Define KarooGP params
    gp_args = {'display': 'm',
               'kernel': 'r',
               'tree_pop_max': 50,
               'tree_depth_min': 20,
               'tree_depth_max': 40,
               'menu': False}

    # Init and train the evolver
    ev = AttentiveEvolver('test_evolver', True, False, gp_args)
    ev.train('static/datasets/nouns_sum.csv', epochs=3, ttype='r')

    ev.forward(None)

    # TODO: for i in range... do mass test
