#!/usr/bin/env python
""" A module for genetically evolving a tensor using the Karoo GP library.

    If persistent mode enabled, Evolver state persists between executions via
    PERSIST_PATH/ID.MODEL_EXT and status msgs logged to PERSIST_PATH/ID.LOG_EXT

    Karoo GP Verision Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version here: https://github.com/kstaats/karoo_gp/pull/9)
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py')

    Structure:
        EvolveTensor is the main interface, with KarooEvolver as the interface
        to Karoo GP

    Dependencies:
        TensorFlow
        numpy
        sympy
        scikit-learn
        matplotlib

    Conventions:
        t = A tensor
        pop = Population
        indiv = Individual


    # TODO: 
        Encapsulate main contents into ANN.from_csv()



    Author: Dustin Fast, 2018
"""

import os
import logging
import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp


PERSIST_PATH = 'var/evolve/'    # Evolver model and log file path
MODEL_EXT = '.ev'               # Evolver model file extension
LOG_EXT = '.log'                # Evolver log file extension


class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP wrapper class.
        Based on https://github.com/kstaats/karoo_gp.py.
    """
    def __init__(self, 
                 kernel='c',            # (c)lassifier, (r)egression, or (m)atching
                 display='s',           # (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug
                 tree_pop_max=50,       # Maximum population size
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=3,      # Max tree depth
                 generation_max=10,     # Max generations to evolve
                 tourn_size=10,         # Individuals in each "tournament"
                 precision=6,           # Float points for fx_fitness_eval
                 menu=False             # Denotes Karoo GP "pause" menu enabled
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
        self.evolve_point = int(0.0 * self.tree_pop_max)    # Point
        self.evolve_branch = int(0.2 * self.tree_pop_max)   # Branch
        self.evolve_cross = int(0.7 * self.tree_pop_max)    # Crossover

    def get_best(self, tourn_size=10):
        """ Returns winner of a new tourney of tourney_size random contenders.
        """
        return self.fx_fitness_tournament(tourn_size)

    def gen_first_pop(self, datafile='', tree_type='f', tree_depth_base=3):
        """ Generates the initial population tree and fitness function.
            tree_type (char)        : (f)ull, (g)row, or (r)amped 50/50
            tree_depth_base (int)   : Initial population tree's depth
            datafile (str)          : Data set filename
        """ 
        # Load training and validation data (divided into train/val internally)
        self.fx_karoo_data_load(tree_type, tree_depth_base, datafile)
        
        # Construct the first population tree
        self.generation_id = 1
        self.population_a = ['Generation ' + str(self.generation_id)]
        self.fx_karoo_construct(tree_type, tree_depth_base)

        # self.fx_display_tree(self.tree)  # debug
        self._eval_first_pop()  # Setup eval func and do eval on populations

    def _eval_first_pop(self):
        """ Evaluate the first generation of population trees.
            Returns null for divide by zero, else the simplified expression.
        """     
        # Setup eval expression, eval and compare fitness of first pop tree
        self.fx_fitness_gym(self.population_a)  

        # Done if only 1 generation or < 10 trees requested. Else gen next pop
        if self.tree_pop_max < 10 or self.generation_max == 1:
            self._show_menu
        else:
            self._gen_next_pop()

    def _gen_next_pop(self):
        """ Generates a new genetically evolved population from the current.
        """
        # Evolve population trees for each generation specified
        for self.generation_id in range(2, self.generation_max + 1):
            self.population_b = []          # The evolving/next generation
            self.fx_fitness_gene_pool()     # Init gene pool contstraints
            self.fx_karoo_reproduce()       # Do reproduction
            self.fx_karoo_point_mutate()    # Do point mutation
            self.fx_karoo_branch_mutate()   # Do branch mutation
            self.fx_karoo_crossover()       # Do crossover reproduction
            self.fx_eval_generation()       # Eval all trees of curr generation

            # Set curr population to the newly evolved population
            self.population_a = self.fx_evolve_pop_copy(self.population_b, [])
        
        self._show_menu()  # Done generating requested number of gens

    def _show_menu(self):
        """ Does fx_karoo_eol(), AKA the Karoo "pause" menu, iff enabled.
        """
        if self.menu:
            self.fx_karoo_eol()


class EvolveTensor(object):
    """ A genetically evolving tensor.
    """
    def __init__(self, ID, karoo_evolve, **kwargs):
        """ ID (str)                    : This Evolver's unique ID number

            **kwargs:
                persist (bool)          :   Persit mode flag
                console_out (bool)      :   Output log stmts to console flag
        """
        # Generic object params
        self.ID = ID
        self.model_file = None
        self.logger = None
        self.persist = False
        self.console_out = False

        self.evolver = karoo_evolve     # The interface to Karoo GP
        self.curr_tensor = None         # The current version of the tensor

        # kwarg handlers...
        if kwargs.get('console_out'):
            self.console_out = True

        if kwargs.get('persist'):
            self.persist = True
            if not os.path.exists(PERSIST_PATH):
                os.mkdir(PERSIST_PATH)

            # Init logger and output init statment
            logging.basicConfig(filename=PERSIST_PATH + ID + LOG_EXT,
                                level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self._log('*** Evolver initialized ***:\n' + str(self))

            # Init, and possibly load, model file
            self.model_file = PERSIST_PATH + ID + MODEL_EXT
            if os.path.isfile(self.model_file):
                self.load()

    def __str__(self):
        return self

    def _log(self, log_str, level=logging.info):
        """ Logs the given string to the Evolver's log file, iff in persist mode.
            Also outputs the string to the console, iff in console_out mode.
        """
        if self.persist:
            level(log_str)

        if self.console_out:
            print(log_str)

    def save(self):
        """ Saves a model of the Evolver.
        """
        try:
            raise NotImplementedError
            # self.fx_archive_tree_write(self.evolver.population_a, 'a')
            # self.fx_archive_params_write('Desktop')
            self._log('Saved Evolver model to: ' + self.model_file)
        except Exception as e:
            self._log('Error saving model: ' + str(e), level=logging.error)

    def load(self):
        """ Loads a model of the Evolver.
        """
        try:
            raise NotImplementedError
            self._log('Loaded Evolver model from: ' + self.model_file)
        except Exception as e:
            self._log('Error loading model: ' + str(e), level=logging.error)
            exit(0)

    def forward(self, t):
        """ Returns the next "fittest" evolved version of the given tensor.
        """
        raise NotImplementedError
        # self.curr_tensor = None
       
    
if __name__ == '__main__':
    karoov = KarooEvolve(menu=False)
    karoov.gen_first_pop(datafile='static/datasets/test_headers.data')
    ev = EvolveTensor('test_evolver', karoov)
