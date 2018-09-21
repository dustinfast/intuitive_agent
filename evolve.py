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
        Evolver is the main interface, with KarooEvolver as the helper
        interface to Karoo GP and Model as a helper for logging and save/load.

    Usage:
        See __main__ for example usage.        

    Conventions:
        t = A tensor
        pop = Population
        indiv = Individual

    # TODO: 


    Author: Dustin Fast, 2018
"""

# Imports
from classlib import Model
import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp

# Constants
MODEL_EXT = '.tree'


class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP wrapper class.
        Based on https://github.com/kstaats/karoo_gp.py.
    """
    def __init__(self, 
                 kernel='c',            # (c)lassifier, (r)egression, or (m)atching
                 display='m',           # (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug
                 tree_pop_max=50,       # Maximum population size
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=3,      # Max tree depth
                 generation_max=1,      # Max generations to evolve
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

    def gen_first_pop(self, datafile, tree_type='f', tree_depth_base=3):
        """ Generates the initial population tree and fitness function.
            tree_type (char)        : (f)ull, (g)row, or (r)amped 50/50
            tree_depth_base (int)   : Initial population tree's depth
            data (str)              : CSV filename
        """ 
        # Load training data (data divided into train/validate sets internally)
        self.fx_karoo_data_load(tree_type, tree_depth_base, filename=datafile)
        
        # Construct the first population tree
        self.generation_id = 1
        self.population_a = ['Generation ' + str(self.generation_id)]
        self.fx_karoo_construct(tree_type, tree_depth_base)

        # Setup eval func and do eval on this initial population
        self._eval_first_pop()

    def _eval_first_pop(self):
        """ Evaluate the first generation of population trees.
            Returns null for divide by zero, else the simplified expression.
        """     
        # Setup eval expression, eval and compare fitness of first pop tree
        self.fx_fitness_gym(self.population_a)  

        # If only 1 generation or < 10 trees, finished. Else gen next pop
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
        
        # Done generating populations...
        self._show_menu()

    def _show_menu(self):
        """ Displays the Karoo "pause" menu, iff enabled.
        """
        if self.menu:
            self.fx_karoo_eol()


class Evolver(object):
    """ A genetically evolving numpy array.
    """
    def __init__(self, ID, console_out, persist):
        """ ID (str)                : This Evolver's unique ID number
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
        """
        # Generic object params
        self.ID = ID
        self.logger = None
        self.persist = persist
        self.model_file = None
        self.console_out = False

        self.evolver = KarooEvolve(menu=False)     # The karoo_gp interface
        self.array = None                          # The current array

        # Init the Model obj, which handles load, save, log, and console output
        # save_func = "self.evolver.fx_archive_tree_write(self.evolver.population_a, 'a')"
        # save_func2 = self.evolver.fx_archive_params_write('Desktop')
        self.model = Model(self,
                           console_out,
                           persist)
                        #    model_ext=MODEL_EXT,
                        #    save_func=save_func,
                        #    load_func=load_func)

    def __str__(self):
        str_out = 'ID = ' + self.ID + '\n'
        str_out += 'Data = ' + str(self.array)
        return str_out

    def forward(self):
        """ Sets the next "fittest" version of the array
        """
        raise NotImplementedError
       
    
if __name__ == '__main__':
    ev = Evolver('test_evolver', console_out=True, persist=False)
    ev.evolver.gen_first_pop(datafile='static/datasets/test_r.data')
