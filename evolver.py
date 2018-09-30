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
        Apply tensor ops for each input node according to genetic expr

    Author: Dustin Fast, 2018
"""

import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp

# from classlib import ModelHandler

MODEL_EXT = '.tree'

class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP wrapper class.
        Based on https://github.com/kstaats/karoo_gp.py.
    """
    def __init__(self, 
                 kernel='r',            # (c)lassifier, (r)egression, or (m)atching
                 display='m',           # (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug
                 tree_pop_max=10,       # Maximum population size
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=10,     # Max tree depth
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
        self.evolve_point = int(0.1 * self.tree_pop_max)    # Point
        self.evolve_branch = int(0.1 * self.tree_pop_max)   # Branch
        self.evolve_cross = int(0.7 * self.tree_pop_max)    # Crossover

    def get_best(self, tourn_size=10):
        """ Returns winner of new tourney of "tourney_size" random contenders.
        """
        return self.fx_fitness_tournament(tourn_size)

    def gen_first_pop(self, datafile, tree_type='r', tree_depth_base=5):
        """ Generates the initial population tree and fitness function.
            tree_type (char)        : (f)ull, (g)row, or (r)amped 50/50
            tree_depth_base (int)   : Initial population tree's depth
            data (str)              : CSV filename
        """ 
        # Load training data
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
            self.fx_fitness_gene_pool()     # Init gene pool consttraints
            self.fx_karoo_reproduce()       # Do reproduction
            self.fx_karoo_point_mutate()    # Do point mutation
            self.fx_karoo_branch_mutate()   # Do branch mutation
            self.fx_karoo_crossover()       # Do crossover reproduction
            self.fx_eval_generation()       # Eval all trees of curr generation

            # Set curr population to the newly evolved population
            self.population_a = self.fx_evolve_pop_copy(self.population_b, [])
        
        # Done generating populations...
        self._show_menu()
    
    def next_pop(self):
        """ Returns a new population bred from the current population. 
            Differs from _gen_next_pop in that no "learning" takes place.
        """
        # Evolve population trees for each generation specified
        self.population_b = []          # The evolving/next generation
        self.fx_fitness_gene_pool()     # Init gene pool consttraints
        self.fx_karoo_reproduce()       # Do reproduction
        self.fx_karoo_point_mutate()    # Do point mutation
        self.fx_karoo_branch_mutate()   # Do branch mutation
        self.fx_karoo_crossover()       # Do crossover reproduction
        self.fx_eval_generation()       # Eval all trees of curr generation

        # Set curr population to the newly evolved population
        return self.population_b

    def _show_menu(self):
        """ Displays the Karoo "pause" menu, iff enabled.
        """
        if self.menu:
            self.fx_karoo_eol()


class AttentiveEvolver(object):
    """ A handler for an evolving expression tree in the context of the 
        intutive agent.
    """
    def __init__(self, ID, console_out, persist):
        """ ID (str)                : This Evolver's unique ID number
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
        """
        # Generic object params
        self.ID = ID
        self.model_file = None
        self.persist = persist

        # The karoo_gp interface
        self.evolver = KarooEvolve(display='m',
                                   kernel='r',
                                   tree_pop_max=50,
                                   tree_depth_min=15,
                                   tree_depth_max=10,
                                   generation_max=2,
                                #    generation_max=20,
                                   menu=False)  

    def __str__(self):
        return 'ID = ' + self.ID

    def train(self, datafile):
        """ Trains the evolver from the given data file.
            Accepts:
                datafile (str)      : The data file name/path
        """
        self.evolver.gen_first_pop(datafile=datafile,
                                   tree_type='r', 
                                   tree_depth_base=5)

    def forward(self, inputs):
        """ Performs the evolver's expressions on the given inputs.
            Returns a list of lists, one for each of the evolvers expressions.
        """
        results = []    # Results container
        processed = []  # Contains evaluated expressions, to avoid duplicates

        # Advance the population
        population = self.evolver.next_pop()

        # Iterate each 
        for tree in range(1, len(population)):
            self.evolver.fx_eval_poly(population[tree])
            
            if self.evolver.algo_sym in processed:
                continue

            # alg_sym will be an expr str. Ex: -C - 3*D + 2*s - E
            expr = self.evolver.algo_sym.replace('', '')
            result = inputs



       
    
if __name__ == '__main__':
    ev = AttentiveEvolver('test_evolver', console_out=True, persist=False)
    ev.train('static/datasets/nouns_sum.csv')
