#!/usr/bin/env python
""" A genetic algorithm to evolve a tensor using the Karoo GP library.

    Karoo GP Verision Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version here: https://github.com/kstaats/karoo_gp/pull/9)
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py')

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


    Author: Dustin Fast, 2018
"""

import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp


class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP library wrapper.
        Based on karoo_gp.py at: https://github.com/kstaats/karoo_gp
    """
    def __init__(self, 
                 kernel='c',            # (c)lassification, (r)egression, or (m)atching
                 data='',               # Data input
                 max_pop_sz=100,        # Maximum population size
                 tree_type='f',         # (f)ull, (g)row, or (r)amped 50/50
                 tree_depth_base=3,     # Initial population tree depth
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=3,      # Max tree depth
                 generations_max=10,    # Max generations to evolve
                 display='m',           # (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug
                 tourny_sz=10,          # Individuals in each "tournament"
                 precision=6            # Float points for fx_fitness_eval
                 ):
        """
        """
        super(KarooEvolve, self).__init__()
        self.kernel = kernel
        self.tree_pop_max = max_pop_sz
        self.tourn_size = tourny_sz     
        self.precision = precision     
        self.tree_type = tree_type
        self.tree_depth_base = tree_depth_base
        self.tree_depth_max = tree_depth_max
        self.tree_depth_min = tree_depth_min
        self.generation_max = generations_max
        self.data = data
        self.display = display

        # Ratio between mutation types (must sum to 0)
        self.evolve_repro = int(0.1 * self.tree_pop_max)    # Reproductive
        self.evolve_point = int(0.0 * self.tree_pop_max)    # Point
        self.evolve_branch = int(0.2 * self.tree_pop_max)   # Branch
        self.evolve_cross = int(0.7 * self.tree_pop_max)    # Crossover

    def _gen_first_pop(self):
        """ Generates the initial population tree.
        """ 
        print('Generating first population')

        self.fx_karoo_data_load(self.tree_type, self.tree_depth_base, self.data)
        
        self.generation_id = 1  # Unique generation ID
        self.population_a = ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)] # an empty list which will store all Tree arrays, one generation at a time
        # self.population_a = []  # Tree array container, by generation

        # construct the first population of Trees
        self.fx_karoo_construct(self.tree_type, self.tree_depth_base)
        print('Constructed Gen 1 population of', self.tree_pop_max, ' trees\n')

        self.fx_display_tree(self.tree)
        # self.fx_archive_tree_write(self.population_a, 'a')  # save tree to disk

        self._eval_first_pop()

    def _eval_first_pop(self):
        """ Evaluate the first generation of population trees.
            Returns null for divide by zero, else the simplified expression.
        """     
        print('Evaluating the first generation of trees')
        self.fx_fitness_gym(self.population_a)  # generate expression, evaluate fitness, compare fitness
        self.fx_archive_tree_write(self.population_a, 'a')  # Save this gen of trees to disk

        # Done if only 1 generation or l.t. 10 trees requested. Else continue.
        if self.tree_pop_max < 10 or self.generation_max == 1:
            # self.fx_archive_params_write('Desktop')  # run-time params to disk
            self.fx_karoo_eol()
        else:
            self._gen_next_pop()

    def _gen_next_pop(self):
        """ Generates a new genetically evolved population from the previous.
        """
        for self.generation_id in range(2, self.generation_max + 1):
            print('\n Evolving pop of Trees for Gen ', self.generation_id, '...')

            self.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation

            # Generate viable gene pool by evolving the prior generation
            self.fx_fitness_gene_pool()
            self.fx_karoo_reproduce()       # Do reproduction
            self.fx_karoo_point_mutate()    # Do point mutation
            self.fx_karoo_branch_mutate()   # Do branch mutation
            self.fx_karoo_crossover()       # Do crossover reproduction
            self.fx_eval_generation()       # Eval all trees of curr generation

            self.population_a = self.fx_evolve_pop_copy(self.population_b, ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)])
        
        self.fx_karoo_eol()

    def _eval_fitness(self, p, f):
        """ Evaluates the given tensor for fitness, based on heuristic func f.
        """
        raise NotImplementedError

    def _select_indiv(self):
        raise NotImplementedError

    def _avg_fitness(self):
        raise NotImplementedError

    def _best_fitness(self):
        raise NotImplementedError

    def forward(self):
        self._gen_first_pop()


if __name__ == '__main__':
    ev = KarooEvolve()
    ev.forward()
