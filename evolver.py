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
        evolver.validate()
        evolver.update()
        evolver.forward results should denote which inputs were in context?

    Author: Dustin Fast, 2018
"""

import re
from random import randint
import sys; sys.path.append('lib')
import karoo_gp.karoo_gp_base_class as karoo_gp

from classlib import ModelHandler

MODEL_EXT = '.ev'

class KarooEvolve(karoo_gp.Base_GP):
    """ A Karoo GP wrapper class.
        Based on https://github.com/kstaats/karoo_gp.py.
    """
    def __init__(self,        # (c)lassifier, (r)egression, or (m)atching
                 kernel='r',  # (i)ntrctv, (g)nrtn, (m)in, (s)ilent, or (d)ebug
                 display='m',
                 tree_pop_max=10,       # Maximum population size
                 tree_depth_min=3,      # Min nodes of any tree
                 tree_depth_max=10,     # Max tree depth
                 generation_max=10,     # Max generations to evolve
                 tourn_size=10,         # Individuals in each "tournament"
                 precision=6,           # Float points for fx_fitness_eval
                 write_runs=False,      # Denotes Karoo GP records run info
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
        self.write_runs = write_runs
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

        # Setup kernel and eval first population's fitness
        self.fx_fitness_gym(self.population_a)

        # Generate successive populations as specified
        if self.generation_max > 1:
            self._gen_next_pop(self.generation_max)

        # Show KarooGP menu (iff enabled)
        self._show_menu()

    def _evolve(self):
        """ Sets self.population_b to a newly evolved generation.
        """
        
        self.fx_karoo_crossover()       # Do crossover reproduction

    def _gen_next_pop(self, num_generations):
        """ Evolves the current population over "num_generations" generations. 
            Accepts:
                num_generations (int)   : Number of generations to evolve
        """
        # Evolve population trees for each generation specified
        r_start = self.generation_id + 1
        r_end = self.generation_id + num_generations + 1
        for self.generation_id in range(r_start, r_end):
            self.population_b = []          # The evolving/next generation
            self.fx_fitness_gene_pool()     # Init gene pool consttraints
            self.fx_karoo_reproduce()       # Do reproduction
            self.fx_karoo_point_mutate()    # Do point mutation
            self.fx_karoo_branch_mutate()   # Do branch mutation
            self.fx_eval_generation()       # Eval all trees for fitness

            # Set curr population to the newly evolved population
            self.population_a = self.fx_evolve_pop_copy(
                self.population_b, 'Generation ' + str(self.generation_id))

    def trees_byfitness(self):
        """ Returns a list of the current population's tree ID's, sorted by
            fitness (L to R).
        """
        rev = {'min': False, 'max': True}.get(self.fitness_type)
        trees = [t for t in range(1, len(self.population_a))]
        trees = sorted(
            trees, key=lambda x: self.population_a[x][12][1], reverse=rev)
        return trees

    def sym_expr(self, treeID):
        """ Returns the sympified expression of the given population tree.
        """
        self.fx_eval_poly(self.population_a[treeID])  # Update self.algo_sym
        return self.algo_sym

    def expr_strings(self):
        """ Returns the current population's sympy expressions in string form.
        """
        results = ''
        for treeID in range(1, len(self.population_a)):
            expr = str(self.sym_expr(treeID))
            results += 'Tree ' + str(treeID) + ': ' + expr + '\n'
        return results


class Evolver(object):
    """ An evolving population of expression trees used by the intutitve 
        agent's layer-two to "mask" the input it receives before outputting
        it to the agent's layer-three.
    """
    def __init__(self, ID, console_out, persist, gp_args):
        """ ID (str)                : This object's unique ID number
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
        """
        # Generic object params
        self.ID = ID
        self.persist = persist
        self.ops = None             # Expression operand labels
        # TODO: self.train_tree = None       # Last training tree type, set on train()
        # TODO: self.train_min_depth = None  # Last training min depth, set on train()

        # The karoo_gp interface. See class KarooEvolve (above) for args
        self.gp = KarooEvolve(**gp_args)

        # Init the load, save, log, and console output handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, console_out, persist,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        return 'ID = ' + self.ID

    def _save(self, filename):
        """ Saves a model of the expression tree. For use by ModelHandler.
        """
        with open(filename, 'w') as f:
            # Write model params in dict form
            f.write("{'operands': " + str(self.ops))
            f.write(", 'kernel': '" + str(self.gp.kernel) + "'")
            f.write(", 'tree_depth_max': " + str(self.gp.tree_depth_max))
            f.write(", 'tree_depth_min': " + str(self.gp.tree_depth_min))
            f.write(", 'tourn_size': " + str(self.gp.tourn_size))
            f.write(", 'tree_pop_max': " + str(self.gp.tree_pop_max))
            f.write(", 'generation_id': " + str(self.gp.generation_id))
            f.write("}\n")
            
            # Write population
            f.write(str(self.gp.population_a))
            
    def _load(self, filename):
        """ Loads the expression tree from file. For use by ModelHandler.
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

        # Init self from the params and population
        params = eval(params)
        self.ops = params['operands'] 
        self.gp.fx_karoo_load_raw(params, population)

    def train(self, fname, epochs=10, ttype='r', start_depth=5, verbose=False):
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
        # TODO: Update "last trained" vars
        # self.train_tree = ttype
        # self.train_min_depth = start_depth

        # Output log stmt
        info_str = 'kernel=%s, ' % self.gp.kernel
        info_str += 'population_sz=%d, ' % self.gp.tree_pop_max
        info_str += 'treetype=%s, ' % ttype
        info_str += 'treedepth_max=%d, ' % self.gp.tree_depth_max
        info_str += 'treedepth_min=%d, ' % self.gp.tree_depth_min
        info_str += 'tree_start_depth=%d, ' % start_depth
        info_str += 'epochs=%d, ' % epochs
        info_str += 'file=%s.' % fname
        self.model.log('Training started: ' + info_str)

        for i in range(epochs):
            if i == 0:
                # On epoch 1, generate initial population
                self.gp.generation_max = 1
                self.gp.gen_first_pop(datafile=fname,
                                      tree_type=ttype,
                                      tree_depth_base=start_depth)
            else:
                # Generate successive populations
                self.gp._gen_next_pop(1)
            
            if verbose:
                t = self.gp.expr_strings()
                self.model.log('Training epoch %d generated:\n%s' % (i, t))

        # Denote operands from csv col headers (excluding solutions)
        self.ops = [t for t in self.gp.terminals if t != 's']

        t = self.gp.expr_strings()
        self.model.log('Training complete. Final population:\n%s' % t)
       
        if self.persist:
            self.model.save()

    def forward(self, inputs, n_results=10, split=.8):
        """ Peforms each tree's expression on the given inputs and returns 
            the results as a dict denoting the source tree ID
            Accepts:
                inputs (list)   : A list of lists, one for each input "row"
                n_results (int) : Max number of results to return
                split (float)   : Ratio of fittest expression results to
                                    randomly chosen expressions
            Returns:
                dict: { treeID: [result1, result2, ... ], ... }
        """
        try:
            trees = self.gp.trees_byfitness()  # leftmost = most fit
        except AttributeError:
            self.model.log('ERROR: Forward attempted on uninitialized model.')
            exit(-1)

        # If strings in input, rm exprs w/neg operators - they're nonsensical
        for inp in inputs:
            if not [i for i in inp if type(i) is str]: break
        else:
            trees = [t for t in trees if '-' not in str(self.gp.sym_expr(t))]

        # Filter trees having duplicate expressions
        added = set()
        trees = [t for t in trees 
                 if str(self.gp.sym_expr(t)) not in added and
                 (added.add(str(self.gp.sym_expr(t))) or True)]

        # At this point, every t in trees is useable, so do fit/random split
        fit_count = int(n_results * split)
        split_trees = trees[:fit_count]     
        rand_pool = trees[fit_count:]

        while len(split_trees) < n_results and rand_pool:
            idx = randint(0, len(rand_pool))
            split_trees.append(rand_pool.pop(idx))

        # Perform each tree's expression on the given inputs
        results = {}  # Results container { treeID: [result1, ... ] }
        for treeID in trees:
            results[treeID] = []  # Tree-specific results container
            expr = str(self.gp.sym_expr(treeID))

            # Reform the expr by mapping each operand to an input index - Ex:
            #   If expr = 'A + B + 2*D + F + E', then
            #   new_expr = 'row[0] + row[1] + 2*row[3] + row[5] + row[4]'
            new_expr = ''
            for ch in expr:
                if ch in self.ops:
                    new_expr += 'row[' + str(self.ops.index(ch)) + ']'
                else:
                    new_expr += ch
            expr = re.split("[+\-]+", expr)

            # print('Processing tree ' + str(treeID) + ': ' + expr)  # debug
            # print(new_expr)  # debug

            # Eval reformed expr against each input, noting the source tree ID
            for row in inputs:
                try:
                    res = eval(new_expr)  # row var used inside eval()
                    # print(row)  # debug
                    # print(res)  # debug
                    results[treeID].append(res)
                except IndexError:
                    pass  # The inputs are too short (may occur in debug)

        # Filter out trees w/no results
        results =  {k: v for k, v in results.items() if v}
        # print(results)  # debug

        return results

    def update(self, fit_trees, online=False):
        """ Evolves a new population after favorably weighting fitness of each 
            tree denoted by "fit_trees". Evolution occurs in a seperate thread
            to avoid blocking, as it is computationally expensive.
            Accepts:
                fit_trees (list)    : ID (int) of each tree to favor
                online (bool)       : Denotes if update causes learning
        """
        max_fitness = {'min': 0, 'max': 9999999.0}.get(self.gp.fitness_type)

        for treeID in fit_trees:
            self.gp.population_a[treeID][12][1] = max_fitness

        for treeID in range(1, len(self.gp.population_a)):
            print(str(self.gp.sym_expr(treeID)))
            print(self.gp.population_a[treeID][12][1])

        # Advance the population # TODO: In a new thread
        self.gp._gen_next_pop(1)

        for treeID in range(1, len(self.gp.population_a)):
            print(str(self.gp.sym_expr(treeID)))
            print(self.gp.population_a[treeID][12][1])

        if online:
            # TODO: learning params to model file
            if self.persist:
                self.model.save()


if __name__ == '__main__':
    # Define the training/validation files
    trainfile = 'static/datasets/words_sum.csv'
    valfile = 'static/datasets/words.dat'

    # Define KarooGP parameters - see KarooEvolver() for possible args.
    gp_args = {'display': 'm',
               'kernel': 'r',
               'tree_pop_max': 50,
               'tree_depth_min': 5,
               'tree_depth_max': 20,
               'menu': False}

    # Init and train the evolver
    ev = Evolver('test_gp', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=15, ttype='r', start_depth=5, verbose=True)
    # ev.validat(valfile)

    # Example inputs
    inputs = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
              ['L', 'K', 'J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']]

    # Example forward and update
    results = ev.forward(inputs)
    print(results)
    fitness = [k for k in results.keys()]
    ev.update(fitness)

    # debug - file builder
    # f_out = open('static/datasets/words.dat', 'w')
    # with open('static/datasets/words_1.txt', 'r') as f:
    #     for line in f:
    #         if len(line) > 1 and len(line) < 11 and '/' not in line and ord(line[1]) > 97 and ord(line[2]) > 97:
    #             # ev.forward([line])
    #             f_out.write(line)
    # f_out.close()

    # debug - test runs
    # trainfile = 'static/datasets/words_sum.csv'
    # print('\n\n******* REGRESSION *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'r',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_rsum_sd3_100e', console_out=False, 
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # print('\n\n******* MATCHING *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'm',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_msum_sd3_100e', console_out=False,
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # print('\n\n******* CLASSIFIER *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'c',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_csum_sd3_100e', console_out=False,
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # trainfile = 'static/datasets/words_cat.csv'
    # print('\n\n******* REGRESSION CAT *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'r',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_rcat_sd3_100e', console_out=False,
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # print('\n\n******* CLASSIFIER CAT*******')
    # gp_args = {'display': 'm',
    #            'kernel': 'c',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_ccat_sd3_100e', console_out=False,
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # print('\n\n******* MATCHING CAT*******')
    # gp_args = {'display': 'm',
    #            'kernel': 'm',
    #            'tree_pop_max': 100,
    #            'tree_depth_min': 3,
    #            'tree_depth_max': 30,
    #            'menu': False}

    # ev = Evolver('test_gp_mcat_sd3_100e', console_out=False,
    #              persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=3, verbose=True)

    # print('\n\n******* DEEP REGRESSION *******')
    # gp_args = {'display': 's',
    #            'kernel': 'r',
    #            'tree_pop_max': 50,
    #            'tree_depth_min': 20,
    #            'tree_depth_max': 50,
    #            'menu': False}
    # ev = Evolver('test_gp_sum_rgdeep', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='g', start_depth=8, verbose=True)
