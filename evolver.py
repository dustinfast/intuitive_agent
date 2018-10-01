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
        evolver.update()
        remove run writes from karoo base class

    Author: Dustin Fast, 2018
"""
import sympy
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
            self.population_a = self.fx_evolve_pop_copy(
                self.population_b, 'Generation ' + str(self.generation_id))

    def new_pop(self):
        """ Returns a new population bred from the current population, leaving
            the current population intact/unmodified.
        """
        self._evolve()  # Evolve self.population_b
        return self.population_b


class Evolver(object):
    """ An evolving population of expression trees used by the intutitve 
        agent's layer two to "mask" the input it receives before outputting
        iy to the agent's layer three.
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

    def _iter_pop(self, f):
        """ Does f(tree) for every tree in the current population and returns
            the results as a list.
            Note: Tree IDs start at 1
        """
        results = []
        for treeID in range(1, len(self.gp.population_a)):
            results.append(f(self.gp.population_a[treeID]))
        return results

    def _get_sym_expr(self, tree):
        """ Returns the sympified expression of the given population tree.
        """
        self.gp.fx_eval_poly(tree)  # Update the gp.algo_sym
        return self.gp.algo_sym

    def _expr_strs(self):
        """ Returns the current population's sympy expressions in string form.
        """
        results = ''
        trees = self._iter_pop(lambda x: str(self._get_sym_expr(x)))
        for i, tree in enumerate(trees):
            results += 'Tree ' + str(i) + ': ' + tree + '\n'
        return results

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
        info_str = 'population_sz=%d, ' % self.gp.tree_pop_max
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
                t = self._expr_strs()
                self.model.log('Training epoch %d generated:\n%s' % (i, t))

        # Denote operands from csv col headers (excluding solutions)
        self.ops = [t for t in self.gp.terminals if t != 's']

        t = self._expr_strs()
        self.model.log('Training complete. Final population:\n%s' % t)
       
        if self.persist:
            self.model.save()

    def forward(self, inputs):
        """ For each tree in the population, peforms that tree's expression on
            the given list of inputs. 
            Returns: a list of lists, one for each expression result.
        """
        
        # debug - print the next 2 populations
        # for i in range(0, 2):
        #     print('\n*** Next population: ' + str(i))
        #     population = self.gp.new_pop()
        #     for treeID in range(1, len(population)):
        #         print(self._get_sym_expr(population[treeID]))
        #         expr = str(self._get_sym_expr(population[treeID]))
        #         if 'A + B + C + D + E' in expr:
        #             print('Found: ' + expr)

        results = {}    # Results container: { treeID: [result1, ... ] }
        processed = []  # Contains evaluated expressions, to avoid duplicates

        # Iterate each expression in the current population
        population = self.gp.population_a
        for treeID in range(1, len(population)):
            self.gp.fx_eval_poly(population[treeID])  # Update gp.algo_sym
            results[treeID] = []                      # Tree results container

            # Get the tree's sympy expression str. Ex: "-C - B + 3*D + 2*E + F"
            expr = str(self.gp.algo_sym) 

            # Ensure unique expression w/no nonsensical
            if expr in processed or '-' in expr:
                continue
            processed.append(self.gp.algo_sym)
            print('Processing tree ' + str(treeID) + ': ' + expr)  # debug

            # Reform the expression by mapping each operand to an input index
            # Ex new_expr: row[0] + row[1] + 2*row[3] + row[5] + row[4]
            new_expr = ''
            for ch in expr:
                if ch in self.ops:
                    new_expr += 'row[' + str(self.ops.index(ch)) + ']'
                else:
                    new_expr += ch
            expr = new_expr.split('+')
            print(new_expr)

            # Eval reformed expr against each input, noting the source tree ID
            for row in inputs:
                try:
                    results[treeID].append((row, eval(new_expr)))
                except IndexError:
                    pass  # The inputs are too short (may occur in debug)
        print(results)

    def update(self, fitness):
        """ Evolves a new population based on the given fitness metrics.
            New population evolution occurs in a seperate thread for perf.
            Accepts:
                fitness (list)  : ID (int) of each tree confirmed fit
        """
        # Set each tree's fitness param, then do gen_next_pop
        if self.persist():
            # do save..
            
            # Advance the population # TODO: In a new thread
            population = self.gp.new_pop()


if __name__ == '__main__':
    # Define the training file - see Evolver.train() for format info.
    trainfile = 'static/datasets/nouns_sum.csv'

    # Define KarooGP parameters - see KarooEvolver() for possible args.
    gp_args = {'display': 's',
               'kernel': 'r',
               'tree_pop_max': 50,
               'tree_depth_min': 20,
               'tree_depth_max': 15,
               'menu': False}

    # Init and train the evolver
    ev = Evolver('test_gp_min', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=5, ttype='r', start_depth=5, verbose=True)

    # # Example inputs
    # inputs = [[['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    #           ['G', 'F', 'E', 'D', 'C', 'B', 'A']]]

    # # Example forward
    # ev.forward(inputs) 

    # debug
    # f_out = open('static/datasets/words.dat', 'w')
    # with open('static/datasets/words_1.txt', 'r') as f:
    #     for line in f:
    #         if len(line) > 1 and len(line) < 11 and '/' not in line and ord(line[1]) > 97 and ord(line[2]) > 97:
    #             # ev.forward([line])
    #             f_out.write(line)
    # f_out.close()
            

    
    # debug
    # print('\n\n******* REGRESSION *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'r',
    #            'tree_pop_max': 50,
    #            'tree_depth_min': 15,
    #            'tree_depth_max': 25,
    #            'menu': False}

    # ev = Evolver('test_gp_r', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=5, verbose=True)

    # print('\n\n******* MATCHING *******')
    # gp_args = {'display': 'm',
    #            'kernel': 'm',
    #            'tree_pop_max': 50,
    #            'tree_depth_min': 15,
    #            'tree_depth_max': 25,
    #            'menu': False}

    # ev = Evolver('test_gp_m', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=30, ttype='r', start_depth=5, verbose=True)

    # gp_args = {'display': 's',
    #             'kernel': 'r',
    #             'tree_pop_max': 50,
    #             'tree_depth_min': 20,
    #             'tree_depth_max': 25,
    #             'menu': False}

    # print('\n\n******* DEEP REGRESSION *******')
    # ev = Evolver('test_gp_sd7g', console_out=True, persist=True, gp_args=gp_args)
    # ev.train(trainfile, epochs=3, ttype='g', start_depth=7, verbose=True)
