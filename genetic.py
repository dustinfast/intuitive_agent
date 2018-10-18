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
        forward() need to denote which inputs are in curr context
        forward() perf improvements
        input_sz currently limited to <= 26

    Author: Dustin Fast, 2018
"""

from random import randint
import sys; sys.path.append('lib')

from numpy import array
import karoo_gp.karoo_gp_base_class as karoo_gp

from classlib import ModelHandler

MODEL_EXT = '.ev'
# OPERATORS = [['+', '2']]
OPERATORS = [['+', '2'], ['+ abs', '2']]  # We use abs as our alpha negate op
NEG_OP = ' abs('                       # String denoting abs operator


class GPMask(karoo_gp.Base_GP):
    """ An evolving population of expressions used by the intutitve agent
        to "mask" data between layers two and three. Learns in an online, or
        can be pre-trained.
    """
    def __init__(self, ID, max_pop, max_depth, input_sz, cout, persist, model=False):
        """ ID (str)                : This object's unique ID number
            max_pop (int)           : Max num expression trees (< 10 not ideal)
            max_depth (int)         : Max tree mutate depth
            input_sz (int)          : Max number of inputs to expect
            cout (bool)      : Output log stmts to console flag
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
        self.functions = array(OPERATORS)    # Expression operators/arity
        self.precision = 6                      # Fitness floating points

        # Terminal symbols - one ucase letter for each input, plus label(s)
        self.terminals = [chr(i) for i in range(65, min(91, 65 + input_sz))]
        self.terminals += ['s']
        
        # Update gp mutation ratios based on max pop given
        self.evolve_repro = int(0.2 * max_pop)
        self.evolve_point = int(0.2 * max_pop)
        self.evolve_branch = int(0.1 * max_pop)
        self.evolve_cross = int(0.4 * max_pop)

        # Init the load, save, log, and console output handler if none given
        self.model = model
        if not self.model:
            f_save = "self.save('MODEL_FILE')"
            f_load = "self.load('MODEL_FILE')"
            self.model = ModelHandler(self, cout, persist,
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
                self.population_a[treeID][12][1] = 0.0
    
    def __str__(self):
        str_out = 'ID = ' + self.ID + '\nSize = (\n  '
        str_out += 'max_depth: ' + str(self.tree_depth_max) + '\n  '
        str_out += 'max_pop: ' + str(self.tree_pop_max) + '\n  '
        str_out += 'inputs: ' + str(self.input_sz) + '\n)'
        return str_out

    def _set_mratio(self, max_pop):
        """ Sets the mutation ratios, based on the given max population metric.
        """
        self.evolve_repro = int(0.1 * max_pop)
        self.evolve_point = int(0.1 * max_pop)
        self.evolve_branch = int(0.1 * max_pop)
        self.evolve_cross = int(0.7 * max_pop)

    def save(self, filename=None):
        """ Saves a model of the current population. For use by ModelHandler.
            Iff no filename given, does not save to file but instead returns
            the string that would have otherwise been written.
        """
        # Build model params in dict form
        writestr = "{ 'operands': " + str(self.terminals)
        writestr += ", 'inputs_sz': " + str(self.input_sz)
        writestr += ", 'tree_depth_max': " + str(self.tree_depth_max)
        writestr += ", 'tourn_size': " + str(self.tourn_size)
        writestr += ", 'tree_pop_max': " + str(self.tree_pop_max)
        writestr += ", 'generation_id': " + str(self.generation_id)
        writestr += ", 'pop_tree_type': '" + str(self.pop_tree_type) + "'"
        writestr += ", 'population': \"" + str(self.population_a) + "\""
        writestr += "}"

        if not filename:
            return writestr

        with open(filename, 'w') as f:
            f.write(writestr)
            
    def load(self, filename, not_file=False):
        """ Loads model from file. For use by ModelHandler.
            Iff not_file, does not load from file but instead loads from the
            given string as if it were file contents.
        """
        if not_file:
            loadfrom = filename
        else:
            with open(filename, 'r') as f:
                loadfrom = f.read()

        data = eval(loadfrom.replace('\n', ''))
        self.terminals = data['operands']
        self.input_sz = data['inputs_sz']
        self.tree_depth_max = data['tree_depth_max']
        self.tourn_size = data['tourn_size']
        self.tree_pop_max = data['tree_pop_max']
        self._set_mratio(self.tree_pop_max)
        self.generation_id = data['generation_id']
        self.pop_tree_type = data['pop_tree_type']
        self.population_a = eval(data['population'])

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
        self.fx_eval_poly(self.population_a[treeID])  # Update self.algo_sym
        return self.algo_sym

    def _raw_expr(self, treeID):
        """ Returns the raw expression of the tree with the given ID
        """
        self.fx_eval_poly(self.population_a[treeID])  # Update self.algo_raw
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

    def forward(self, inputs, ordered, max_results=0, gain=.8):
        """ Peforms each tree's expression on the given inputs and returns 
            the results as a dict denoting the source tree ID
            Accepts:
                inputs (list)     : A list of lists, one for each input "row"
                max_results (int) : Max results to return (0=population size)
                gain (float)     : Ratio of fittest expressions used to
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

        # print('Trees: ' + self._expr_strings(symp_expr=ordered))  # debug

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

        # At this point, every t in trees is useable, so do fit/random gain
        # Determine max results
        if not max_results: max_results = self.tree_pop_max
        fit_count = int(max_results * gain)
        gain_trees = trees[:fit_count]
        rand_pool = trees[fit_count:]

        while len(gain_trees) < max_results and rand_pool:
            idx = randint(0, len(rand_pool) - 1)
            gain_trees.append(rand_pool.pop(idx))

        # Iterate every tree that hasn't been filtered out
        results = {}
        for treeID in trees:
            results[treeID] = []
            orig_expr = str(f_expr(treeID))

            # print(str(treeID) + ' - Mask o: ' + str(orig_expr))  # debug

            # Reform the expr by mapping each operand to an input index
            # At the same time, denote where negate operator gets applied
            #   Ex: expr 'A + 2*D + B' -> 'row[0] + 2*row[3] + row[1]'
            new_expr = ''
            negate_at = []
            i_ch = -1
            i_term = -1
            goback = len(NEG_OP)
            for ch in orig_expr:
                i_ch += 1
                if ch in self.terminals and ch != 's':
                    i_term += 1

                    # If term proceeds negate operator, denote and rm
                    if orig_expr[i_ch - goback:i_ch] == NEG_OP:
                        negate_at.append(i_term)
                        new_expr = new_expr[:-goback]
                    
                    # Map term to "row[idx]" and add to new expression str
                    new_expr += 'row[' + str(self.terminals.index(ch)) + ']'

                else:
                    new_expr += ch

            # Clean up open/close parens - we may have left strays doing neg op
            new_expr = new_expr.replace('(', '').replace(')', '')

            # print(str(treeID) + ' - Mask n: ' + str(new_expr))  # debug

            # Eval expr against each input
            for row in inputs:
                output = eval(new_expr)
                
                # Apply negate operators if needed, as previously denoted
                if negate_at:
                    r = [c for c in output]
                    for i in negate_at:
                        r[i] = self.negate(r[i])
                    output = ''.join(r)
                
                # Associate it with its tree
                results[treeID].append(output)

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
                self.population_a[treeID][12][1] = 0.0
                
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

    @staticmethod
    def negate(x):
        """ Returns a negated version of the given string/digit.
        """
        # String negate
        if type(x) is str:
            if x.isupper():
                return x.lower()
            return x.lower()
        
        # Numerical negate
        return (x * -1)


if __name__ == '__main__':
    # Import for pretty-printing of demo data
    from pprint import pprint

    # Example input row
    row = [['A', 'B', 'C', 'D', 'E', 'F']]

    # Init the genetically evolving expression trees
    ev = GPMask('treegp', 20, 10, len(row[0]), cout=True, persist=True)

    sequential = False  # Denote inputs should not be considered sequential
    epochs = 25         # Learning epochs

    # Example learning - 
    # forward() gets results, we set fitness based on them, then call update()
    for z in range(0, epochs):
        print('\n*** Epoch %d ***' % z)

        # Get results using current population
        results = ev.forward(row, ordered=sequential, gain=1)
        print('Results:'); pprint(results)

        # Update fitness of each tree - 
        # Our demo's desired result is a string of all d's w/len <=  4
        fitness = {k: 0.0 for k in results.keys()}
        for k, v in results.items():
            for j in [j for j in v if len(j) <= 4]:
                for c in [c for c in j if c == 'd']:
                    fitness[k] += 1

        # Evolve a new population with the new fitness values
        ev.update(fitness)
