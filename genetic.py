#!/usr/bin/env python
""" A module for genetically evolving a population of expression trees 
    and applying them to user-specified data.

    Module Structure:
        GPM is the main interface.
        Persistence and output handled by sharedlib.ModelHandler.
    
    Usage: 
        See main() for example usage.
        
    Karoo GP Version Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version from https://github.com/kstaats/karoo_gp/pull/9.
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py')

    # TODO: 
        Trim mutation for paring down large trees when acc is low
        Remove is_seq and dep sympy
        L2 feedback only if in context

    Author: Dustin Fast, 2018
"""

import re
import random
import numpy as np
from numpy import array
import sys; sys.path.append('lib')

import karoo_gp.karoo_gp_base_class as karoo_gp
from sharedlib import ModelHandler, AttrIter, Queue, negate

# User configurable
MODEL_EXT = '.ev'       # File extensions for model file save/load
DEFAULT_MREPRO = 0.15   # Default genetic mutation ration: Reproduction
DEFAULT_MPOINT = 0.15   # Default genetic mutation ration: Point
DEFAULT_MBRANCH = 0.0   # Default genetic mutation ration: Branch
DEFAULT_MCROSS = 0.7    # Default genetic mutation ration: Crossover

# Not user configurable
OP_FLIP_STR = '+ abs'       # Alphabetic case flip operator in string form
KERNEL_MODES = [1, 2]       # See class docstring for kernel descriptions
KERNEL_OPERATORS = {1: [['+', '2']],
                    2: [['+', '2'],
                        ['+ abs', '2']]}

class Genetic(karoo_gp.Base_GP):
    """ An evolving population of genetically evolving expression trees with 
        methods for applying them as a "mask" to the supplied data according to
        the given kernel:
        Kernel 1: String masking kernel with + operator
        Kernel 3: String masking kernel with + and "case flip" operators
    """
    def __init__(self, ID, kernel, max_pop, max_depth, max_inputs, mem_depth=1,
                 tourn_sz=10, console_out=True, persist=False):
        """ ID (str)                : This object's unique ID number
            kernel (int)            : Operation kernel (see class docstring)
            max_pop (int)           : Max num expression trees (< 11 not ideal)
            max_depth (int)         : Max tree depth
            max_inputs (int)        : Max number of inputs to expect
            tourn_size (int)        : Fitness competition group size
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persist mode flag
        """
        if kernel not in KERNEL_MODES:
            raise AttributeError('Invalid kernel specified.')
            
        super(Genetic, self).__init__()
        self.ID = ID
        self.persist = persist
        self.tree_pop_max = max_pop
        self.tree_depth_max = max_depth
        self.tourn_size = tourn_sz
        self.display = 's'          # Silence Karoo GP menus/output
        self.precision = 6          # Tourney fitness floating points
        self.fitness_type = 'max'   # Maximizing kernel function
        self.functions = np.array(KERNEL_OPERATORS.get(kernel))
        self.set_mratio()          # Set initial mutation ratios

        # Apply mem depth - 
        #  Mem depth 1 is our "input" size, where mem depth 2 is input fed as
        #  feedback rom the last input to mem depth 1. Mem depth 3 is then mem
        #  depth 2's prev feedback. Prev_deepmem implemented as a shove queue
        self.mem_width = max_inputs * mem_depth
        self.mem = Queue(self.mem_width)
        
        # Init terminal symbols (i.e. genetic expression leaf-nodes) - 
        #  We work with string inputs, so our terminals must be non-numeric 
        #  (or sympy will simplify them) and not sympy func names (or sympy will
        #  try to call them). Acceptable terminals ex: 'bzzz', 'gh', etc.
        self.terminals = []
        trailing_chars = 1
        curr_char = 0
        for _ in range(1, self.mem_width + 1):
            curr_char += 1
            if curr_char >= 26:
                trailing_chars += 1
                curr_char = 1
            ch = curr_char + 96
            self.terminals.append(chr(ch) + chr(ch + 1) * trailing_chars)
        self.terminals += ['s']  # 's' label unused but required by KarooGP

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
        except AttributeError:
            self.generation_id = 1
            self.population_a = ['Generation 1']
            self.fx_karoo_construct('r', 6)  # r = most diverse initial pop
            for treeID in range(1, len(self.population_a)):
                self.population_a[treeID][12][1] = 0.0
    
    def __str__(self):
        str_out = '\nID = ' + self.ID + '\nSize = (\n  '
        str_out += 'max_depth: ' + str(self.tree_depth_max) + '\n  '
        str_out += 'max_pop: ' + str(self.tree_pop_max) + '\n  '
        str_out += 'max_inputs: ' + str(len(self.terminals)) + '\n  '
        str_out += 'evolve_repro : ' + str(self.evolve_repro) + '\n  '
        str_out += 'evolve_point: ' + str(self.evolve_point) + '\n  '
        str_out += 'evolve_branch: ' + str(self.evolve_branch) + '\n  '
        str_out += 'evolver_cross: ' + str(self.evolve_cross) + '\n)'
        return str_out

    def _save(self, filename=None):
        """ Saves a model of the current population. For use by ModelHandler.
            Iff no filename given, does not save to file but instead returns
            the string that would have otherwise been written.
        """
        # Build model params in dict form
        writestr = "{ 'operands': " + str(self.terminals)
        writestr += ", 'tree_depth_max': " + str(self.tree_depth_max)
        writestr += ", 'tourn_size': " + str(self.tourn_size)
        writestr += ", 'tree_pop_max': " + str(self.tree_pop_max)
        writestr += ", 'mem_width': " + str(self.mem_width)
        writestr += ", 'generation_id': " + str(self.generation_id)
        writestr += ", 'pop_tree_type': '" + str(self.pop_tree_type) + "'"
        writestr += ", 'evolve_repro': \"" + str(self.evolve_repro) + "\""
        writestr += ", 'evolve_point': \"" + str(self.evolve_point) + "\""
        writestr += ", 'evolve_branch': \"" + str(self.evolve_branch) + "\""
        writestr += ", 'evolve_cross': \"" + str(self.evolve_cross) + "\""
        writestr += ", 'population': \"" + str(self.population_a) + "\""
        writestr += "}"

        if not filename:
            return writestr

        with open(filename, 'w') as f:
            f.write(writestr)

    def _load(self, filename, not_file=False):
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
        self.tree_depth_max = data['tree_depth_max']
        self.tourn_size = data['tourn_size']
        self.tree_pop_max = data['tree_pop_max']
        self.tourn_size = data['tourn_size']
        self.generation_id = data['generation_id']
        self.pop_tree_type = data['pop_tree_type']
        self.population_a = eval(data['population'])
        self.set_mratio(int(data['evolve_repro']),
                        int(data['evolve_point']),
                        int(data['evolve_branch']),
                        int(data['evolve_cross']))

        self.mem_width = data['mem_width']
        self.mem = Queue(self.mem_width)

    def _tree_IDs(self):
        """ Returns a list of the populations tree IDs.
            If no trees, raises Attribute Error
        """
        trees = [t for t in range(1, len(self.population_a))]

        if not trees:
            raise AttributeError('Population has no trees.')

        return trees

    def _trees_byfitness(self):
        """ Returns a list of the current population's tree ID's, sorted by
            fitness (L to R).
            If no trees, raises Attribute Error
        """
        rev = {'min': False, 'max': True}.get(self.fitness_type)
        trees = self._tree_IDs()
        if not trees:
            raise AttributeError('Population has no trees.')

        ftrees = sorted(
            trees, key=lambda x: self.population_a[x][12][1], reverse=rev)

        # If list is unchanged, assume trees have same fitness & randomize
        if ftrees == trees:
            random.shuffle(ftrees)

        return ftrees

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
            f_getexpr = self._symp_expr
        else:
            f_getexpr = self._raw_expr

        exprs = ''
        for treeID in range(1, len(self.population_a)):
            expr = str(f_getexpr(treeID))
            exprs += 'Tree ' + str(treeID) + ': ' + expr + '\n'

            if w_fit:
                fit = str(self.population_a[treeID][12][1])
                exprs += ' (fitness: ' + fit + ')\n'

        return exprs

    def _expr_to_lst(self, string):
        """ Given an expression in string form, returns it as a list.
        """
        lst = re.split('[()]', string)[:-1]  # raw_expr
        if not lst:
            # If regex failed to split, assume sympy exr
            string = string.replace('*', ' * ')  # So * op splits correctly
            return re.split(' ', string)
        return lst[1:]

    def set_mratio(self, repro=DEFAULT_MREPRO, point=DEFAULT_MPOINT,
                   branch=DEFAULT_MBRANCH, cross=DEFAULT_MCROSS):
        """ Sets the genetic mutation ratios.
        """
        # If not already initialized, assume a ratio, else assume integer
        try:
            self.evolve_repro  # throws AttributeError if not exists
            self.evolve_repro = repro
            self.evolve_point = point
            self.evolve_branch = branch
            self.evolve_cross = cross
        except AttributeError:
            self.evolve_repro = int(repro * self.tree_pop_max)
            self.evolve_point = int(point * self.tree_pop_max)
            self.evolve_branch = int(branch * self.tree_pop_max)
            self.evolve_cross = int(cross * self.tree_pop_max)

    def clear_mem(self):
        """ Clears any symbols from the working memory.
        """
        self.mem.reset()

    def apply(self, inputs, is_seq=False):
        """ Applies each tree's expression to the given inputs.
            Accepts:
                inputs (list)     : A list of lists, one for each input "row"
                is_seq (bool)     : Denotes input row-order must persist 
            Returns:
                A dictionary, by tree ID, of lists representing the inputs
                after masking, as well as the unmasked input indexes, as:
                    { treeID: { output: [ ... ], from_inps: [ ... ], ... } 
        """
        outputs = AttrIter()

        # Denote use of either order-preserving sympy expression or raw expr
        f_getexpr = self._raw_expr
        if is_seq:
            f_getexpr = self._symp_expr

        # Restructure inputs into current inputs plus prev mem depths
        new_inputs = []

        for row in inputs:
            # print('row: ' + str(row))  # debug
            # print('prev: ' + str(self.mem.get_items()))  # debug

            # new_row becomes inputs(current) + inputs(all but oldest)
            self.mem.items = [m for m in self.mem.get_items() if m != '']
            for r in row:
                self.mem.shove(r)
            new_row = self.mem.get_items()
            # print('new row: ' + str(new_row))  # debug
            # print('new prev: ' + str(self.mem.get_items()))  # debug

            # Pad up to mem width w/ emptry string elements
            for i in range(len(new_row) + 1, self.mem_width + 1):
                new_row.append('')
                self.mem.shove('')
            # print('post_pad row: ' + str(new_row))  # debug
            # print('post pad prev: ' + str(self.mem.get_items()))  # debug

            new_inputs.append(new_row)
        inputs = new_inputs
        # print(inputs)  # debug

        # Do not evaluate anything until mem is full
        if [s for s in inputs if '' in s]:
            return outputs

        for treeID in self._tree_IDs():
            # Get current tree's expression. Ex: expr 'A + D'
            expression = self._expr_to_lst(str(f_getexpr(treeID)))

            # Build expr str by mapping inputs to terms. Ex: 'row[0]+row[3]'
            expr_str = ''
            negate_at = []
            in_context = set()
            idx_term = -1
            for el in expression:  
                if el in self.terminals:
                    input_idx = self.terminals.index(el)
                    in_context.add(input_idx)
                    expr_str += 'row[' + str(input_idx) + ']'
                    idx_term += 1
                else:
                    if el != OP_FLIP_STR:
                        # Append non-negate operators as-is
                        expr_str += el
                    else:
                        # For negate operators, denote and append '+' op
                        negate_at.append(idx_term + 1)
                        expr_str += '+'

            # Eval each row of input against the expression
            for row in inputs:
                try:
                    output = eval(expr_str)
                except ZeroDivisionError:
                    continue

                # Apply negate operator at indexes previously denoted
                if negate_at:
                    r = [c for c in output]
                    for i in negate_at:
                        r[i] = negate(r[i])
                    output = ''.join(r)

                # Results = output and contributing terminals, by treeID
                outputs.push(treeID, 'output', output)
                outputs.push(treeID, 'from_inputs', list(in_context))

        return outputs

    def _new_genepool(self, max_results, gain):
        """ Rreturns a list of "max_results" trees, a mix of fittest and random
            trees, as specified by the gain parameter.
            Accepts:
                max_results (int) : Max results to return, 0=all (I.e. no gain)
                gain (float)      : Ratio of fittest to randomly chosen
        """
        trees = self._trees_byfitness()
        trees = [t for t in trees if self._symp_expr(t)]  # exclude nulls

        if max_results:
            fit_count = int(max_results * gain)
            gain_trees = trees[:fit_count]
            rand_pool = trees[fit_count:]

            while len(gain_trees) < max_results and rand_pool:
                idx = random.randint(0, len(rand_pool) - 1)
                gain_trees.append(rand_pool.pop(idx))

            # TODO: skew according to gain if unbalanced.

            trees = gain_trees

        return trees

    def update(self, fitness, gain=.75):
        """ Evolves a new population of trees after updating tree fitness. 
        Accepts:
            fitness (dict)    : { treeID: fitness }
            gain (float)      : Variance gain
        """

        def randtree(trees):
            """ Generator for returning a random tree ID from a list of trees.
            """
            ubound = len(trees) - 1
            while True:
                rand_index = random.randint(0, ubound)
                yield trees[rand_index]

        # Reset each tree's baseline fitness
        for treeID in range(1, len(self.population_a)):
            self.population_a[treeID][12][1] = 0.0

        # Update tree fitnesses given by fitness arg, then get new genepool
        for k, v in fitness.items():
            self.population_a[k][12][1] = v
        self.population_a[2][12][1] = 1.0

        # Setup new population and gene pool
        self.population_b = []
        gene_pool = self._new_genepool(self.tourn_size, gain)
        rtree = randtree(gene_pool)

        # Perform reproductions/mutations according to current ratios
        for _ in range(self.evolve_repro):
            tree = np.copy(self.population_a[next(rtree)])
            self.population_b.append(tree)  # Reproduction is a straight copy

        for _ in range(self.evolve_point):
            tree = np.copy(self.population_a[next(rtree)])
            tree, _ = self.fx_evolve_point_mutate(tree)
            self.population_b.append(tree)

        for _ in range(self.evolve_branch):
            tree = np.copy(self.population_a[next(rtree)])
            branch = self.fx_evolve_branch_select(tree)
            tree = self.fx_evolve_grow_mutate(tree, branch)
            self.population_b.append(tree)

        for _ in range(self.evolve_cross // 2):
            tree_a = np.copy(self.population_a[next(rtree)])
            tree_b = np.copy(self.population_a[next(rtree)])
            brnch_a = self.fx_evolve_branch_select(tree_a)
            brnch_b = self.fx_evolve_branch_select(tree_b)
            tree_c = np.copy(tree_a)
            brnch_c = np.copy(brnch_a)
            tree_d = np.copy(tree_b)
            brnch_d = np.copy(brnch_b)
            child1 = self.fx_evolve_crossover(tree_a, brnch_a, tree_b, brnch_b)
            self.population_b.append(child1)
            child2 = self.fx_evolve_crossover(tree_d, brnch_d, tree_c, brnch_c)
            self.population_b.append(child2) 

        # New generation takes place of the current
        self.generation_id += 1
        self.population_a = self.fx_evolve_pop_copy(
            self.population_b, 'Generation ' + str(self.generation_id))

        if self.persist:
            self.model.save()


if __name__ == '__main__':
    """ The following is a demonstration of the genetic algorithm with a
        utility function compelling it to converge on the string "AK" from  its
        input string of the characters A-K.
    """
    from pprint import pprint  # For pretty-printing demo output

    # Define inputs (may contain more than one inner list. Ex: 2nd line down)
    inputs = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']]
    # inputs = [['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '10']]
    input_length = max([len(inputs[i]) for i in range(len(inputs))])

    # Init the genetically evolving expression trees
    gp = Genetic(ID='gp_demo',
                 kernel=1,
                 max_pop=40, 
                 max_depth=6, 
                 max_inputs=input_length,
                 mem_depth=1,
                 console_out=True, 
                 persist=False)

    # Get results, eval fitness, and backprogate fitness
    iterations = 49
    for z in range(1, iterations + 1):
        print('\n*** Epoch %d ***' % z)

        results = gp.apply(inputs=inputs)
        fitness = {k: 0.0 for k in results.keys()}  # init
        
        print('Results:')
        for trees in results:
            for treeID, attrs in trees.items():
                output = attrs['output']
                pprint('Tree %d: %s' % (treeID, output))

                # Evaluate fitness
                length = len(output)
                if output[:1] == 'A':
                    fitness[treeID] += 70
                    if output[1:2] == 'K':
                        fitness[treeID] += 9
                    while fitness[treeID] > 1 and length:
                            length -= 1
                            fitness[treeID] -= 1

        # Evolve a new population with the new fitness values
        gp.update(fitness, gain=10)  # gain=1 because we want specific solution
