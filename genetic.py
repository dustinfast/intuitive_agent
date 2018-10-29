#!/usr/bin/env python
""" A module for genetically evolving a population of expression trees using 
    and applying them to user-specified data in predefined ways.
    
    Module Structure:
        GPM is the main interface.
        Persistence and output handled by sharedlib.ModelHandler.
        
    Karoo GP Version Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version from https://github.com/kstaats/karoo_gp/pull/9.
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py')

    # TODO: 
       

    Author: Dustin Fast, 2018
"""

import re
from numpy import array
from random import random
import sys; sys.path.append('lib')

import karoo_gp.karoo_gp_base_class as karoo_gp
from sharedlib import ModelHandler, AttributesIter, negate


MODEL_EXT = '.ev'           # File extensions for model file save/load
OP_FLIP_STR = '+ abs'       # Alphabetic case flip operator in string form
KERNEL_MODES = [1, 2, 3]    # See class docstring for mode descriptions

ATTRIB_OUTPUT = 'output'            # Attribute label
ATTRIB_INCONTEXT = 'from_inputs'    # Attribute label


class Genetic(karoo_gp.Base_GP):
    """ An evolving population of expression trees with methods for applying
        them to supplied data and reproducing based on fitness according to
        the given kernel:
        Kernel 1: Numeric minimizing kernel with +, -, *, and / operators
        Kernel 2: String masking kernel with + operator
        Kernel 3: String masking kernel with + and "case flip" operators
    """
    def __init__(self, ID, kernel, max_pop, max_depth, max_inputs, tourn_sz=10,
                 console_out=True, persist=False):
        """ ID (str)                : This object's unique ID number
            kernel (int)            : Operation kernel (see class docstring)
            max_pop (int)           : Max num expression trees (< 11 not ideal)
            max_depth (int)         : Max tree depth
            max_inputs (int)        : Max number of inputs to expect
            tourn_size (int)        : Fitness competition group size
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persist mode flag
        """
        super(Genetic, self).__init__()
        self.ID = ID
        self.persist = persist
        self.tree_pop_max = max_pop
        self.tree_depth_max = max_depth
        self.tourn_size = tourn_sz

        self._set_mratio()                    # Set initial mutation ratios
        self._set_kernel(kernel, max_inputs)  # Setup the specified kernel
        
        # Init the load, save, log, and console output handler if none given
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load('MODEL_FILE')"
        self.model = ModelHandler(self, console_out, persist,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)
    
    def __str__(self):
        str_out = '\nID = ' + self.ID + '\nSize = (\n  '
        str_out += 'max_depth: ' + str(self.tree_depth_max) + '\n  '
        str_out += 'max_pop: ' + str(self.tree_pop_max) + '\n  '
        str_out += 'max_inputs: ' + str(len(self.terminals)) + '\n  '
        str_out += 'evolve_repro : ' + str(self.evolve_repro) + '\n  '
        str_out += 'evolve_point: ' + str(self.evolve_point) + '\n  '
        str_out += 'evolve_branch: ' + str(self.evolve_branch) + '\n  '
        str_out += 'evolver_cross: ' + str(self.evolver_cross) + '\n)'
        return str_out

    def _set_mratio(self, repro=0.15, point=0.15, branch=0.0, cross=0.70):
        """ Sets the mutation ratios, based on the given max population metric.
        """
        self.evolve_repro = int(repro * self.tree_pop_max)
        self.evolve_point = int(point * self.tree_pop_max)
        self.evolve_branch = int(branch * self.tree_pop_max)
        self.evolve_cross = int(cross * self.tree_pop_max)

    def _set_kernel(self, kernel, max_inputs):
        """ Sets up GP operators and methods, depending on the given kernel.
        """
        if kernel not in KERNEL_MODES:
            raise AttributeError('Invalid kernel requested.')

        # Gradient directions, by kernel
        mode = {1: 'min',
                2: 'max',
                3: 'max'}

        # Apply methods, by kernel
        f_apply = {1: self._mode1_apply,
                   2: self._mode2n3_apply,
                   3: self._mode2n3_apply}

        # Operators, by kernel
        opers = {1: [['+', '2'],        
                     ['-', '2'],
                     ['*', '2'],
                     ['/', '2']],
                 2: [['+', '2']],
                 3: [['+', '2'],        
                     ['+ abs', '2']]}

        # Kernel-dependent attributes
        self._mode = kernel
        self._f_apply = f_apply.get(kernel)
        self.fitness_type = mode.get(kernel)
        self.functions = array(opers.get(kernel))

        # Kenrnel 1 gets a set of random floats from [0, 1)
        if kernel == 1:
            self.terminals = [random() for i in range(max_inputs)]

        # Kernels 2 & 3 work with string inputs, so terminals must be
        # non-numeric (sympy will simplify those) and non sympy function name
        # (sympy will try to call those) placeholders. Ex: 'bzzz', 'gh', etc.
        elif kernel == 2 or kernel == 3:
            self.terminals = []
            trailing_chars = 1
            curr_char = 0
            for _ in range(1, max_inputs + 1):
                curr_char += 1
                if curr_char >= 26:
                    trailing_chars += 1
                    curr_char = 1
                ch = curr_char + 96
                self.terminals.append(chr(ch) + chr(ch + 1) * trailing_chars)

        #  Kernel independent attributes...
        self.terminals += ['s']  # 's' label unused required by KarooGP
        self.display = 's'                    # Silence Karoo GP menus/output
        self.precision = 6                    # Tourney floating points

        # Init first generation (if not already loaded)
        try:
            self.population_a
            self.pop_tree_type = 'g'
        except AttributeError:
            self.generation_id = 1
            self.population_a = ['Generation 1']
            self.fx_karoo_construct('r', 6)  # r = most diverse initial pop
            for treeID in range(1, len(self.population_a)):
                self.population_a[treeID][12][1] = 0.0

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
        trees = self._trees_IDs()
        trees = sorted(
            trees, key=lambda x: self.population_a[x][12][1], reverse=rev)

        if not trees:
            raise AttributeError('Population has no trees.')

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
        lst = re.split('[()]', string)[:-1]
        return lst[1:]

    def save(self, filename=None):
        """ Saves a model of the current population. For use by ModelHandler.
            Iff no filename given, does not save to file but instead returns
            the string that would have otherwise been written.
        """
        # Build model params in dict form
        writestr = "{ 'operands': " + str(self.terminals)
        writestr += ", 'tree_depth_max': " + str(self.tree_depth_max)
        writestr += ", 'tourn_size': " + str(self.tourn_size)
        writestr += ", 'tree_pop_max': " + str(self.tree_pop_max)
        writestr += ", 'generation_id': " + str(self.generation_id)
        writestr += ", 'pop_tree_type': '" + str(self.pop_tree_type) + "'"
        writestr += ", 'population': \"" + str(self.population_a) + "\""
        writestr += ", 'evolve_repro': \"" + str(self.evolve_repro) + "\""
        writestr += ", 'evolve_point': \"" + str(self.evolve_point) + "\""
        writestr += ", 'evolve_branch': \"" + str(self.evolve_branch) + "\""
        writestr += ", 'evolve_cross': \"" + str(self.evolve_cross) + "\""
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
        self.tree_depth_max = data['tree_depth_max']
        self.tourn_size = data['tourn_size']
        self.tree_pop_max = data['tree_pop_max']
        self.generation_id = data['generation_id']
        self.pop_tree_type = data['pop_tree_type']
        self.population_a = eval(data['population'])
        self._set_mratio(data['evolve_repro'],
                         data['evolve_point'],
                         data['evolve_branch'],
                         data['evolve_cross'])

    def apply(self, **kwargs):
        """ Applies the kernel to the given inputs (if any) and returns the
            results as a list.
        """
        return self._f_apply(**kwargs)

    def _mode1_apply(self, normalize=False, gain=.75):
        """ Returns a list of max_inputs (as passed to constructor) numbers
            generated by the max_inputs number of fittest trees.
        """
        results = []
        trees = self._tree_IDs()

        for i in range(len(self.terminals) - 1):
            treeID = trees[i]
            result = self._symp_expr(treeID) * self.terminals[i]
            results.append(result)

        if normalize:
            norm_max = max(results)
            norm_min = min(results)
            results = [(r - norm_min) / (norm_max - norm_min) for r in results]

        return results

    def _mode2n3_apply(self, inputs, is_seq=False):
        """ Applies each tree's expression to the given inputs.
            Accepts:
                inputs (list)     : A list of lists, one for each input "row"
                is_seq (bool)     : Denotes row (in input) order must persist 
            Returns:
                A dictionary, by tree ID, of lists representing the inputs
                after masking, as well as the unmasked input indexes, as:
                    { treeID: { output: [ ... ], from_inps: [ ... ], ... } 
        """
        # Denote use of order-preserving raw expression, or simplified
        f_getexpr = self._raw_expr
        if is_seq:
            f_getexpr = self._symp_expr

        trees = self._tree_IDs()

        # Iterate every tree that has made the cut
        outputs = AttributesIter()
        for treeID in trees:
            # Get current tree's expression. Ex: expr 'A + D'
            expression = self._expr_to_lst(f_getexpr(treeID))

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
                outputs.push(treeID, ATTRIB_OUTPUT, output)
                outputs.push(treeID, ATTRIB_INCONTEXT, list(in_context))

        # Remove empty results and return
        outputs.rm_empties(ATTRIB_OUTPUT)
        return outputs

    def update(self, fitness):
        """ Evolves a new population of trees after updating tree fitness. 
        Accepts:
            fitness (dict)    : { treeID: fitness }
        """
        # Rest each trees baseline fitness
        for treeID in range(1, len(self.population_a)):
                self.population_a[treeID][12][1] = 0.0

        # Update tree fitness as given by "fitness"
        for k, v in fitness.items():
            self.population_a[k][12][1] = v

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

    def _mode1_update(self, expr):
        pass

    def _mode2_update(self, expr):
        pass


if __name__ == '__main__':
    """ TODO: Demo info... 
    """
    from pprint import pprint  # For pretty-printing demo output

    # Define inputs
    inputs = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']]
    is_seq = False   # Denote inputs should not be considered sequential

    # Init the genetically evolving expression trees (mode 2)
    lengs = [len(inputs[i]) for i in range(len(inputs))]
    gp = Genetic(ID='treegp', 
                 kernel=2,
                 max_pop=15, 
                 max_depth=4, 
                 max_inputs=max(lengs),
                 tourn_sz=min(lengs),
                 console_out=True, 
                 persist=False)

    # Get results, eval fitness, and backprogate fitness
    iterations = 30
    for z in range(1, iterations + 1):
        print('\n*** Epoch %d ***' % z)

        results = gp.apply(inputs=inputs)
        fitness = fitness = {k: 0.0 for k in results.keys()}  # init
        
        print('Results:')
        for trees in results:
            for treeID, attrs in trees.items():
                output = attrs[ATTRIB_OUTPUT]
                pprint('Tree %d: %s' % (treeID, output))

                # Evaluate fitness
                if len(output) > 2:
                    continue
                if output[:1] == 'A':
                    fitness[treeID] += 1
                if output[1:2] == 'K':
                    fitness[treeID] += 1
                if output == 'AK':
                    fitness[treeID] += 10

        # Evolve a new population with the new fitness values
        gp.update(fitness)
