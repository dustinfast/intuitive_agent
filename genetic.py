#!/usr/bin/env python
""" A module for genetically evolving a population of expression trees using 
    the Karoo GP library.

    Module Structure:
        GPM is the main interface.
        Persistence and output handled by classlib.ModelHandler.
        
    Karoo GP Version Info: 
        Karoo GP was written by Kai Staats for Python 2.7. We use the adapted 
        Python 3 version from https://github.com/kstaats/karoo_gp/pull/9.
        Small, non-systemic changes to karoo_gp.Base_GP were made by Dustin
        Fast for use in this module (see notes in 'lib/karoo_gp_base_class.py')

    # TODO: 
        forward() perf improvements
        input_sz currently limited to <= 26
        Modes docstring

    Author: Dustin Fast, 2018
"""

from random import randint
import sys; sys.path.append('lib')

from numpy import array
import karoo_gp.karoo_gp_base_class as karoo_gp

from classlib import ModelHandler

MODEL_EXT = '.ev'       # File extensions for model file save/load
OP_ABS_STR = ' abs('    # Negate operator in string form

ATTRIB_OUTPUT = 'output'
ATTRIB_INCONTEXT = 'from_inputs'

class Genetic(karoo_gp.Base_GP):
    """ An evolving population of expression trees able to operate in -
        Mode 1: Produces numeric expressions with +, -, *, and / operators
        Mode 2: Evolves string "mask" expressions without case mutation
        Mode 3: Evolves string "mask" expressions with case mutation
    """
    def __init__(self, ID, mode, max_pop, max_depth, max_inputs, tourn_sz,
                 console_out=True, persist=False):
        """ ID (str)                : This object's unique ID number
            mode (int)              : Operation mode (see class docstring)
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

        self.display = 's'      # Silence Karoo GP menus/output
        self.precision = 6      # Tourney floating points
        self._set_mratio()      # Set initial mutation ratios
        self._set_mode(mode)    # Setup operators and methods

        # Init terminals - one ucase letter for each input, plus unused "s"
        self.terminals = [chr(i) for i in range(65, min(91, 65 + max_inputs))]
        self.terminals += ['s']

        # Init the load, save, log, and console output handler if none given
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load('MODEL_FILE')"
        self.model = ModelHandler(self, console_out, persist,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        # Init first generation if not already loaded by model handler
        try:
            self.population_a
            self.pop_tree_type = 'g'
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
        str_out += 'evolver_cross: ' + str(self.evolver_cross) + '\n)'
        return str_out

    def _set_mratio(self, repro=0.15, point=0.15, branch=0.0, cross=0.70):
        """ Sets the mutation ratios, based on the given max population metric.
        """
        self.evolve_repro = int(repro * self.tree_pop_max)
        self.evolve_point = int(point * self.tree_pop_max)
        self.evolve_branch = int(branch * self.tree_pop_max)
        self.evolve_cross = int(cross * self.tree_pop_max)

    def _set_mode(self, mode):
        """ Sets up GP operators and methods, depending on the given mode.
        """
        # Operators, by mode
        opers = {1: [['+', '2'],        
                     ['-', '2'],
                     ['*', '2'],
                     ['/', '2']],
                 2: [['+', '2']],
                 3: [['+', '2'],        
                     ['+ abs', '2']]}

        # Kernels, by mode
        kernel = {1: 'max',
                  2: 'max',
                  3: 'min'}

        # Update and Forward methods
        f_forward = {1: mode1_forward,
                     2: mode2_forward,
                     3: mode2_forward}

        f_update = {1: mode1_udpate,
                    2: mode2_udpate,
                    3: mode2_udpate}
        self.f_forward = f_forward.get(mode)
        self.f_update = f_update.get(mode)

        self.functions = array(opers.get(mode))
        self.fitness_type = kernel.get(mode)

    def forward(self, **kwargs):
        """ Performs a forward, depending on the current mode.
            Accepts the following optional key/value args (inputs required): 
            inputs (list)     : A list of lists, one for each input "row"
            max_results (int) : Max results to return (0=population size)
            gain (float)      : Ratio of fittest to randomly chosen expressions
            ordered (bool)    : Denotes row-input order must persist 
        """
        return self.f_forward(self, **kwargs)

    def update(self, fitness_results):
        """ Performs an update, depending on the current mode.
            Accepts a single argument of type FitResults
        """
        return self.f_update(self, fitness_results)

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

        exprs = ''
        for treeID in range(1, len(self.population_a)):
            expr = str(f_expr(treeID))
            exprs += 'Tree ' + str(treeID) + ': ' + expr + '\n'

            if w_fit:
                fit = str(self.population_a[treeID][12][1])
                exprs += ' (fitness: ' + fit + ')\n'

        return exprs

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


#################################################
# Forward/update methods (per mode) and helpers #
#################################################

class TreeResults(object):
    """ A parrallel arrangment of FIFO attribute queues, indexed by treed ID.
    """
    def __init__(self):
        self._results = {}  # {TREE_ID: {attr_1: [val_x], attr_n: [val_x]}, .. }

    def enqueue(self, treeID, attr, value):
        """ Enqueue the given results attribute, by tree ID.
        """
        try:
            tree = self._results[treeID]
        except KeyError:
            self._results[treeID] = {}
            tree = self._results[treeID]

        try:
            val_list = tree[attr]
        except KeyError:
            tree[attr] = []
            val_list = tree[attr]

        val_list.append(value)

    def filter_empty(self, attr):
        """ Removes all trees having no attributes of the given name.
        """
        self._results = {k: v for k, v in self._results.items() if v[attr]}

    def is_empty(self, attr):
        """ Returns True if attributes of the given name exists. Else False.
        """
        if {k: v for k, v in self._results.items() if v[attr]}:
            return False
        return True

    def dequeue(self):
        """ Dequeues and returns a single set of attributes for each tree as:
            { TREEID_1: {attr_1: val, attr_n: val}, ... }
        """
        attributes = {}

        for treeID, attrs in self._results.items():
            attributes[treeID] = {}
            for k, v in attrs.items():
                try:
                    attributes[treeID][k] = v.pop(0)
                except IndexError:
                    attributes[treeID][k] = None
        return attributes


class Fitness(object):
    """ Container for returning fitness data to the update methods.
    """
    def __init__(self):
        pass


def mode1_forward(obj, inputs, ordered=False, max_results=0, gain=.8):
    pass


def mode1_udpate(obj, fitness):
    pass


def mode2_forward(obj, inputs, ordered, max_results=0, gain=.8):
    """ Peforms each tree's expression on the given inputs and returns 
        the results as a dict denoting the source tree ID
        Accepts:
            inputs (list)     : A list of lists, one for each input "row"
            max_results (int) : Max results to return (0=population size)
            gain (float)      : Ratio of fittest to randomly chosen expressions
            ordered (bool)    : Denotes row-input order must persist 
        Returns:
            A dictionary of lists representing the masked inputs, by tree:
            { treeID: { masked: [ ... ], in_context: [ ... ], ... } 
    """
    try:
        trees = obj._trees_byfitness()  # leftmost = most fit
    except AttributeError:
        raise Exception('Forward attempted on an uninitialized model.')

    # If ordered specified, use raw expression, else use sympified
    if ordered:
        f_expr = obj._symp_expr
    else:
        f_expr = obj._raw_expr

    # print('Trees: ' + obj._expr_strings(symp_expr=ordered))  # debug

    # If strings in inputs, rm exprs w/neg operators - they're nonsensical
    for inp in inputs:
        if not [i for i in inp if type(i) is str]: 
            break
    else:
        trees = [t for t in trees if '-' not in str(f_expr(t))]

    # Filter trees having duplicate expressions
    added = set()
    trees = [t for t in trees if str(f_expr(t)) not in added and
             (added.add(str(f_expr(t))) or True)]

    # Every tree is useable at this pt, so do apply gain if specified
    if max_results: 
        fit_count = int(max_results * gain)
        gain_trees = trees[:fit_count]
        rand_pool = trees[fit_count:]

        while len(gain_trees) < max_results and rand_pool:
            idx = randint(0, len(rand_pool) - 1)
            gain_trees.append(rand_pool.pop(idx))
        
        trees = gain_trees

    # Iterate every tree that has made the cut
    results = TreeResults()
    for treeID in trees:
        orig_expr = str(f_expr(treeID))

        # Reform the expr by mapping each operand to an input index
        # At the same time, denote where negate operator gets applied,
        # and also set "in_context", to denote which inputs are in result.
        #   Ex: expr 'A + 2*D + B'-> 'row[0] + 2*row[3] + row[1]'
        #   Ex: expr 'abs(A) + C'-> 'row[0] + row[3]' w/negate_at = [0]
        new_expr = ''
        negate_at = []
        in_context = set()
        i_ch = -1
        i_term = -1
        goback = len(OP_ABS_STR)
        for ch in orig_expr:
            i_ch += 1
            if ch in obj.terminals and ch != 's':
                i_term += 1

                # If term proceeds negate operator, denote and rm operator
                if orig_expr[i_ch - goback:i_ch] == OP_ABS_STR:
                    negate_at.append(i_term)
                    new_expr = new_expr[:-goback]
                
                # Map term to idx, udpate in_context, & rowify for new_expr
                input_idx = obj.terminals.index(ch)
                in_context.add(input_idx)
                new_expr += 'row[' + str(input_idx) + ']'
            else:
                new_expr += ch

        # Clean up open/close parens - we may have left strays doing neg op
        new_expr = new_expr.replace('(', '').replace(')', '')

        # print(str(treeID) + ' - Mask n: ' + str(new_expr))  # debug

        # Eval expr against each input in inputs
        for row in inputs:
            output = eval(new_expr)
            
            # Apply negate operators if needed, as previously denoted
            if negate_at:
                r = [c for c in output]
                for i in negate_at:
                    r[i] = obj._negate(r[i])
                output = ''.join(r)
            
            # Associate output and its "in context" inputs with the tree's ID
            results.enqueue(treeID, ATTRIB_OUTPUT, output)
            results.enqueue(treeID, ATTRIB_INCONTEXT, in_context)
            # results[treeID]['masked'].append(output)
        
        # Add the inputs used in this tree's results, to the results
        # results[treeID]['in_context'] = in_context

        results.filter_empty(ATTRIB_OUTPUT)

    return results

def mode2_udpate(obj, fitness):
    """ Evolves a new population of trees after updating the fitness of 
        each existing tree's expression according to "fitness". 
        Accepts:
            fitness (dict)    : Each key (int) is a tree ID and each value
                                denotes it's new fitness value
    """
    # Give each tree a baseline fitness score
    for treeID in range(1, len(obj.population_a)):
            obj.population_a[treeID][12][1] = 0.0
            
    # Update tree's fitness as given by "fitness" arg
    for k, v in fitness.items():
        obj.population_a[k][12][1] = v

    # Build the new gene pool
    obj.gene_pool = [t for t in range(1, len(obj.population_a))
                     if obj._symp_expr(t)]

    # Evolve a new population
    obj.population_b = []
    obj.fx_karoo_reproduce()
    obj.fx_karoo_point_mutate()
    obj.fx_karoo_branch_mutate()
    obj.fx_karoo_crossover()
    obj.generation_id += 1
    obj.population_a = obj.fx_evolve_pop_copy(
        obj.population_b, 'Generation ' + str(obj.generation_id))

    if obj.persist:
        obj.model.save()


def _negate(x):
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
    row = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']]

    # Init the genetically evolving expression trees (mode 2)
    lengs = [len(row[i]) for i in range(len(row))]
    gp = Genetic(ID='treegp', 
                 mode=2,
                 max_pop=15, 
                 max_depth=4, 
                 max_inputs=max(lengs),
                 tourn_sz=min(lengs),
                 console_out=True, 
                 persist=False)

    sequential = False   # Denote inputs should not be considered sequential
    epochs = 75          # Learning epochs

    # Get results with forward(), eval fitness, then update(fitness)
    for z in range(0, epochs):
        print('\n*** Epoch %d ***' % z)

        # Get mask results
        results = gp.forward(inputs=row, ordered=sequential, gain=1)

        while not results.is_empty(ATTRIB_OUTPUT):
            r = results.dequeue()
            pprint(r)
        exit()

        # For this demo, we're only interested in each trees masked output
        results = {k: v['masked'] for k, v in results._results.items()}
        print('Results:'); pprint(results)

        # Update fitness of each tree based on this demo's desired result
        fitness = {k: 0.0 for k in results.keys()}
        for k, v in results.items():
            v = v[0]
            if len(v) > 2:
                continue
            if v[:1] == 'A':
                fitness[k] += 1
            if v[1:2] == 'K':
                fitness[k] += 1

        # Evolve a new population with the new fitness values
        gp.update(fitness)
