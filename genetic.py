#!/usr/bin/env python
""" A module for genetically evolving a population of expression trees using 
    the Karoo GP library.

    Module Structure:
        GPM is the main interface.
        Persistence and output handled by sharedlib.ModelHandler.
        
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

from sharedlib import ModelHandler, AttributesIter, negate

MODEL_EXT = '.ev'       # File extensions for model file save/load
OP_ABS_STR = ' abs('    # Negate operator in string form

ATTRIB_OUTPUT = 'output'
ATTRIB_INCONTEXT = 'from_inps'
ATTRIB_FITNESS = 'fit'

class Genetic(karoo_gp.Base_GP):
    """ An evolving population of expression trees with methods for applying
        them to supplied data and reproducing based on fitness according to
        the given kernel:
        Kernel 1: Minimizing function with +, -, *, and / operators
        Kernel 2: Maximinzing function without + operator
        Kernel 3: Maximinzing function without + and negate operators
    """
    def __init__(self, ID, kernel, max_pop, max_depth, max_inputs, tourn_sz,
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

        self.display = 's'          # Silence Karoo GP menus/output
        self.precision = 6          # Tourney floating points
        self._set_mratio()          # Set initial mutation ratios
        self._set_kernel(kernel)    # Setup the specified kernel

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

    def _set_kernel(self, kernel):
        """ Sets up GP operators and methods, depending on the given kernel.
        """
        # Operators, by kernel
        opers = {1: [['+', '2'],        
                     ['-', '2'],
                     ['*', '2'],
                     ['/', '2']],
                 2: [['+', '2']],
                 3: [['+', '2'],        
                     ['+ abs', '2']]}

        # Goal type, by kernel
        goals = {1: 'max',
                 2: 'max',
                 3: 'min'}

        self.functions = array(opers.get(kernel))
        self.fitness_type = goals.get(kernel)

    def _trees_byfitness(self):
        """ Returns a list of the current population's tree ID's, sorted by
            fitness (L to R).
            If no trees, raises Attribute Error
        """
        rev = {'min': False, 'max': True}.get(self.fitness_type)
        trees = [t for t in range(1, len(self.population_a))]
        trees = sorted(
            trees, key=lambda x: self.population_a[x][12][1], reverse=rev)

        if not trees:
            raise AttributeError('Population has no trees.')

        return trees

    def _filtered_trees(self, get_expr, exclude_ops=[]):
        """ Returns a list of the current population's expression, arranged by 
            fitness (leftmost = most fit) after filtering trees with duplicate
            expressions and unwanted operators.
        """
        # Get trees sorted by fitness
        trees = self._trees_byfitness()

        # Remove duplicates
        added = set()
        trees = [t for t in trees if str(get_expr(t)) not in added and
                 (added.add(str(get_expr(t))) or True)]

        # Remove unwanted oeprators
        for op in exclude_ops:
            trees = [t for t in trees if op not in str(get_expr(t))]

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

    # def forward_nomask(self, heuristics):
    #     """
    #     """
    #     trees = self._trees_byfitness()  # leftmost = most fit
    #     return trees[

    def _apply_gain(self, trees, max_results, gain):
        """ Given a list of trees, returns the list containing only max_results
            of fittest/random trees, as specified by the gain.
            Accepts:
                trees (list)      : Tree IDs (ints)
                max_results (int) : Max results to return, 0=all (I.e. no gain)
                gain (float)      : Ratio of fittest to randomly chosen
        """
        if max_results:
            fit_count = int(max_results * gain)
            gain_trees = trees[:fit_count]
            rand_pool = trees[fit_count:]

            while len(gain_trees) < max_results and rand_pool:
                idx = randint(0, len(rand_pool) - 1)
                gain_trees.append(rand_pool.pop(idx))

            trees = gain_trees
        
        return trees

    def forward(self, inputs, is_seq=False, max_results=0, gain=.8, 
                expr_only=False):
        """ Peforms each tree's expression on the given inputs and returns 
            the results as a dict denoting the source tree ID
            Accepts:
                inputs (list)     : A list of lists, one for each input "row"
                is_seq (bool)     : Denotes row (in input) order must persist 
                max_results (int) : Max results to return (0=all)
                gain (float)      : Ratio of fittest to randomly chosen exprs
                expr_only (bool)  : Denotes to return only the expressions
            Returns:
                If expr_only=True, a list of expressions (len=max_results)
                Else, a dictionary, by tree ID, of lists representing the 
                inputs after they are applied to the expression and the inputs
                that contribute to them:
                    { treeID: { output: [ ... ], from_inps: [ ... ], ... } 
        """
        # Denote use of order-preserving raw expression, or simplified
        if is_seq:
            f_getexpr = self._symp_expr
        else:
            f_getexpr = self._raw_expr

        # If inputs contain any strings, denote neg operator as nonsensical
        bad_ops = []
        for inp in inputs:
            if [i for i in inp if type(i) is str]:
                bad_ops.append('-')
                break

        trees = self._filtered_trees(f_getexpr, exclude_ops=bad_ops)

        # print('Trees: ' + self._expr_strings(symp_expr=is_seq))  # debug
        
        # At this point every tree is usable - apply gain and cap results
        trees = self._apply_gain(trees, max_results, gain)

        # Iterate every tree that has made the cut
        outputs = AttributesIter()
        for treeID in trees:
            orig_expr = str(f_getexpr(treeID))

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
                if ch in self.terminals and ch != 's':
                    i_term += 1

                    # If term proceeds negate operator, denote and rm operator
                    if orig_expr[i_ch - goback:i_ch] == OP_ABS_STR:
                        negate_at.append(i_term)
                        new_expr = new_expr[:-goback]

                    # Map term to idx, udpate in_context, & rowify for new_expr
                    input_idx = self.terminals.index(ch)
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
                        r[i] = self._negate(r[i])
                    output = ''.join(r)

                # Associate output and its "in context" inputs with the tree's ID
                outputs.push(treeID, ATTRIB_OUTPUT, output)
                outputs.push(treeID, ATTRIB_INCONTEXT, in_context)

        # Remove empty results and return
        outputs.rm_empties(ATTRIB_OUTPUT)
        return outputs

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
    # Import for pretty-printing of demo data
    from pprint import pprint

    inputs = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']]
    is_seq = False   # Denote inputs should not be considered sequential
    epochs = 30          # Learning epochs

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

    # Get results, eval fitness, then backprogate fitness
    for z in range(1, epochs + 1):
        print('\n*** Epoch %d ***' % z)

        results = gp.forward(inputs=inputs, gain=1)
        fitness = fitness = {k: 0.0 for k in results.keys()}  # init
        
        print('Results:')
        for trees in results:
            # pprint(trees)
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

        # Evolve a new population with the new fitness values
        gp.update(fitness)
