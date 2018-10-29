#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for description of the agent and the application as a whole.
        
    If CONSOLE_OUT = True:
        Agent and its sub-modules output to stdout

    If PERSIST = True:
        Agent and its sub-module states persist between executions via file
        PERSIST_PATH/ID.MODEL_EXT (saved each time the agent thread stops).
        Output is also logged to PERSIST_PATH/ID.LOG_EXT. 

    Module Structure:
        Agent() is the main interface. It expects training/validation data as
        an instance obj of type sharedlib.DataFrom(). 
        Persistence and output is handled by sharedlib.ModelHandler().

    Dependencies:
        PyTorch
        Requests
        Scikit-learn
        Matplotlib
        TensorFlow
        Sympy
        Numpy
        Scipy
        Pandas

    Usage: 
        Run from the terminal with './agent.py'.

    # TODO: 
        Check private notation for each class member
        L2.node_map[].weight (logarithmic decay over time/frequency)
        L2.node_map[].kb/correct/solution strings
        REPL vs. Flask interface?
        in_data must contain all labels, otherwise ANN inits break 
        Accuracy: Print on stop, Check against kb
        GP tunables - mutation ratios, pop sizes, etc
        Adapt ann.py to accept dataloader and use MNIST (or similiar)
        Refactor save/loads into ModelHandler.get_savestring?
        "provides feedback" connector - ex: True if the action returns a value
        Add other common heuristics to layer 2

    Author: Dustin Fast, 2018
"""

import copy
import logging
import threading
from pprint import pprint

from ann import ANN
from genetic import Genetic
from connector import Connector
from sharedlib import ModelHandler, DataFrom, WeightedValues

CONSOLE_OUT = False
PERSIST = False
MODEL_EXT = '.agnt'

L2_EXT = '.lyr2'
L2_KERNEL = 2       # Kernel for the L2 mask
L2_MAX_DEPTH = 4    # 10 is max, per Karoo user man. Has perf affect.
L2_MAX_POP = 12     # Number of expressions to generate. Has perf affect.
L2_TOURNYSZ = int(L2_MAX_POP / 2)

L3_EXT = '.lyr3'
L3_ADVISOR = Connector.is_python_kwd


class ConceptualLayer(object):
    """ An abstraction of the agent's conceptual layer (i.e. layer one), which
        represents its "sensory input". 
        Provides interfaces to each layer-node and a its current output.
        Nodes at this level are ANN's representing a single sensory 
        input processing channel, where the input to each channel is a sample
        of some subset of the agent's environment. Its output is then its
        "classification" of what that input represents.
        On init, each node is loaded by the ANN object from file iff PERSIST.
        Note: This layer must be trained offline via self.train(). After 
        training, each node saves its model to file iff PERSIST.
        """
    def __init__(self, ID, depth, dims, inputs):
        """ Accepts:
                ID (str)        : This layers unique ID
                depth (int)     : How many nodes this layer contains
                dims (list)     : 3 ints - ANN in/hidden/output layer sizes
        """
        self._nodes = []        # A list of nodes, one for each layer 1 depth
        self._depth = depth     # This layer's depth. I.e., it's node count

        for i in range(depth):
            nodeID = ID + '_node_' + str(i)
            self._nodes.append(ANN(nodeID, dims[i], CONSOLE_OUT, PERSIST))
            self._nodes[i].set_labels(inputs[i].class_labels)
    
    def train(self, train_data, val_data, epochs=500, lr=.01, alpha=.9):
        """ Trains each node from the given training/validation data.
            Accepts:
                train_data (list)       : A list of DataFrom objects
                val_data (list)         : A list of DataFrom objects
                epochs (int)            : Number of training iterations
                lr (float)              : Learning rate
                alpha (float)           : Learning gain/momentum
        """
        for i in range(self._depth):
            self._nodes[i].train(
                train_data[i], epochs=epochs, lr=lr, alpha=alpha, noise=None)
            self._nodes[i].validate(val_data[i], verbose=True)

    def forward(self, inputs):
        """ Moves the given inputs through the layer, setting self.outputs
            appropriately.
            Accepts:
                inputs (list)   : A list of tensors, one per node.
            Returns:
                A list of outputs with one element (a list) for each node.
            
        """
        outputs = []
        for i in range(self._depth):
            outputs.append(self._nodes[i].classify(inputs[i]))
        return outputs

    
class IntuitiveLayer(object):
    """ An abstraction of the agents second layer, representing the ability to
        learn which information to devote attention to. This layer is 
        implemented as sets of heuristics for each node's unique input applied 
        to genetically evolving weights. In this way, we're attempt to optimize
        heuristic usage. The heuristics are ultimitely used to determine which
        inputs get passed through the layer and which are discarded.
    """
    def __init__(self, ID, depth):
        """ Accepts:
                ID (str)        : This layers unique ID
                depth (int)     : Num nodes this layer contains
        """
        self.ID = ID            # This layer's unique ID
        self._depth = depth     # This layer's depth. I.e., its node count
        self._nodes = []        # A list of nodes, one at each depth
        self._optimizers = []   # Heuristics optimizer lists, one list per node
        self._t_heur = None     # A template of fresh heuristics
        self._prev_inputs = []  # Previous input values, one for each node
        self._id_prefix = ID + '_node_'  # Optimizer ID prefixes

        # Init heuristics template
        self._t_heur = WeightedValues()
        self._t_heur.set('count')    # Curr input encountered count
        self._t_heur.set('pos')      # Input resulted in "fit" output count
        self._t_heur.set('neg')      # Input resulted in "unfit" output count
        self._t_heur.set('diff')     # Diff btwn prev and curr input
       
        # Init each node and its helpers
        for i in range(depth):
            self._nodes.append({})       # {'UNIQUEINPUT': WeightedValuesObj }
            self._prev_inputs.append(None)

        # Init heuristic optimizers - one per heuristic
        for h in self._t_heur.keys():
            optID = self._id_prefix + '_' + h
            self._optimizers.append(
                Genetic(ID=optID,
                        kernel=1,
                        max_pop=L2_MAX_POP,
                        max_depth=L2_MAX_DEPTH,
                        max_inputs=L2_MAX_POP,
                        tourn_sz=L2_TOURNYSZ,
                        console_out=CONSOLE_OUT,
                        persist=PERSIST))

        # Init the model handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L2_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nNodes = ' + str(len(self._nodes))
        return str_out

    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write each node ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')
            for node in self._nodes:
                key = node.optimizer.ID
                savestr = node.save()
                f.write('"' + key + '": """' + savestr + '""", ')
            f.write('}')

    def _load(self, filename):
        """ Loads the layer from file. For use by ModelHandler.
        """
        # Restore the layer nodes, one at a time
        self._nodes = {}
        i = 0

        with open(filename, 'r') as f:
            data = f.read()

        for k, v in eval(data).items():
            ID = self._id_prefix + str(i)
            node = Genetic(ID, 0, 0, 0, CONSOLE_OUT, False)
            node.load(v, not_file=True)
            self._nodes[k] = (node, None)
            i += 1

    def forward(self, inputs, is_seq=False):
        """ Moves the given inputs through the layer and returns the output.
            Accepts: 
                inputs (list)       : Data elements, one for each node
                is_seq (bool)       : Denotes inputs order is significant
            Returns:
                dict: { TreeID: {'output': [], 'in_context':[]}, ... }
        """
        outputs = {}

        # Get heuristic weight "tries" - i.e. all results from each optimizer
        # [ [ heuristic1 tries ... ], [ heuristic tries...], ... ]
        weights = []
        for opt in self._optimizers:
            w = opt.apply()
            weights.append(w)
        weights = list(list(zip(*weights)))  # Transpose

        # Update heuristics for each depth's current unique input
        for i in range(self._depth):
            node = self._nodes[i]
            node_input = inputs[i]
            
            # Get heuristics for the node's current unique input
            try:
                heurs = node[node_input]
            except KeyError:
                # None found - init w/fresh heuristics
                node[node_input] = copy.deepcopy(self._t_heur)
                heurs = node[node_input]

            # Increment count heuristic
            heurs.adjust('count', 1)
            
            # Set diff heuristic
            if node_input == self._prev_inputs[i] or not self._prev_inputs[i]:
                diff = 0  # no last input, or same as last input
            else:
                # If char input
                if type(node_input) is str and len(node_input) == 1:
                    diff = ord(self._prev_inputs[i]) - ord(node_input)
                # If numeric input
                elif isinstance(node_input, (int, float, complex)):
                    diff = self._prev_inputs[i] - node_input
                # Else, can't determine diff for curr input type
                else:
                    diff = 0
            heurs.set('diff', diff)

            # Append node's input to results if heurs dictates to do so
            for j, w in enumerate(weights):
                heurs.set_wts(list(w))
                wtd_heurs = heurs.get_list(normalize=False)
                # print(str(j+1) + ': ' + str(list(w)))
                # print(str(j+1) + ': ' + str(wtd_heurs))

                # If at least one heur >= 1, add node's input to tree's results
                # if [k for k in wtd_heurs if k >= 1]:
                if sum(wtd_heurs) >= 1:
                    treeID = j + 1
                    if not outputs.get(treeID):
                        outputs[treeID] = {'output': [], 'in_context': []}
                    outputs[treeID]['output'].append(node_input)
                    outputs[treeID]['in_context'].append(i)

            # debug
            # print('Input: ' + node_input)
            # print('WtdHeurs: ' + str(wtd_heurs))
            # print('NodeWt: ' + str(node_wt))

        pprint(outputs)
        print('\n')
        self._prev_inputs = inputs  # Denote curr inputs for next time
        return outputs

    def update(self, results):
        """ Updates each node according to the given fitness data.
            Accepts:
                fitness (dict) : { treeID: {'fitness': x, 'in_context':[]} }
        """
        fitness = {k: 0.0 for k in results.keys()}
        
        for i in range(self._depth):
            node = self._nodes[i]
            node_input = self._prev_inputs[i]
            heurs = node[node_input]
            
            # Examine fitness scores for each output this node contributed to
            for treeID, attrs in results.items():
                if i in attrs['in_context']:
                    score = attrs['fitness']

                    # Update fitness for backpropagation
                    fitness[treeID] += score

                    # Update heuristics
                    if score:
                        heurs.adjust('pos', 1)
                    else:
                        heurs.adjust('neg', 1)

        # Backpropagate fitness scores
        pprint(fitness)
        for optimizer in self._optimizers:
            optimizer.update(fitness)


class LogicalLayer(object):
    """ An abstraction of the agent's Logical layer (i.e. layer three), which
        evaluates the fitness of it's input according to the given kernel.
        This layer does no persistence or logging at this time.
    """
    def __init__(self, ID, kernel):
        """ Accepts:
                ID (str)            : This layers unique ID
                kernel (function)   : Any func returning True or false when
                                      given some layer-two output
        """
        self.ID = ID
        self._kernel = kernel
        self.kb = []  # debug
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L3_EXT,
                                  save_func='MODEL_FILE',  # i.e., unused
                                  load_func='MODEL_FILE')  # i.e., unused

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nMode = ' + self._kernel.__name__
        return str_out

    def forward(self, results):
        """ Checks each result in results and appends a fitness score (as
            determined by self._kernel) to the results dict.
            Accepts:
                results (dict) : { treeID: {'output': [], 'in_context':[]} }
            Returns:
                The results parameter with a 'fitness' key added.
        """

        for treeID, attrs in results.items():
            score = 0.0
            # for i in attrs['output']:
            #     if i == 'u':
            #         score += 1
            if len(attrs['output']) <= 3:
                score = 1
            if len(attrs['output']) <= 2:
                score = 3

            results[treeID]['fitness'] = score
        return results

        # for k, v in results.items():
        #     for j in v['output']:
        #         self.model.log('L3 TRYING: ' + j)
        #         if self._kernel(j):
        #             fitness[k] += 1
        #             self.model.log('TRUE!')

        #             # debug output
        #             if j not in self.kb:
        #                 self.kb.append(j)
        #                 print('L3 Learned: ' + j)
        #             else:
        #                 print('L3 Encountered: ' + j)


class Agent(threading.Thread):
    """ The intutive agent.
        The constructor accepts the agent's "sensory input" data, from which
        the layer dimensions are derived. After init, start the agent from 
        the terminal with 'agent start', which runs the agent as a seperate
        thread (running does not cause the agent to block, so the user may
        stop it from the command line, etc.
        Persistence: On each iteration of the input data, the agent is saved 
        to a file.
    """
    def __init__(self, ID, inputs, is_seq):
        """ Accepts the following parameters:
            ID (str)            : The agent's unique ID
            inputs (list)       : Agent input data, one for each L1 node
            is_seq (bool)       : Denotes input data is sequential in nature,
                                  i.e., the layer 2 mask will use only ordered
                                  expressions, such as 'A + C + E', as opposed
                                  to something like 'C + E + A'
        """
        threading.Thread.__init__(self)
        self.ID = ID
        self.depth = None           # Layer 1 depth/node count
        self.model = None           # The model handler
        self.running = False        # Agent thread running flag (set on start)
        self.inputs = inputs        # The agent's "sensory input" data
        self.is_seq = is_seq        # Denote inputs is sequential in nature
        self.max_iters = None       # Num inputs iters, set on self.start
        self.L2_nodemap = None      # Pop via ModelHandler, for loading L2

        # Init the load, save, log, and console output handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        # Determine agent shape from given input data TODO: Move to l1 class?
        l1_dims = []
        self.depth = len(inputs)

        for i in range(self.depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x sz
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y sz
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg

        # Init layers
        id_prefix = self.ID + '_'
        ID = id_prefix + 'L1'
        self.l1 = ConceptualLayer(ID, self.depth, l1_dims, inputs)
        ID = id_prefix + 'L2'
        self.l2 = IntuitiveLayer(ID, self.depth)
        ID = id_prefix + 'L3'
        self.l3 = LogicalLayer(ID, L3_ADVISOR)

    def __str__(self):
        return 'ID = ' + self.ID

    def _save(self, filename):
        """ Saves a model of the agent. For use by ModelHandler.
            Also causes saves to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        self.l2.model.save()

    def _load(self, filename):
        """ Loads the agent model from file. For use by ModelHandler.
            Also causes loads to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        self.l2.model.load()

    def _step(self, data_row):
        """ Steps the agent forward one step with the given data row: A list
            of tuples (one for each depth) of inputs and targets. Each tuple,
            by depth, is fed to layer one, who's ouput is fed to layer 2, etc.
            Note: At this time, the 'targets' in the data row are not used.
            Accepts:
                data_row (list)     : [inputs, ... ]
        """
        # Ensure well formed data_row
        if len(data_row) != self.depth:
            err_str = 'Bad data_row size - expected sz ' + str(self.depth)
            err_str += ', recieved size' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Step Layer 1 -------------------------------
        # L1.outputs[i] becomes L1.nodes[i]'s classification of data_row[i]
        # ------------------------------------------------------------------
        l1_row = [d[0] for d in data_row]  # TODO: Fix [0] in run()

        for i in range(self.depth):
            self.model.log('-- Feeding L1[%d]:\n%s' % (i, str(l1_row[i])))

        l1_outputs = self.l1.forward(l1_row)
        
        # --------------------- Step Layer 2 -------------------------------
        # L2.outputs becomes the "masked" versions of all L1.outputs
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L2:\n %s' % (l1_outputs))
        l2_outputs = self.l2.forward(l1_outputs)
        
        # --------------------- Step Layer 3 -------------------------------
        # L3 evals fitness of it's input from L2 and backpropagates signals
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L3:\n%s' % str(l2_outputs))
        l3_outputs = self.l3.forward(l2_outputs)

        self.model.log('-- L2 Backprop:\n%s' % str(l3_outputs))
        self.l2.update(l3_outputs)
        # TODO: Send feedback/noise/"in context" to level 1

    def start(self, max_iters=10):
        """ Starts the agent thread.
            Accepts:
                max_iters (int)     : Max times to iterate data set (0=inf)
        """
        self.max_iters = max_iters
        threading.Thread.start(self)

    def run(self):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
        """
        self.model.log('Agent thread started.')
        min_rows = min([data.row_count for data in self.inputs])
        self.running = True
        iters = 0

        # Step the agent forward with each row of each dataset
        while self.running:
            for i in range(min_rows):
                row = []
                for j in range(self.depth):
                    row.append([row for row in iter(self.inputs[j][i])])
        
                self.model.log('\n** STEP - iter: %d depth:%d **' % (iters, i))
                self._step(row)
                
            if self.max_iters and iters >= self.max_iters - 1:
                self.stop('Agent stopped: max_iters reached.')
            iters += 1

    def stop(self, output_str='Agent stopped.'):
        """ Stops the thread. May be called from the REPL, for example.
            Also logs the given output string (if any) and saves the agent to
            file (iff PERSIST).
        """
        self.running = False
        self.model.log(output_str)

        if PERSIST:
            self.model.save()

    @staticmethod
    def _shape_fromdata(in_data):
        """ Determines agent's shape from the given list of input data sets.
            Assumes each layer 1 node has 3 layers (x, h, and y).
            Accepts:
                in_data (list)     : A list of input data, as DataFrom objects
            Returns:
                2-tuple: L1 depth and L1 dims, as (int, [int, int, int]) 
        """
        depth = len(in_data)
        l1_dims = []
        l2_dims = []

        for i in range(depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x node count
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y node count
            l1_dims[i][1] = int(     
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg
            l2_dims.append(in_data[i].class_count)          # y node counts
            
        return depth, l1_dims, l2_dims


if __name__ == '__main__':
    # Agent "sensory input" data. Length denotes the agent's L1 and L2 depth.
    in_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]
    # in_data = [DataFrom('static/datasets/letters0.csv', normalize=True),
    #            DataFrom('static/datasets/letters1.csv', normalize=True),
    #            DataFrom('static/datasets/letters2.csv', normalize=True),
    #            DataFrom('static/datasets/letters3.csv', normalize=True)]
    #            DataFrom('static/datasets/letters4.csv', normalize=True),
    #            DataFrom('static/datasets/letters5.csv', normalize=True),
    #            DataFrom('static/datasets/letters6.csv', normalize=True),
    #            DataFrom('static/datasets/letters7.csv', normalize=True),
    #            DataFrom('static/datasets/letters8.csv', normalize=True),
    #            DataFrom('static/datasets/letters9.csv', normalize=True)]

    # Layer 1 training data (one per node) - length must match len(in_data) 
    l1_train = [DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 1 validation data (one per node) - length must match len(in_data)
    l1_vald = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # Instantiate the agent (agent shape is derived automatically from in_data)
    agent = Agent('agent1', in_data, is_seq=False)

    # Train and validate the agent
    # agent.l1.train(l1_train, l1_vald)

    # Start the agent thread in_data as input data
    agent.start(max_iters=20)
