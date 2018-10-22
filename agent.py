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

from ann import ANN
from genetic import Genetic
from connector import Connector
from sharedlib import ModelHandler, DataFrom, WeightedValues

CONSOLE_OUT = True
PERSIST = False
MODEL_EXT = '.agnt'

L2_EXT = '.lyr2'
L2_KERNEL = 2       # Kernel for the L2 mask
L2_MAX_DEPTH = 4    # 10 is max, per Karoo user man. Has perf affect.
L2_MAX_POP = 25     # Number of expressions to generate. Has perf affect.
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
    def __init__(self, ID, id_prefix, depth, dims, inputs):
        """ Accepts:
            id_prefix (str)     : Each nodes ID prefix. Ex: 'Agent1_'
            depth (int)         : How many nodes this layer contains
            dims (list)         : 3 ints - ANN in/hidden/output layer sizes
        """
        self.nodes = []      # A list of nodes, one for each layer 1 depth
        self.outputs = []    # A list of outputs, one for each node
        self.depth = depth  # This layer's depth. I.e., it's node count

        for i in range(depth):
            nodeID = ID + '_node_' + str(i)
            self.nodes.append(ANN(nodeID, dims[i], CONSOLE_OUT, PERSIST))
            self.outputs.append([None for i in range(depth)])
            self.nodes[i].set_labels(inputs[i].class_labels)
    
    def train(self, train_data, val_data, epochs=500, lr=.01, alpha=.9):
        """ Trains each node from the given training/validation data.
            Accepts:
                train_data (list)       : A list of DataFrom objects
                val_data (list)         : A list of DataFrom objects
                epochs (int)            : Number of training iterations
                lr (float)              : Learning rate
                alpha (float)           : Learning gain/momentum
        """
        for i in range(self.depth):
            self.nodes[i].train(
                train_data[i], epochs=epochs, lr=lr, alpha=alpha, noise=None)
            self.nodes[i].validate(val_data[i], verbose=True)

    
class AttentiveLayer(object):
    """ An abstraction of the agents second layer, representing the ability to
        learn which information to devote attention to. This layer is 
        implemented as sets of heuristics for each node's unique input applied 
        to genetically evolving weights. In this way, we're attempt to optimize
        heuristic usage. The heuristics are ultimitely used to "mask" the layer's
        inputs - the results are the layer's output.
    """
    def __init__(self, ID, id_prefix, depth, dims):
        """ Accepts:
            id_prefix (str)     : Each node's ID prefix. Ex: 'Agent1_'
            depth (int)         : How many nodes this layer contains
            dims (list)         : A list of each node's input length
        """
        self.ID = ID            # This layer's unique ID
        self.depth = depth      # This layer's depth. I.e., its node count
        self.nodes = []         # A list of nodes, one at each depth
        self.outputs = []       # A list of outputs, one for each node
        self.prev_inputs = []   # Previous input values, one for each node
        self._optimizers = []   # Evolving heuristics optimizers, one per node
        self._outmask = None    # Evolving output mask
        self._t_heur = None     # A template of fresh heuristics
        self._id_prefix = id_prefix + ID + '_node_'

        # Init heuristics template
        self.t_heur = WeightedValues()
        self.t_heur.set('count')    # Curr input encountered count
        self.t_heur.set('pos')      # Input resulted in "fit" output count
        self.t_heur.set('neg')      # Input resulted in "unfit" output count
        self.t_heur.set('mag')      # Ascii code if str, num if num, else 0
        self.t_heur.set('notprev')  # 1 if input doesn't match prev, else -1
       
        # Init each node and helpers
        for i in range(depth):
            self.nodes.append({})  # {'UNIQUEINPUT': WeightedValuesObj }
            self.outputs.append(None)
            self.prev_inputs.append(None)

            # Init node optimizer (persist=False - the layer will handle it)
            self._optimizers.append(Genetic(ID=self._id_prefix + str(i),
                                            kernel=1,
                                            max_pop=L2_MAX_POP,
                                            max_depth=L2_MAX_DEPTH,
                                            max_inputs=len(self.t_heur),
                                            tourn_sz=L2_TOURNYSZ,
                                            console_out=CONSOLE_OUT,
                                            persist=False))
                 
        # Init output mask (persist=False because this layer handles it)
        self._outmask = Genetic(ID=ID + '_outmask',
                                kernel=L2_KERNEL,
                                max_pop=L2_MAX_POP,
                                max_depth=L2_MAX_DEPTH,
                                max_inputs=sum(dims),
                                tourn_sz=L2_TOURNYSZ,
                                console_out=CONSOLE_OUT,
                                persist=False)

        # Init the model handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L2_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nNodes = ' + str(len(self.nodes))
        return str_out

    def forward(self, inputs, is_seq=False):
        """ Moves the given inputs through the layer & returns the output.
            Accepts: 
                inputs (list)       : Data elements, one for each node
                is_seq (bool)       : Denotes inputs order is significant
        """
        # Iterate the input for each node
        for i in range(self.depth):
            inp = inputs[i]
            
            # Get heuristics for the current input
            try:
                heurs = self.nodes[i][inp]
            except KeyError:
                # None found - init w/fresh heuristics
                self.nodes[i][inp] = copy.copy(self.t_heur)
                heurs = self.nodes[i][inp]

            # Update heuristic weights, according to node's optimizer
            weights = self._optimizers[i].apply(normalize=True)
            heurs.set_wts(weights)

            # Increment count heuristic
            heurs.adjust('count', 1)
            
            # Set magnitude heuristic
            if type(inp) is str and len(inp) == 1:
                heurs.set('mag', (ord(inp) - 64) / 100)
            elif isinstance(inp, (int, float, complex)):
                heurs.set('mag', inp)
            else:
                heurs.set('mag', 0)
            
            # Set not prev heuristic
            if inp != self.prev_inputs[i]:
                heurs.set('notprev', 1)
            else:
                heurs.set('notprev', -1)

            # The sum of all weighted heuristics is the grand node weight
            print(heurs)  # debug
            wtd_heurs = heurs.get_list(normalize=False)
            node_wt = sum(wtd_heurs)

            # debug
            node_wt = 0
            # node_wt += heurs.get('count')
            # node_wt += heurs.get('pos')
            # node_wt += heurs.get('neg')
            node_wt += heurs.get('mag')
            node_wt += heurs.get('notprev')
            print(wtd_heurs)
            print(node_wt)

            exit()

        self.prev_inputs = inputs  # Denote curr inputs for next time
        return inputs

    def update(self, fitness_data):
        """ Update active node with the given fitness data dict.
        """
        # fitness = {k: 0.0 for k in results.keys()}
        # for k, v in results.items():
        #     for j in v['masked']:
        #         self.model.log('L3 TRYING: ' + j)
        #         if self.kernel(j):
        #             fitness[k] += 1
        #             self.model.log('TRUE!')

        #             # debug output
        #             if j not in self.kb:
        #                 self.kb.append(j)
        #                 print('L3 Learned: ' + j)
        #             else:
        #                 print('L3 Encountered: ' + j)
        pass
        
    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write each node ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')

            # Write each node optimizer
            for node in self.nodes:
                key = node.optimizer.ID
                savestr = node.save()
                f.write('"' + key + '": """' + savestr + '""", ')
            
            # Write the outmasker
            f.write('"outmasker": """' + self._outmask.save() + '""", ')

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


class LogicalLayer(object):
    """ An abstraction of the agent's Logical layer (i.e. layer three), which
        evaluates the fitness of it's input according to the given kernel.
        This layer does no persistence or logging at this time.
        Note: This is the most computationally expensive layer
    """
    def __init__(self, ID, kernel):
        """ Accepts:
                kernel (function)  : Any func returning True or false when
                                     given some layer-two output
        """
        self.ID = ID
        self.kernel = kernel
        self.node = self.check_fitness
        self.kb = []  # debug
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L3_EXT,
                                  save_func='MODEL_FILE',  # i.e., unused
                                  load_func='MODEL_FILE')  # i.e., unused

    def check_fitness(self, results):
        """ Checks each result in results and returns a dict of fitness scores
            corresponding to each, as determined by self.kernel.
            Accepts:
                results (dict)  : { treeID:  { masked:    [ ... ], 
                                               in_context: [ ... ], ... }
            Returns a tuple containing:
                dict            : Fitness as { treeID: fitness score (float) }
        """
        # Extract each tree's masked output
        # results = {k: v['masked'] for k, v in results.items()}

        # Update fitness of each tree based on this demo's desired result
        fitness = {k: 0.0 for k in results.keys()}
        for k, v in results.items():
            for j in v['masked']:
                self.model.log('L3 TRYING: ' + j)
                if self.kernel(j):
                    fitness[k] += 1
                    self.model.log('TRUE!')

                    # debug output
                    if j not in self.kb:
                        self.kb.append(j)
                        print('L3 Learned: ' + j)
                    else:
                        print('L3 Encountered: ' + j)

        return fitness


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

        # Determine agent shape from the given input data
        l1_dims = []
        l2_dims = []
        self.depth = len(inputs)

        for i in range(self.depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x node count
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y node count
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg
            l2_dims.append(in_data[i].class_count)          # y node counts

        # Init layers
        id_prefix = self.ID + '_'  # For prefixing sub elements w/agent ID
        ID = id_prefix + 'L1'
        self.l1 = ConceptualLayer(ID, id_prefix, self.depth, l1_dims, inputs)
        ID = id_prefix + 'L2'
        self.l2 = AttentiveLayer(ID, id_prefix, self.depth, l2_dims)
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
            err_str += ', recieved sz' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Step Layer 1 -------------------------------
        # L1.outputs[i] becomes L1.nodes[i]'s classification of self.inputs[i]
        # ------------------------------------------------------------------
        for i in range(self.depth):
            inputs = data_row[i][0]
            self.model.log('-- Feeding L1 node[%d]:\n%s' % (i, str(inputs)))
            self.l1.outputs[i] = self.l1.nodes[i].classify(data_row[i][0])
        # --------------------- Step Layer 2 -------------------------------
        # L2.outputs becomes the "masked" versions of all L1.outputs
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L2:\n %s' % (self.l1.outputs))
        self.l2.outputs = self.l2.forward(self.l1.outputs)
        
        # --------------------- Step Layer 3 -------------------------------
        # L3 evals fitness of it's input from L2 and backpropagates signals
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L3:\n%s' % str(self.l2.outputs))
        fitness = self.l3.check_fitness(self.l2.outputs)

        self.model.log('-- L2 Backprop:\n%s' % str(fitness))
        self.l2.update(fitness)
        # TODO: Send feedback /noise/"in context" to level 1

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
    #            DataFrom('static/datasets/letters3.csv', normalize=True),
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
    agent.start(max_iters=5)
