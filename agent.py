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
        L2 does not output node count in multimode
        Seperate log output option and persist

    Author: Dustin Fast, 2018
"""

import logging
import threading
from pprint import pprint

from ann import ANN
from genetic import Genetic
from connector import Connector
from sharedlib import ModelHandler, DataFrom

CONSOLE_OUT = True
PERSIST = True
MODEL_EXT = '.agnt'

L2_EXT = '.lyr2'
L2_MAX_DEPTH = 2    # 10 is max, per Karoo user manual. Has perf affect.
L2_MAX_POP = 15     # Number of expressions to generate. Has perf affect.
L2_TOURNYSZ = int(L2_MAX_POP * .25)  # Random fitness tourney selection pool sz
L2_TOURNYSZ = 10  # Random fitness tourney selection pool sz
L2_MEMDEPTH = 2     # Agent's "recurrent" memory, a multiple of L1's input sz

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
    """ An abstraction of the agent's second layer, representing the ability to
        intuitively form new connections between symbols, as well as its
        recurrent memory (i.e. it's last L2_MEMDEPTH
    """
    def __init__(self, ID, size):
        """ Accepts:
                ID (str)        : This layers unique ID
                size (int)      : Num nodes this layer contains
        """
        self.ID = ID            # This layer's unique ID
        self._size = size       # This layer's size. I.e., it's input count
        self._node = None       # The GP element
        self._prev_inputs = []  # Previous input values, one for each node
        self._nodeID = ID + '_node'  # GP node's unique ID

        # Init the layer's node - a genetically evolving tree of expressions
        self._node = Genetic(ID=self._nodeID,
                             kernel=2,
                             max_pop=L2_MAX_POP,
                             max_depth=L2_MAX_DEPTH,
                             max_inputs=3,  # debug
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
        str_out += '\nSize = ' + str(self._size)
        return str_out

    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write the node's ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')
            savestr = self._node.save()
            f.write('"' + self._nodeID + '": """' + savestr + '""", ')
            f.write('}')

    def _load(self, filename):
        """ Loads the layer from file. For use by ModelHandler.
            Note: The node ID in the file is ignored
        """
        with open(filename, 'r') as f:
            data = f.read()

        loadme = next(iter(eval(data).values()))
        ID = self._nodeID
        node = Genetic(ID, 2, 0, 0, 0, 0, CONSOLE_OUT, False)
        node.load(loadme, not_file=True)
        self._node = node

    def forward(self, inputs, is_seq=False):
        """ Moves the given inputs through the layer and returns the output.
            Accepts: 
                inputs (list)       : Data elements, one for each node
                is_seq (bool)       : Denotes inputs order is significant
            Returns:
                dict: { TreeID: {'output': [], 'in_context':[]}, ... }
        """
        self._prev_inputs = inputs  # Note curr inputs for next time
        return self._node.apply(inputs=list([inputs]), is_seq=is_seq)

    def update(self, fitness):
        """ Updates the layer's node according to the given fitness data.
            Accepts:
                fitness (dict) : { treeID: {'fitness': x, 'in_context':[]} }
        """
        self._node.update(fitness)


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
        self.kb = []  # TODO: Expand kb/move to L2?
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L3_EXT,
                                  save_func='MODEL_FILE',  # i.e. unused
                                  load_func='MODEL_FILE')  # i.e. unused

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nMode = ' + self._kernel.__name__
        return str_out

    def forward(self, results):
        """ Checks each result in results and appends a fitness score as
            determined by self._kernel to the results dict.
            Accepts:
                fitness (AttrIter) : An AttrIter obj with 'ouput' key
            Returns:
                dict: { treeID: FitnessScore }
        """
        fitness = {k: 0.0 for k in results.keys()}

        for trees in results:
            for treeID, attrs in trees.items():
                score = 0.0
                if len(attrs['output']) <= 3:
                    score = 1
                if len(attrs['output']) <= 2:
                    score = 3

                fitness[treeID] = score

        return fitness

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

        # Determine agent shape from input data
        l1_dims = []
        l2_size = 0
        self.depth = len(inputs)

        for i in range(self.depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x sz
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y sz
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg
        l2_size = self.depth + (self.depth * L2_MEMDEPTH)

        # Init layers
        id_prefix = self.ID + '_'
        ID = id_prefix + 'L1'
        self.l1 = ConceptualLayer(ID, self.depth, l1_dims, inputs)
        ID = id_prefix + 'L2'
        self.l2 = IntuitiveLayer(ID, l2_size)
        ID = id_prefix + 'L3'
        self.l3 = LogicalLayer(ID, L3_ADVISOR)

    def __str__(self):
        return 'ID = ' + self.ID

    def _save(self, filename):
        """ Saves a model of the agent. For use by ModelHandler.
            Also causes saves to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        self.l2.model.save()  # L1 does own persistence, L3 does no peristence

    def _load(self, filename):
        """ Loads the agent model from file. For use by ModelHandler.
            Also causes loads to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        self.l2.model.load()  # L1 does own persistence, L3 does no peristence

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
        # L1_outputs[i] becomes L1._nodes[i]'s classification of data_row[i]
        # ------------------------------------------------------------------
        l1_row = [d[0] for d in data_row]  # TODO: Fix [0] in run()

        for i in range(self.depth):
            self.model.log('-- Feeding L1[%d]:\n%s' % (i, str(l1_row[i])))

        l1_outputs = self.l1.forward(l1_row)
        
        # --------------------- Step Layer 2 -------------------------------
        # L2.outputs are the "masked" versions of all L1.outputs w/recurrance
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
        # TODO: Send feedback/noise/"in context" to level 1 ?

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
    agent.start(max_iters=1)
