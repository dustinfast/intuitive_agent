#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for description of the agent and the application as a whole.
        
    If CONSOLE_OUT = True:
        The Agent and its sub-modules print their output to stdout

    If PERSIST = True:
        Agent and its sub-module states persist between executions via file
        PERSIST_PATH/ID.MODEL_EXT (saved each time the agent thread stops).
        Output is also logged to PERSIST_PATH/ID.LOG_EXT. 

    Module Structure:
        Agent() is the main interface. It expects training/validation data as
        an instance obj of type classlib.DataFrom(). 
        Agent persistence and output is handled by classlib.ModelHandler().

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
        Auto-tuned training lr/epochs based on data files
        L2.node_map[].weight (logarithmic decay over time/frequency)
        L2.node_map[].kb/correct/solution strings
        L3 logging?
        Add branching after some accuracy threshold
        REPL (Do last)
        Changing in_data row count breaks ANN's - it determines their shape 


    Author: Dustin Fast, 2018
"""
import logging
import threading

from ann import ANN
from genetic import GPMask
from connector import Connector
from classlib import ModelHandler, DataFrom

CONSOLE_OUT = True
PERSIST = True
MODEL_EXT = '.agnt'

L2_EXT = '.intu'
L2_MAX_DEPTH = 15                               # Has big perf effect
L3_FITNESS_MODE = Connector.is_python_kwd       # Fitness evaluator

class ConceptualLayer(object):
    """ An abstraction of the agent's conceptual layer (i.e. layer one), which
        represents its "sensory input". 
        Provides interfaces to each layer-node and a its current output.
        Nodes at this level are ANN's representing a single sensory 
        input processing channel, where the input to each channel is a sample
        of some subset of the agent's environment. Its output is then its
        "classification" of  what that input represents.
        On init, each node is loaded by the ANN object from file iff PERSIST.
        Note: This layer must be trained offline via self.train(). After 
        training, each node saves its model to file iff PERSIST.
        """
    def __init__(self, id_prefix, depth, dims, inputs):
        """ Accepts:
            id_prefix (str)     : Each nodes ID prefix. Ex: 'Agent1_'
            depth (int)         : How many nodes this layer contains
            dims (list)         : 3 ints - ANN in/hidden/output layer sizes
        """
        self.node = []  # A list of nodes, one for each layer 1 depth
        self.output = []    # A list of outputs, one for each node
        self.depth = depth  # This layers depth. I.e. it's node count

        for i in range(depth):
            ID = id_prefix + 'L1_node_' + str(i)
            self.node.append(ANN(ID, dims[i], CONSOLE_OUT, PERSIST))
            self.output.append([None for i in range(depth)])
            self.node[i].set_labels(inputs[i].class_labels)
    
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
            self.node[i].train(
                train_data[i], epochs=epochs, lr=lr, alpha=alpha, noise=None)
            self.node[i].validate(val_data[i], verbose=True)


class IntuitiveLayer(object):
    """ An abstration of the agent's Intuitive layer (i.e. layer two), which 
        represents the intutive "bubbling up" of "pertinent" information to 
        layers above it.
        Nodes at this level represent a single genetically evolving population
        of expressions. They are created dynamically, one for each unique 
        input the layer receives (from layer-one), with each node's ID then 
        being that unique input (as a string).
        A population's expressions represent a "mask" applied to the layer's 
        input as it passes through it.
        On init, each previously existing node is loaded from file iff PERSIST.
        Note: This layer is trained in an "online" fashion - as the agent runs,
        its model file is updated for every call to node.update() iff PERSIST.
    """ 
    def __init__(self, ID, id_prefix):
        """ Accepts:
            ID (str)            : This layer's unique ID
            id_prefix (str)     : Each node's ID prefix. Ex: 'Agent1_L2_'
        """
        self.ID = ID
        self.output = None      # Placeholder for current output
        self._curr_node = None  # The node for the current unique input
        self._nodes = {}        # Nodes, as: { nodeID: (obj_instance, output) }
        self.id_prefix = id_prefix + ID + '_node_'
        
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L2_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def forward(self, data, is_seq):
        """ Returns the layer's output after moving the given input_data 
            through it and setting the currently active node according to it
            (the node is created first, if it doesn't already exist).
            Note: We leave it to the caller to set self.output, if desired.
        """
        if not self._nodes.get(data):
            # Init new node ("False", because we'll handle its persistence)
            sz = len(data)
            pop_sz = sz * 10
            node = GPMask(data, pop_sz, L2_MAX_DEPTH, sz, CONSOLE_OUT, False)
            self._nodes[data] = (node, None)
        else:
            node = self._nodes[data][0]

        self._curr_node = node
        return node.forward(list([data]), is_seq)

    def update(self, data):
        """ Updates the currently active node with the given fitness data dict.
        """
        self._curr_node.update(data)

    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write each node ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')
            for k, v in self._nodes.items():
                savestr = v[0].save()
                f.write('"' + k + '": """' + savestr + '""", ')
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
            ID = self.id_prefix + str(i)
            node = GPMask(ID, 0, 0, 0, CONSOLE_OUT, False)
            node.load(v, not_file=True)
            self._nodes[k] = (node, None)
            i += 1


class LogicalLayer(object):
    """ An abstraction of the agent's Logical layer (i.e. layer three), which
        evaluates the fitness of it's input according to the given mode.
        This layer does no persistence or logging at this time.
    """
    def __init__(self, mode):
        """ Accepts:
                mode (function)  : Any func returning True or false when given
                a layer-two output, denoting if that output is fit/productive
        """
        self.mode = mode
        self.node = self.check_fitness

    def check_fitness(self, results):
        """ Checks each result in results and returns a dict of fitness scores
            corresponding to each, as determined by self.mode.
            Accepts:
                results (dict)  : { ID: result }
            Returns:
                fitness (dict)  : { ID: fitness score (float) }
        """
        fitness = {k: 0 for k in results.keys()}
        for k, v in results.items():
            for j in v:
                # print('L3: ' + j, sep=': ')
                # if Connector.is_python(j):
                #     print('TRUE')
                #     fitness[k] += .3
                # else:
                #     print('False')
                if j[0] == 'X' :
                    if len(j) == 5:
                        fitness[k] += 1
                elif len(j) == 3:
                    fitness[k] += 1
        return fitness


class Agent(threading.Thread):
    """ The intutive agent.
        The constructor accepts the agent's "sensory input" data, from which
        the layer dimensions are derived. After init, start the agent from 
        the terminal with 'agent start', which runs the agent as a seperate
        thread (running this does not cause the agent to block, so the user
        can stop it from the command line, etc.
        Persistence: On each iteration of the input data, the agent is saved 
        to a file.
    """
    def __init__(self, ID, input_data, is_seq):
        """ Accepts the following parameters:
            ID (str)           : The agent's unique ID
            input_data (list)  : List of agent input data, one for each L1 ANN
            is_seq (bool)      : Denotes input data is sequential in nature,
                                 i.e., the layer 2 mask will use only ordered
                                 expressions, such as 'A + C + E', as opposed
                                 to something like 'C + E + A'
        """
        threading.Thread.__init__(self)
        self.ID = ID
        self.l1_depth = None        # Layer 1 depth/node count
        self.model = None           # The model handler
        self.running = False        # Agent thread running flag (set on start)
        self.inputs = input_data    # The agent's "sensory input" data
        self.is_seq = is_seq        # Denote input_data is sequential in nature
        self.max_iters = None       # Num input_data iters, set on self.start
        self.L2_nodemap = None      # Pop via ModelHandler, for loading L2

        # Init the load, save, log, and console output handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        # Determine agent shape from input_data
        dims = tuple(self._shape_fromdata(input_data))
        self.l1_depth = dims[0]

        # Init layers
        id_prefix = self.ID + '_'   # Sub-layer node-ID prefix
        self.l1 = ConceptualLayer(id_prefix, self.l1_depth, dims[1], input_data)
        self.l2 = IntuitiveLayer(id_prefix + 'L2', id_prefix)
        self.l3 = LogicalLayer(L3_FITNESS_MODE)

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
        self.model.log('\n****** AGENT STEP ******')

        # Ensure well formed data_row
        if len(data_row) != self.l1_depth:
            err_str = 'Bad data_row size - expected sz ' + str(self.l1_depth)
            err_str += ', recieved sz' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Step Layer 1 -------------------------------
        # L1.output[i] becomes L1.node[i]'s classification of self.inputs[i]
        # ------------------------------------------------------------------
        for i in range(self.l1_depth):
            inputs = data_row[i][0]
            self.model.log('-- Feeding L1 node[%d] w/\n%s' % (i, str(inputs)))
            self.l1.output[i] = self.l1.node[i].classify(data_row[i][0])
            
        # --------------------- Step Layer 2 -------------------------------
        # L2.output becomes the "masked" versions of all L1.outputs
        # ------------------------------------------------------------------
        l2_input = ''.join(self.l1.output)  # stringify L1's output
        self.model.log('-- Feeding L2 node[%s] w/\n: %s' % (l2_input, self.l1.output))
        self.l2.output = self.l2.forward(l2_input, self.is_seq)
        
        # --------------------- Step Layer 3 -------------------------------
        # L3 evals fitness of it's input from L2 and backpropagates signals
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L3 w/\n%s' % str(self.l2.output))
        fitness = self.l3.check_fitness(self.l2.output)

        self.model.log('-- L2 Backprop w/\n%s' % str(fitness))
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
                for j in range(self.l1_depth):
                    row.append([row for row in iter(self.inputs[j][i])])
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
        """ Determines agent's shape from the given list of data sets.
            Assumes each layer 1 node has 3 layers (x, h, and y).
            Accepts:
                in_data (list)     : A list of DataFrom objects
            Returns:
                2-tuple: L1 depth and L1 dims, as (int, [int, int, int]) 
        """
        l1_depth = len(in_data)
        l1_dims = []

        for i in range(l1_depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x node count
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y node count
            l1_dims[i][1] = int(     
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg
            
        return l1_depth, l1_dims


if __name__ == '__main__':
    # Agent "sensory input" data. Length of this list denotes the agent depth.
    in_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 1 training data (one per node) - length must match len(in_data) 
    l1_train = [DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 1 validation data (one per node) - length must match len(in_data)
    l1_vald = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # Instantiate the agent (agent shape is derived automatically from in_data)
    agent = Agent('agent1', in_data, is_seq=False)

    # Train and validate the agent
    # agent.l1.train(l1_train, l1_vald)

    # Start the agent thread in_data as input data
    agent.start(max_iters=5)
