#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for description of the agent and the application as a whole.
        
    If CONSOLE_OUT = True:
        The Agent and its sub-modules print their output to stdout

    If PERSIST = True:
        Agent and its sub-module states persist between executions via files
        PERSIST_PATH/ID.MODEL_EXT, and their output is logged to 
        PERSIST_PATH/ID.LOG_EXT.

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
        Auto-tuned training lr/epochs 
        Agent should write to var/models/agent folder
        L2 train/validate function
        Agent model: trained, labels, 
        L2.nodes[].weight (logarithmic decay over time/frequency)
        L2 Persistence
        L2 logger
        L3 logger
        Add branching after some accuracy threshold
        REPL (Do this last)


    Author: Dustin Fast, 2018
"""
import logging
import threading

from ann import ANN
from genetic import GPMask
from logical import Logical
from classlib import ModelHandler, DataFrom

CONSOLE_OUT = True
PERSIST = True
MODEL_EXT = '.agent'
FITNESS_MODE = Logical.is_python


class ConceptualLayer(object):
    """ An abstraction of the agent's conceptual layer (i.e. layer one). 
        Each node loads it's previous state from file, if exists, on init.
        Note: This layer must be trained offline via self.train().
        """
    def __init__(self, id_prefix, depth, dims, inputs):
        """ Accepts:
            id_prefix (str)     : Each nodes ID prefix. Ex: 'Agent1_'
            depth (int)         : How many nodes this layer contains
            dims (list)         : 3 ints - in/hidden/output layer sizes
        """
        self.node = []      # A list of nodes, one for each layer 1 depth
        self.output = []    # A list of outputs, one for each node
        self.depth = depth

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
    """ An abstration of the agent's Intuitive layer (i.e. layer two). 
        A node is created dynamically for each unique layer-one output 
        via self.set_node(). On init, each node will load itself from file, 
        if exists.
        Note: This layer is trained "online" by the operation of the agent.
              It may also be trained offline via self.train().
              
    """ 
    def __init__(self, id_prefix):
        self.ID = id_prefix + 'L2_nodes'
        self.nodes = {}     # All L2 nodes: { ID: node }
        self.node = None    # Currently active node
        self.output = None  # Placholder for output

    def set_node(self, nodeID):
        """ If self.node[ID] exists, sets self.node to that node. Else, creates
            a new node at that ID before setting self.node to it. In this way
            each unqiue output gets it's own intutive "attention" mask.
            Accepts:
                nodeID (str)  : A string (generally some layer one output)
        """
        if not self.nodes.get(nodeID):
            sz = len(nodeID)
            max_trees = sz * 10
            node = GPMask(nodeID, max_trees, 15, sz, CONSOLE_OUT, False)
            self.nodes[nodeID] = node
        else:
            node = self.nodes[nodeID]
        self.node = node

    def train(self):
        raise NotImplementedError


class LogicalLayer(object):
    """ An abstraction of the agent's Logical layer (i.e. layer three).
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
            corresponding to each based on self.mode.
            Accepts:
                results (dict)  : { ID: result }
            Returns:
                fitness (dict)  : { ID: fitness score (float)}
        """
        fitness = {k: 0 for k in results.keys()}
        for k, v in results.items():
            for j in v:
                print('L3: ' + j, sep=': ')
                # if Logical.is_python(j):
                #     print('TRUE')
                #     fitness[k] += .3
                # else:
                #     print('False')
                if len(j) == 3:
                    fitness[k] += 1
                    if j == 'AAA':
                        fitness[k] += 1
        return fitness


class Agent(threading.Thread):
    """ The intutive agent.
        The constructor accepts the agent's "sensory input" data, from which
        the layer dimensions are derived. After init, start the agent from 
        the terminal with 'agent start', which runs the agent as a seperate
        thread (running this does not cause the agent to block, so the user
        can stop it from the command line, etc.
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
        self.l1_depth = None        # Layer 1 depth
        self.model = None           # The model handler
        self.running = False        # Agent thread running flag (set on start)
        self.inputs = input_data    # The agent's "sensory input" data
        self.seq_inputs = is_seq    # Denote input_data is sequential in nature
        self.max_iters = None       # Num input_data iters, set on self.start
        self.verbose = False        # Denotes verbose output, set on self.start
        id_prefix = self.ID + '_'   # Sets up the ID prefix for the sub-layers

        # Determine agent shape from input_data
        dims = tuple(self._shape_fromdata(input_data))
        self.l1_depth = dims[0]

        # Init layers
        self.l1 = ConceptualLayer(id_prefix, self.l1_depth, dims[1], input_data)
        self.l2 = IntuitiveLayer(id_prefix)
        self.l3 = LogicalLayer(FITNESS_MODE)

        # Init the load, save, log, and console output handler
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = 'ID = ' + self.ID + '\nShape = (\n  '
        str_out += 'l1_depth: ' + str(self.l1_depth) + '\n  '
        str_out += 'l2_nodes: ' + str(len(self.l2.nodes.keys())) + '\n)'
        return str_out

    def _step(self, data_row):
        """ Steps the agent forward one step with the given data row: A list
            of tuples (one for each depth) of inputs and targets. Each tuple,
            by depth, is fed to layer one, who's ouput is fed to layer 2, etc.
            Note: At this time, the 'targets' in the data row are not used.
            Accepts:
                data_row (list)     : [inputs, ... ]
                verbose (bool)      : Denotes verbose output
        """
        # Ensure well formed data_row
        if len(data_row) != self.l1_depth:
            err_str = 'Bad data_row size - expected sz ' + str(self.l1_depth)
            err_str += ', recieved sz' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Update Layer 1 ----------------------
        for i in range(self.l1_depth):
            inputs = data_row[i][0]
            if self.verbose:
                self.model.log('L1 node[%d] input:\n%s' % (i, str(inputs)))

            # Output[i] is node[i]'s classification
            self.l1.output[i] = self.l1.node[i].classify(inputs)
            
        # --------------------- Update Layer 2 ------------------------
        l2_nodeID = ''.join(self.l1.output)
        if self.verbose:
            self.model.log(
                'L2 node[%s] input:\n%s' % (l2_nodeID, self.l1.output))
        self.l2.set_node(l2_nodeID)
        self.l2.output = self.l2.node.forward(
            list([self.l1.output]), self.seq_inputs, verbose=self.verbose)
        
        # --------------------- UpdateLayer 3 --------------------------
        if self.verbose:
            self.model.log('Feeding L3 w:\n%s' % str(self.l2.output))

        # Check fitness of each l2 result
        fitness = self.l3.check_fitness(self.l2.output)
        
        # Signal fitness back to layer 2
        self.l2.node.update(fitness)

        # TODO: Send feedback / noise / "in context" to level 1

    def start(self, max_iters=10, verbose=False):
        """ Starts the agent thread.
            Accepts:
                max_iters (int)     : Max times to iterate data set (0=inf)
                verbose (bool)      : Denotes verbose output
        """
        self.max_iters = max_iters
        self.verbose = verbose
        threading.Thread.start(self)

    def run(self):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
        """
        self.model.log('Agent thread started.')
        min_rows = min([data.row_count for data in self.inputs])
        self.running = True
        iters = 0

        # Step the agent foreward with each row of each dataset
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
        """
        self.running = False
        self.model.log(output_str)

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
    in_data = [DataFrom('static/datasets/test/test3_1.dat', normalize=True),
               DataFrom('static/datasets/test/test3_2.dat', normalize=True),
               DataFrom('static/datasets/test/test3_3.dat', normalize=True)]

    # Layer 1 training data (one per node) - length must match len(in_data) 
    l1_train = [DataFrom('static/datasets/test/test3_1.dat', normalize=True),
                DataFrom('static/datasets/test/test3_2.dat', normalize=True),
                DataFrom('static/datasets/test/test3_3.dat', normalize=True)]

    # Layer 1 validation data (one per node) - length must match len(in_data)
    l1_vald = [DataFrom('static/datasets/test/test3_1.dat', normalize=True),
               DataFrom('static/datasets/test/test3_2.dat', normalize=True),
               DataFrom('static/datasets/test/test3_3.dat', normalize=True)]

    # Instantiate the agent (agent shape is derived automatically from in_data)
    agent = Agent('agentD3T2', in_data, is_seq=False)

    # Train and validate the agent
    # agent.l1.train(l1_train, l1_vald)

    # Start the agent thread in_data as input data
    agent.start(max_iters=1, verbose=True)
