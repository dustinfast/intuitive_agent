#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for description of the agent and the application as a whole.
        
    If CON_OUT = True:
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
        REPL (Do this last)

    Author: Dustin Fast, 2018
"""
import logging
import threading

from ann import ANN
from genetic import GPMask
from logical import Logical
from classlib import ModelHandler, DataFrom

CON_OUT = True
PERSIST = True
MODEL_EXT = '.agent'


class Layer(object):
    """ An abstraction of an agent layer, with node(s) and output(s).
    """
    def __init__(self, node, output):
        self.node = node
        self.output = output


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
        self.l1_depth = None        # Layer 1 node count
        self.model = None           # The model handler
        self.running = False        # Agent thread running flag (set on start)
        self.inputs = input_data    # The agent's "sensory input" data
        self.seq_inputs = is_seq    # Denote input_data is sequential in nature

        # Determine agent shape from input_data
        dims = tuple(self._get_dimensions(input_data))
        self.l1_depth = dims[0]
        l1_dims = dims[1]

        # Declare layers
        self.layer1 = Layer(node=[], output=[])
        self.layer2 = Layer(node=None, output=[])
        self.layer3 = Layer(node=None, output=[])

        # Init layer 1 nodes (each node loads prev state from file, if exists)
        id_prefix = self.ID + '_'
        for i in range(self.l1_depth):
            id1 = id_prefix + 'lv1_node_' + str(i)
            self.layer1.node.append(ANN(id1, l1_dims[i], CON_OUT, PERSIST))
            self.layer1.output.append([None for i in range(self.l1_depth)])
            self.layer1.node[i].set_labels(self.inputs[i].class_labels)

        # Init layers 2 & 3 (each will load prev state from file, if exists)
        id2 = id_prefix + 'lv2_node'
        id3 = id_prefix + 'lv3_node'

        self.layer2.node = GPMask(id2, 5, 15, dims[0], CON_OUT, PERSIST)
        self.layer3.node = Logical(id3)

        # Init the load, save, log, and console output handler
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load(MODEL_FILE')"
        self.model = ModelHandler(self, CON_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = 'ID = ' + self.ID + '\nShape = (\n  '
        str_out += 'l1_depth: ' + str(self.l1_depth) + '\n)'
        return str_out

    def train_layer1(self, L1_train, L1_val, epochs=100, lr=.01, alpha=.9):
        """ Trains the agent's layer one from the given data sets.
            Accepts:
                L1_train (DataFrom)     : L1 training data
                L1_val (DataFrom)       : L1 validation data
                epochs (int)            : Number of training iterations
                lr (float)              : Learning rate
                alpha (float)           : Learning gain/momentum
        """
        # Train and validate each layer 1 node
        for i in range(self.l1_depth):
            self.layer1.node[i].train(
                L1_train[i], epochs=epochs, lr=lr, alpha=alpha , noise=None)
            self.layer1.node[i].validate(L1_val[i], verbose=True)

    def _step(self, data_row):
        """ Steps the agent forward one step with the given data row: A list
            of tuples (one for each depth) of inputs and targets. Each tuple,
            by depth, is fed to layer one, who's ouput is fed to layer 2, etc.
            Note: At this time, the 'targets' in the data row are not used.
            Accepts:
                data_row (list)      : [(inputs, targets)]
        """
        # Ensure well formed data_row
        if len(data_row) != self.l1_depth:
            err_str = 'Bad data_row size - expected sz ' + str(self.l1_depth)
            err_str += ', recieved sz' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Update  Layer 1 ---------------------
        for i in range(self.l1_depth):
            inputs = data_row[i][0]
            self.model.log('Feeding L1, node ' + str(i) + ' w:\n' + str(inputs))

            # Output is node's classification
            self.layer1.output[i] = self.layer1.node[i].classify(inputs)
            
        # --------------------- Update Layer 2 ------------------------
        self.model.log('Feeding L2 w:\n' + str(self.layer1.output))
        self.layer2.output = self.layer2.node.forward(
            list(self.layer1.output), self.seq_inputs)
        
        # --------------------- UpdateLayer 3 --------------------------
        self.model.log('Feeding L3 w:\n' + str(self.layer2.output))

        # Check fitness of each l2 result
        fitness = {k: 0 for k in self.layer2.output.keys()}
        for k, v in self.layer2.output.items():
            for j in v:
                print('L3: ' + j)
                if Logical.is_python(j):
                    print('TRUE!')
                    fitness[k] += .3

        # Signal fitness back to layer 2
        self.layer2.node.update(fitness)

        # TODO: Send feedback / noise param / "in context" to level 1

    def start(self, stop_at_eof=False):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
            Accepts:
                stop_at_eof (bool)  : Denotes run til end of self.inputs
        """
        self.model.log('Agent thread started.')
        min_rows = min([data.row_count for data in self.inputs])
        self.running = True

        while self.running:
            # Step the agent foreward with each row of each dataset
            for i in range(min_rows):
                row = []
                for j in range(self.l1_depth):
                    row.append([row for row in iter(self.inputs[j][i])])
                self._step(row)

            if stop_at_eof:
                self.stop('Stopped due to end of data.')

    def stop(self, output_str='Stopped.'):
        """ Stops the thread. May be called from the REPL, for example.
        """
        self.running = False
        self.join()
        self.model.log(output_str)

    @staticmethod
    def _get_dimensions(in_data):
        """ Helper function - determines agent's shape and output.
            Returns:
                A 2-tuple: (int, [int, int, int]), i.e. depth and L1 dims.
            Assumes:
                Agent layer 1 is composed of ANN's with 3 layers (x, h, and y)
            Accepts:
                in_data (list)     : A list of DataFrom objects
        """
        depth = len(in_data)
        l1_dims = []

        for i in range(depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x node count
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y node count
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg

        return depth, l1_dims

    
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
    # agent.train_layer1(l1_train, l1_vald)

    # Start the agent thread in_data as input data
    agent.start(stop_at_eof=True)

    # Generate run number.

    # Change log dir accordingly

    # Suppose data sets A, B, C

    # Train sub-ann[0] on A, sub-ann[1] on B, sub_ann[2] on C, etc...

    # Train genetic sub-layer on training set A U B U C with classifier kernel

    # Train attentive layer on same training set A U B U C

    # Override genetic sub-layer fitness function

    # Run agent

    # On intutive layer discover new connection in validation set A U B U C ->
    # signal good fitness to genetic sub-layer. This represents feedback from env.

    # On genetic sublayer recv good fitness?
