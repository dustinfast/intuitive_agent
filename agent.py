#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for description of the agent and the application as a whole.
        
    If CONSOLE_OUT = True:
        The Agent and its sub-modules print their output to stdout

    If PERSIST = True:
        Agent and its sub-module states persists between executions via files
        PERSIST_PATH/ID.MODEL_EXT, and their output is logged to 
        PERSIST_PATH/ID.LOG_EXT.

    Module Structure:
        Agent() is the main interface. It expects training/validation data as
        an instance obj of type classlib.DataFrom(). 
        Agent persistence and output is handled by classlib.ModelHandler().

    Dependencies:
        PyTorch
        Numpy
        Sympy
        Scikit-learn
        MatplotLib

    Usage: 
        Run from the terminal with './agent.py'.

    # TODO: 
        REPL (Do this last)
        Training func (layer 1 done)
        Layer 2 pipe
        Classify layer3 output in agent.step
        model.log outputs to single file for all sub-modules


    Author: Dustin Fast, 2018
"""

# Imports
import threading
import logging

import torch

from ann import ANN
from evolve import Evolver
from classlib import ModelHandler, DataFrom


# Constants
CONSOLE_OUT = False
PERSIST = True
MODEL_EXT = '.ag'
L3_OUT_NODES = 26


class Agent(threading.Thread):
    """ The intutive agent.
        The constructor accepts the agent's "sensory input" data, from which
        the layer dimensions are derived. After init, start the agent from 
        the terminal with 'agent start', which runs the agent as a seperate
        process - running this way allows the user to still interact with the
        agent while it's running via the terminal.
    """
    def __init__(self, ID, input_data):
        """ Accepts the following parameters:
            ID (str)                : The agent's unique ID
        """
        threading.Thread.__init__(self)
        self.ID = ID
        self.depth = None
        self.data = input_data        
        self.model = None       # The ModelHandler, defined below
        self.running = False    # Denotes thread is running

        # Determine agent layer dimensions from input data
        self.depth, l1_dims, l3_dims = self._get_dimensions(input_data)

        # Define agent layers - each layer is a dict of nodes and outputs
        self.layer1 = {'nodes': [], 'outputs': []}
        self.layer2 = {'node': None, 'outputs': []}
        self.layer3 = {'node': None, 'output': None}

        # Init the agent layers. They will each auto-load, if able, on init.
        for i in range(self.depth):
            # Build node ID prefixes & suffixes, denoting the agent and depth
            prefix = self.ID + '_'
            suffix = str(i)

            # Init layer1 and 2 outputs at this depth
            self.layer1['outputs'].append([None for i in range(self.depth)])
            self.layer2['outputs'].append([None for i in range(self.depth)])

            # Init the layers with singular nodes (i.e. at depth 0 only)
            if i == 0:
                id2 = prefix + 'vl2_node' + suffix
                id3 = prefix + 'lv3_node' + suffix
                self.layer2['node'] = Evolver(id2, CONSOLE_OUT, PERSIST)
                self.layer3['node'] = ANN(id3, l3_dims, CONSOLE_OUT, PERSIST)
                self.layer3['ouputs'] = [None for i in range(l3_dims[2])]

            # Init layer 1 node at this depth
            id1 = prefix + 'lv1_node' + suffix
            self.layer1['nodes'].append(ANN(id1, l1_dims[i], CONSOLE_OUT, PERSIST))

        # Init the load, save, log, and console output handler
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load(MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        ret_str = 'ID = ' + self.ID
        return ret_str

    def train(self, train_data, val_data):
        """ Trains the agent from the given DataFiles as follows -
            Layer One:

            Layer Two:

        """
        # Train each layer1 node
        for i in range(self.depth):
            self.layer1['nodes'][i].train(
                train_data[i], epochs=100, lr=.01, alpha=.9 , noise=None)

        for i in range(self.depth):
            self.layer1['nodes'][i].validate(val_data[i], verbose=True)

        # TODO: Train layer2 from train data

        # TODO: Train layer3 from subset of all three

    def _step(self, data_row):
        """ Steps the agent forward one step with the given data row: A list
            of tuples (one for each depth) of inputs and targets. Each tuple,
            by depth, is fed to layer one, who's ouput is fed to layer 2, etc.
            Note: At this time, the 'targets' in the data row are not used.
            Accepts:
                data_row (list)      : [(inputs, targets), ... ]
        """
        # debug
        print('\nSTEP')
        # for d in data:
        #     print(d)
        #     print('\n')

        # Ensure well formed data_row
        if len(data_row) != self.depth:
            err_str = 'Mismatched data_row size - expected ' + str(self.depth)
            err_str += ', recieved ' + str(len(data_row))
            self.model.log(err_str, logging.error)
            exit(-1)

        # Feed inputs to layer 1
        for i in range(self.depth):
            inputs = data_row[i][0]
            self.model.log('Feeding L1, node ' + str(i) + ':\n' + str(inputs))
            self.layer1['outputs'][i] = self.layer1['nodes'][i](inputs)

            # debug
            print(self.layer1['nodes'][i].classify_outputs(
                self.layer1['outputs'][i]))

        # Feed layer 1 outputs to layer 2 inputs
        for i in range(self.depth):
            self.model.log('Feeding L2:\n' + str(self.layer1['outputs'][i]))
            # TODO: Evolve through layer 2
            self.layer2['outputs'][i] = self.layer1['outputs'][i]

        # Reshape layer 2 outputs to match layer 3 input size
        l3_inputs = torch.cat(
            [self.layer2['outputs'][i] for i in range(self.depth)], 0)

        self.model.log('Feeding L3 w:\n' + str(l3_inputs))
        self.layer3['output'] = self.layer3['node'](l3_inputs)

        # print(self.layer3['output'])
        # print(self.layer3['node'].classify(self.layer3['output']))
            
        # On new connection: Prompt for feedback, or search, to verify

        # for out in self.layer3['outputs']:
        #     print(out)

    def start(self, data, stop_at_eof=False):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
            Accepts:
                data (list)         : A list of classlib.DataFrom objects
                stop_at_eof (bool)  : Denotes run til end of data
        """
        self.model.log('Agent thread started.')
        min_rows = min([d.row_count for d in data])
        self.running = True

        # Init layer1 class labels
        for i in range(self.depth):
            self.layer1['nodes'][i].set_labels(data[i].class_labels)

        while self.running:
            # Iterate each row of each dataset
            for i in range(min_rows):
                row = []
                for j in range(self.depth):
                    row.append([d for d in iter(data[j][i])])
                
                # Step agent forward one step
                self._step(row)

            # If at eof and stop at eof specified, stop.   
            if stop_at_eof:
                break
        self.stop()

    def stop(self):
        """ Stops the thread. May be called from the REPL, for example.
        """
        self.model.log('Stopped.')
        self.running = False

    def _get_dimensions(self, in_data):
        """ Helper function. Determines agent's shape from the given data.
            Returns the following 3-tuple: (depth = an int,
                                            layer1_dims = [int, int, int],
                                            layer3_dims = [int, int int])
            Assumes:
                Agent layer 1 is composed of ANN's with 3 layers (x, h, and y)
            Accepts:
                data (list)     : A list of DataFrom objects
        """
        depth = len(in_data)
        l1_dims = []
        l3_dims = []
        l3_y = 0
        
        # Determine Layer 1 dimensions
        for i in range(depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(in_data[i].feature_count)     # x layer size
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(in_data[i].class_count)       # y layer size
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg
            l3_y += l1_dims[i][2]                           # Total L1 outputs

        # Determine layer 3 dims from layer 1 dims
        l3_dims.append(l3_y)            # x sz is combined l1 outputs
        l3_dims.append(0)               # h layer sz placeholder
        l3_dims.append(L3_OUT_NODES)    # y layer sz
        l3_dims[1] = int((l3_dims[0] + l3_dims[2]) / 2)  # h sz iss xy avg
    
        return depth, l1_dims, l3_dims


if __name__ == '__main__':
    # Agent "sensory input" data. Length of this list denotes the agent depth.
    in_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 1 training data. Length must match len(in_data) 
    l1_train = [DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True),
                DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 1 validation data - Length must match len(in_data)
    l1_vald = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # Layer 2 training data
    l2_data = DataFrom('static/datasets/letters.csv', normalize=True)

    # Layer 3 "Resource" data
    l3_data = DataFrom('static/datasets/letters.csv', normalize=True)

    # Instantiate the agent (agent shape is derived automatically from in_data)
    agent = Agent('agent1', in_data)

    # Train the agent on the training and val sets
    agent.train(l1_train, l1_vald)

    # Start the agent thread with the in_data list
    agent.start(in_data, True)

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
