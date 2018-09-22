#!/usr/bin/env python
"""  The intuitive agent. See README.md for description.
        
    If CONSOLE_OUT = True:
        The Agent and its sub-module output is printed to stdout

    If PERSIST = True:
        Agent and its sub-module states persists between executions via files
        at PERSIST_PATH/ID.MODEL_EXT and their output is logged to 
        PERSIST_PATH/ID.LOG_EXT.

    Module Structure:
        Agent is the main interface. It expects training/validation data as
        a classlib.DataFrom object instance. 
        Agent persistence and output is handled by classlib.ModelHandler.

    Dependencies:
        PyTorch
        Numpy
        Sympy
        Scikit-learn
        MatplotLib

    Usage: 
        Run from the terminal with './agent.py'.

    # TODO: 
        REPL
        Prepend Agent ID to sub-module IDs


    Author: Dustin Fast, 2018
"""

# Imports
import threading
from ann import ANN
from evolve import Evolver
from classlib import ModelHandler, DataFrom


# Constants
CONSOLE_OUT = True
PERSIST = False
MODEL_EXT = '.ag'


class Agent(threading.Thread):
    """ The intutive agent, implemented as a process to allow REPL and
        and demonstrate
    """
    def __init__(self, ID, depth, l1_dims, l3_dims):
        """ Accepts the following parameters:
            ID (int)                : The agent's unique ID
            depth (int)             : Layer 1 and 2 "node" depth
            l1_dims (tuple)         : Layer 1 ann dimensions
            l3_dims (tuple)         : Layer 3 ann dimensions
        """
        threading.Thread.__init__(self)
        self.ID = ID
        self.depth = depth
        self.data = None        # TODO
        self.model = None       # The ModelHandler, defined below
        self.running = False    # Denotes thread is running

        # Define agent layers - each layer is a dict of nodes and outputs
        # Note: Layers 2 and 3 have one node. 
        self.layer1 = {'nodes': [], 'outputs': []}
        self.layer2 = {'node': None, 'outputs': []}
        self.layer3 = {'node': None, 'outputs': []}

        # Init the agent layers. They will each auto-load, if able, on init.
        for i in range(depth):
            # Build node ID prefixes & suffixes, denoting the agent and depth
            prefix = self.ID + '_'
            suffix = str(i)

            # Init layer1 and 2 outputs at this depth
            self.layer1['outputs'].append([None for i in range(depth)])
            self.layer2['outputs'].append([None for i in range(depth)])

            # Init layer 1 node at this depth
            id1 = prefix + 'lv1_node' + suffix
            self.layer1['nodes'].append(ANN(id1, l1_dims, CONSOLE_OUT, PERSIST))

            # Init the layers with singular nodes (i.e. at depth 0 only)
            if i == 0:
                id2 = prefix + 'vl2_node' + suffix
                id3 = prefix + 'lv3_node' + suffix
                self.layer2['node'] = Evolver(id2, CONSOLE_OUT, PERSIST)
                self.layer3['node'] = ANN(id3, l3_dims, CONSOLE_OUT, PERSIST)
                self.layer3['ouputs'] = [None for i in range(l3_dims[2])]

        # Init the load, save, log, and console output handler
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load(MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        print(self.layer1['nodes'][0])
        print(self.layer3['node'])

    def __str__(self):
        raise NotImplementedError

    def train(self, trainfile, validationfile):
        """ Trains the agent from the given training and validation files.
        """
        raise NotImplementedError

    def _step(self, inputs):
        """ Steps the agent forward one step with the given list DataFrom objs
            (one for each depth) as input to layer one, who's ouput is fed to
            layer 2, etc.
        """
        # Feed inputs to layer 1
        for i in range(self.depth):
            self.model.log('Feeding L1, node ' + str(i) + ': ' + str(inputs[i]))
            self.layer1['outputs'][i] = self.layer1['nodes'][i](inputs[i])

        # Feed layer 1 outputs to layer 2 inputs
        for i in range(self.depth):
            self.model.log('Feeding L2: ' + str(self.layer1['outputs'][i]))
            # TODO: Evolve through layer 2
            self.layer2['outputs'][i] = self.layer1['outputs'][i]

        # Feed layer 2 outputs to layer 3 inputs
        for i in range(self.depth):
            # TODO: Convert to the lyer3 input dims
        
        # self.model.log('Feeding L3 ' + str(self.layer2['outputs'][i]))
        # self.layer3['outputs'][i] = self.layer3['node'](
        #     self.layer2['outputs'][i])
            
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
        self.running = True

        # TODO: Init layer1 class labels

        while self.running:
            for row in data:
                inputs = [d for d, _ in iter(row)]
                self._step(inputs)  # Step agent forward one step
            if stop_at_eof:
                break
        self.stop()

    def stop(self):
        """ Stops the thread.
        """
        self.model.log('Stopped.')
        self.running = False


if __name__ == '__main__':
    # Init the agent (ID, depth_for_l1_l2, layer1_dims, layer3_dims)
    depth = 3
    layer1_dims = (16, 14, 26)
    layer2_dims = (26, 14, 16)
    agent = Agent('agent1', depth, layer1_dims, layer2_dims)

    # Define data file
    # datafile = 'static/datasets/test3x4.csv'  # debug
    data1 = DataFrom('static/datasets/test.csv', normalize=True)
    data2 = DataFrom('static/datasets/test.csv', normalize=True)
    data3 = DataFrom('static/datasets/test.csv', normalize=True)
    data = [data1, data2, data3]

    # Start the agent thread with the data list
    agent.start(data, True)

    # agent.start()
    
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
