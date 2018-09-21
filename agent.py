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
            id1 = prefix + 'l1_node' + suffix
            self.layer1['nodes'].append(ANN(id1, l1_dims, CONSOLE_OUT, PERSIST))

            # Init the layers with singular nodes (i.e. at depth 0 only)
            if i == 0:
                id2 = prefix + 'l2_node' + suffix
                id3 = prefix + 'l3_node' + suffix
                self.layer2['node'] = Evolver(id2, CONSOLE_OUT, PERSIST)
                self.layer3['node'] = ANN(id3, l3_dims, CONSOLE_OUT, PERSIST)

        print(self)
        exit()

        # Init the load, save, log, and console output handler
        f_save = "self.save('MODEL_FILE')"
        f_load = "self.load(MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def train(self, trainfile, validationfile):
        """ Trains the agent from the given training and validation files.
        """
        raise NotImplementedError

    def forward(self, inputs):
        """ Steps the agent forward one step, with the given inputs (a tensor)
            as input to layer one, and then each layer's output given as
            input to the next layer.
        """
        raise NotImplementedError

    def start(self, data, stop_at_eof=False):
        """ Steps the agent forward indefinitely until poison pill received.
        """
        if not data:
            return

        i = -1
        while self.running:
            i += 1
            if i >= data.row_count:
                if stop_at_eof:
                    break
                i = 0

            # Step agent forward one step
            self.forward(self.data[i])

            # Prompt for feedback, or search resources, to verify new concept
            
        self.model.log('Stopped.')
        return


if __name__ == '__main__':
    # Init the agent (ID, depth_for_l1_l2, layer1_dims, layer3_dims)
    agent = Agent('agent1', 3, (16, 14, 26), (16, 14, 26))

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
