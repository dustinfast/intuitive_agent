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
        Train layer3
        Classify layer3 output in agent.step


    Author: Dustin Fast, 2018
"""

# Imports
import threading
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
        self.layer1 = {'nodes': [], 'outputs': []}
        self.layer2 = {'node': None, 'outputs': []}
        self.layer3 = {'node': None, 'output': None}

        # Init the agent layers. They will each auto-load, if able, on init.
        for i in range(depth):
            # Build node ID prefixes & suffixes, denoting the agent and depth
            prefix = self.ID + '_'
            suffix = str(i)

            # Init layer1 and 2 outputs at this depth
            self.layer1['outputs'].append([None for i in range(depth)])
            self.layer2['outputs'].append([None for i in range(depth)])

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
        """ Trains the agent from the given training and validation files.
        """
        # Train each layer1 node
        for i in range(self.depth):
            self.layer1['nodes'][i].train(
                train_data[i], epochs=100, lr=.01, alpha=.9 , noise=None)

        for i in range(self.depth):
            self.layer1['nodes'][i].validate(val_data[i], verbose=True)

        # TODO: Train layer3 from subset of all three

    def _step(self, data):
        """ Steps the agent forward one step with the given list of tuples
            (one for each depth) of inputs and targets. Each tuple, by depth,
            is fed to layer one, who's ouput is fed to layer 2, etc.
            Accepts:
                data (list)      : [(inputs, targets), ... ]
        """
        # debug
        print('STEP\n')
        # for d in data:
        #     print(d)
        #     print('\n')

        # Feed inputs to layer 1
        for i in range(self.depth):
            inputs = data[i][0]
            self.model.log('Feeding L1, node ' + str(i) + ' w: ' + str(inputs))
            self.layer1['outputs'][i] = self.layer1['nodes'][i](inputs)

            # debug
            print(self.layer1['nodes'][i]._label_from_outputs(
                self.layer1['outputs'][i]))

        # Feed layer 1 outputs to layer 2 inputs
        for i in range(self.depth):
            self.model.log('Feeding L2 w: ' + str(self.layer1['outputs'][i]))
            # TODO: Evolve through layer 2
            self.layer2['outputs'][i] = self.layer1['outputs'][i]

        # Concatt layer 2 outputs into a tensor of size layer 3 inputs
        l3_inputs = torch.cat(
            [self.layer2['outputs'][i] for i in range(self.depth)], 0)

        self.model.log('Feeding L3 w:' + str(l3_inputs))
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
        """ Stops the thread.
        """
        self.model.log('Stopped.')
        self.running = False


def get_dims(data):
    """ Helper function to determine agents dimensions from the data given.
        Returns a 3-tuple of lists: (depth, layer1_dims, layer3_dims)
        Accepts:
            data (list)     : A list of DataFrom objects 
    """
    # Determine agent shape from given data - assumes 3-layer (x, h, y) anns
    l1_dims = []
    l3_dims = []
    l3_y = 0
    depth = len(in_data)
    for i in range(depth):
        l1_dims.append([])                              # New dim in l1_dims
        l1_dims[i].append(in_data[i].feature_count)     # x layer size
        l1_dims[i].append(0)                            # Placeholder for h sz
        l1_dims[i].append(in_data[i].class_count)       # y layer size
        l1_dims[i][1] = int(
            (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz iss xy avg
        l3_y += l1_dims[i][2]                           # count total outputs

    l3_dims.append(l3_y)            # x sz is combined l1 outputs
    l3_dims.append(0)               # h layer sz placeholder
    l3_dims.append(L3_OUT_NODES)    # y layer sz
    l3_dims[1] = int((l3_dims[0] + l3_dims[2]) / 2)  # h sz iss xy avg
 
    return depth, l1_dims, l3_dims


if __name__ == '__main__':
    # Define the agent "sensory input" datasets.
    in_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    tr_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    vl_data = [DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True),
               DataFrom('static/datasets/letters.csv', normalize=True)]

    # in_data = [DataFrom('static/datasets/letters.csv', normalize=True),
    #            DataFrom('static/datasets/letters.csv', normalize=True),
    #            DataFrom('static/datasets/letters.csv', normalize=True)]

    # Determine agent dimensions based on in_data shape
    depth, l1_dims, l3_dims = get_dims(in_data)
    
    # Init the agent, composed of a layer1 of "depth" ANN's. 
    # Each ANN[i] receives rows from in_data[i] as input simultaneously.
    agent = Agent('agent1', depth, l1_dims, l3_dims)

    # Train the agent on the training and val sets
    agent.train(tr_data, vl_data)

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
