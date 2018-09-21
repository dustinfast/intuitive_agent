#!/usr/bin/env python
"""  The intuitive agent, implemented as a state machine.

    Conventions:
        x = Input layer (i.e. the set of input-layer nodes)
        y = Output layer
        t = Some arbitrary tensor

    The agent layers:
        

    # TODO: 
        REPL


    Author: Dustin Fast, 2018
"""

# Imports
import queue
import multiprocessing

from ann import ANN
from evolve import Evolver


# Constants
CONSOLE_OUT = True
PERSIST = False


class LayerOne(object):
    """ The agent's "conceptual" Layer.
    """
    def __init__(self, size, dims):
        """ Accepts the folllowing parameters:
            size (int)          : Number of anns this layer is composed of
        """
        self.size = size
        self.anns = [ANN(i, dims, CONSOLE_OUT, PERSIST) for i in range(dims[2])]
        self.outputs = [None for i in range(dims[2])]

    def forward(self, inputs):
        """ Steps the layer forward one step with the given inputs.
        """
        for i in range(self.size):
            self.outputs[i] = self.anns[i].classify(inputs[i])


class LayerTwo(object):
    """ The agent's "intutive" layer.
    """
    def __init__(self, size):
        """ Accepts the folllowing parameters:
        """
        self.outputs = None
        self.evolver = Evolver('evolver', CONSOLE_OUT, PERSIST)

    def forward(self, inputs):
        """ Steps the layer forward one step with the given inputs.
        """
        self.outputs = inputs  # temp


class LayerThree(object):
    """ The agent's "attentive" layer.
    """
    def __init__(self, size):
        """ Accepts the folllowing parameters:
        """
        self.outputs = None

    def forward(self, inputs):
        """ Steps the layer forward one step with the given inputs.
        """
        self.outputs = inputs  # temp


class Agent(multiprocessing.Process):
    """ The intutive agent, implemented as a seperate process to allow
        REPL with poison pill.
    """
    def __init__(self, ID, dims, dataset):
        """ Accepts the following parameters:
            ID (int)        : The agent's unique ID
            dims (tuple)    : The layer size dimensions
        """
        multiprocessing.Process.__init__(self)
        self.ID = ID
        self.data = dataset
        self.killq = multiprocessing.Queue()  # Input queue for poison pill

        # Agent layers (see README.md for detailed description)
        self.layer_one = LayerOne(dims[0])
        self.layer_two = LayerTwo(dims[1])
        self.layer_three = LayerThree(dims[3])

    def forward(self, inputs):
        """ Steps the agent forward one step with the given inputs.
        """
        # Step each layer forward w/prev layer's output as next layer's input
        self.layer_one.forward(inputs)
        self.layer_two.forward(self.layer_one.outputs)
        self.layer_three.forward(self.layer_two.outputs)
        
    def run(self):
        """ Steps the agent forward indefinitely until poison pill received.
        """
        i = -1
        while True:
            # Increment i,resetting if needed
            i += 1
            if i >= self.data.size:
                i = 0

            # Step agent forward one step
            self.forward(self.data[i])
            
            # Check for poison pill
            try:
                self.killq.get(timeout=.1)
                return
            except Queue.Empty:
                pass


if __name__ == '__main__':
    dataset = [1, 2, 3, 4, 5]
    agent = Agent('test_agent', ((16, 14, 26), 3, 3), dataset)

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
