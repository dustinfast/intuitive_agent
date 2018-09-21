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
import ann
from evolve import Evolver
from classlib import Pipe

class LayerOne(object):
    """ The agent's "conceptual" Layer.
    """
    def __init__(self, size, ann_args):
        """ size (int):     Number of anns this layer is composed of
            ann_args (tuple):   Well-formed paramters for ann.ANN()
        """
        self.size = size
        self.anns = [ann.ANN(ID=i, *ann_args) for i in range(size)]
        self.outputs = [None for i in range(size)]

    def forward(self, inputs):
        """ Steps the layer forward one step.
        """
        for i in range(self.size):
            self.outputs[i] = self.anns[i].classify(inputs[i])


class LayerTwo(object):
    """ The agent's "intutive" layer.
    """
    def __init__(self):
        self.evolver = Evolver('evolver', console_out=True, persist=False)

    def forward(self, inputs):
        """ Steps the layer forward one step.
        """
        raise NotImplementedError


class LayerThree(object):
    """ The agent's "attentive" layer.
    """
    def __init__(self):
        raise NotImplementedError


class Agent(object):
    """ The intutive agent, implemented as an iterator.
    """
    def __init__(self, ID, max_steps=None):
        self.ID = ID
        self.max_steps = max_steps

    def __iter__(self):
        self.a = 0
        self.b = 1
        return self

    def __next__(self):
        if self.max_steps and self.a > self.max_steps:
            raise StopIteration

        self.a, self.b = self.b, self.a + self.b
        return self


if __name__ == '__main__':
    raise NotImplementedError
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
