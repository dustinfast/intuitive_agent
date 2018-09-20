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
import evolve

class Layer(object):
    """ The top level layer object, from which each layer of the agent inherits
        Each child class is expected to override every member declared here 
        in a way congruent with what they are defined to represent here.
    """
    def __init__(self): 
        self.size = -1      # An n-tuple representing the layer-dependent size

    def forward(self):
        """ Represents the agent state-machine stepping forward one step.
        """
        raise NotImplementedError

    def load(self):
        """ Loads the layer from file in a layer-dependent way.
        """
        raise NotImplementedError


class LayerOne(Layer):
    """
    """
    def __init__(self, num_anns, ann_args):
        """ num_anns (int):     Number of anns this layer is composed of
            ann_args (tuple):   Well-formed paramters for ann.ANN()
        """
        self.size = num_anns
        self.anns = [ann.ANN(ID=i, *ann_args) for i in range(num_anns)]


class LayerTwo(Layer):
    """
    """
    def __init__(self):
        raise NotImplementedError
        # self.evolver = evolve.


class LayerThree(Layer):
    """
    """
    def __init__(self):
        raise NotImplementedError


class Agent(object):
    """
    """
    def __init__(self, ID):
        self.ID = ID


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
