#!/usr/bin/env python
""" 
    Conventions:
        x = Input layer (i.e. the set of input-layer nodes)
        y = Output layer
        t = Some arbitrary tensor

    The agent layers:
        

    # TODO: 
        

    Author: Dustin Fast, 2018
"""

# Suppose data sets A, B, C

# Train sub-ann[0] on A, sub-ann[1] on B, sub_ann[2] on C, etc...

# Train genetic sub-layer on training set A U B U C with classifier kernel

# Train attentive layer on same training set A U B U C

# Override genetic sub-layer fitness function

# Run agent

# On intutive layer discover new connection in validation set A U B U C ->
# signal good fitness to genetic sub-layer. This represents feedback from env.

# On genetic sublayer recv good fitness? 


