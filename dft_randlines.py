#!/usr/bin/env python
""" Randomizes the lines in a file specified at the cmd line.
    Cautin: Mem complexity of this module grows linearly with given file size.

    # TODO: Move to datasets/tools

    Author: Dustin Fast, 2018
"""

import sys
import random


if __name__ == '__main__':
    try:
        # Read in the given file
        with open(sys.argv[1], 'r') as f:
            lines = f.read().split('\n')

        # Randomize the lines
        random.shuffle(lines)

        # Write them back to the file
        with open(sys.argv[1], 'w') as f:
            [f.write(l + "\n") for l in lines]
    except IndexError:
        print('ERROR: No filename specified from command line.')