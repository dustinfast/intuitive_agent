#!/usr/bin/env python
""" Randomizes the lines in a file specified at the cmd line.
    Caution: Mem complexity of this module grows linearly with given file size.

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
        stop_at = len(lines) - 1
        with open(sys.argv[1], 'w') as f:
            for i, line in enumerate(lines):
                f.write(line)
                if i < stop_at:
                    f.write('\n')

    except IndexError:
        print('ERROR: No filename specified from command line.')
