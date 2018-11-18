#!/usr/bin/env python
""" Inverts the case of every alphabetic char in the given file
    Caution: Mem complexity of this module grows linearly with given file size.

    Author: Dustin Fast, 2018
"""

import sys

if __name__ == '__main__':
    try:
        # Read in the given file
        with open(sys.argv[1], 'r') as f:
            lines = f.read().split('\n')

        # Invert the case of each char in each line
        new_lines = []
        for line in lines:
            lst = [ch for ch in line]
            for i, ch in enumerate(lst):
                uni = ord(ch)
                if (uni > 64 and uni < 91):
                    lst[i] = ch.lower()
                elif (uni > 96 and uni < 123):
                    lst[i] = ch.upper()
            new_lines.append(''.join(lst))

        # Write them back to the file
        stop_at = len(new_lines) - 1
        with open(sys.argv[1], 'w') as f:
            for i, line in enumerate(new_lines):
                f.write(line)
                if i < stop_at:
                    f.write('\n')

    except IndexError:
        print('ERROR: No filename specified from command line.')
