#!/usr/bin/env python
""" Creates a number of testfiles from the given file with lines randomized
    Caution: Will overwrite existing files - know what you're doing.
    Caution: Mem complexity of this module grows linearly with given file size.

    Author: Dustin Fast, 2018
"""

import sys
import random


if __name__ == '__main__':
    try:
        in_file = sys.argv[1]
        num_files = int(sys.argv[2])
    except IndexError:
        print('ERROR: Missing filename and/or file count args.')
        exit()
    except ValueError:
        print('ERROR: Invalid file count arg.')
        exit()

    # Read in the given file
    with open(in_file, 'r') as f:
        lines = f.read().split('\n')

    # Denote file extension (for incrementing
    file_ext = None 
    if '.' in in_file:
        file_ext = in_file[in_file.index('.'):]
        in_file = in_file[:-len(file_ext)]
    
    for i in range(num_files):
        # Randomize the lines
        random.shuffle(lines)

        # Build output fname
        outfile = in_file + str(i)
        if file_ext:
            outfile += file_ext
            
        # Write lines to output file
        stop_at = len(lines) - 1
        with open(outfile, 'w') as f:
            for i, line in enumerate(lines):
                f.write(line)
                if i < stop_at:
                    f.write('\n')

    
