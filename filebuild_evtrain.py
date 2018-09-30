#!/usr/bin/env python
""" Builds a file for use in training the intutive agent's genetic algorithm.
    
    Input:
    Expects the file denoted by INFILE to have a single string with no spaces
    on each line

    Results:
    The resulting file, specified by OUTFILE, is of format: LABEL, FEATURES
    This file also has the header "s,1,2,3, ..., n", Where n = MAX_LEN
    (each FEATURES w/ len < MAX_LEN is right-padded with 0 until len = MAX_LEN)
    Each FEATURES is a list of the original label's non-zero index in alpha.
    Each LABEL is a concatenated string of those indexes (without padded 0's).

    Author: Dustin Fast, 2018
"""

import classlib

INFILE = 'static/datasets/test/nouns_4.dat'    # File w/single label per line
OUTFILE = 'static/datasets/test/nouns_4f.dat'  # Resulting file
MAX_LEN = 4                                    # Max length of any label

alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

results = []

if __name__ == '__main__':
    # Open output file
    outfile = open(OUTFILE, 'w')

    # Build and append its header
    header = 's'
    for i in range(MAX_LEN):
        header += ',' + str(i + 1)
    outfile.write(header + '\n')

    # Get lines from INFILE as a list
    with open(INFILE, 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        label = ''
        features = []
        for ch in line:
            ch_idx = str(alpha.index(ch) + 1)
            label += ch_idx
            features.append(ch_idx)

        while len(features) < MAX_LEN:
            features.append('0')

        features = classlib.easy_join(features, ',', ',')
        outfile.write(label + ',' + features + '\n')

    # Close output file
    outfile.close()


