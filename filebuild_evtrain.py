#!/usr/bin/env python
""" Builds a file for use in training the intutive agent's genetic algorithm.
    
    Input:
    Expects the file denoted by INFILE to have a single string with no spaces
    on each line

    Results:
    The resulting file, specified by OUTFILE, is of format: LABEL, FEATURES
    This file also has the header "s,1,2,3, ..., n", Where n = max_len
    (each FEATURES w/ len < max_len is right-padded with 0 until len = max_len)
    Each FEATURES is a list of the original label's non-zero index in alpha.
    Each LABEL is a concatenated string of those indexes (without padded 0's).

    Modes:
        TODO

    Author: Dustin Fast, 2018
"""


INFILE = 'static/datasets/words.dat'      # File w/single label per line
OUTFILE = 'static/datasets/words_cat.csv'  # Resulting file
MODE = 'cat'  # 'cat' or 'sum'

alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

results = []


def easy_join(lst, seperator, last_seperator):
    """ Given a list, seperator (str), and last element seperator (str),
        returns a string of the list items in joined as specified. Example:
            Given lst = ['one', 'two', 'three'],
                  seperator = ', '
                  last_sep = ', and'
            Returns "one, two, and three"
    """
    listlen = len(lst)
    if listlen > 1:
        return seperator.join(lst[:-1]) + last_seperator + lst[-1]
    elif listlen == 1:
        return lst[0]
    elif listlen == 0:
        return ''


if __name__ == '__main__':
    # Get lines from INFILE as a list
    with open(INFILE, 'r') as f:
        lines = f.read().split('\n')
    
    max_len = max([len(line) for line in lines])
    
    # Open output file
    outfile = open(OUTFILE, 'w')

    # Build and append file header
    header = ''
    for i in range(max_len):
        header += alpha[i].upper() + ','
    outfile.write(header + 's\n')

    if MODE == 'sum':
        for line in lines:
            features = []
            for ch in line:
                features.append(alpha.index(ch) + 1)

            while len(features) < max_len:
                features.append(0)

            label = sum(features)
            features = [str(f) for f in features]
            features = easy_join(features, ',', ',')
            outfile.write(features + ',' + str(label) + '\n')

    elif MODE == 'cat':
        for line in lines:
            label = ''
            features = []
            for ch in line:
                dx =  str(alpha.index(ch) + 1)
                label += dx
                features.append(dx)

            while len(features) < max_len:
                features.append('0')

            features = easy_join(features, ',', ',')
            outfile.write(features + ',' + str(label) + '\n')

    # Close output file
    outfile.close()


