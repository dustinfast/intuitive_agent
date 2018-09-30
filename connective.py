#!/usr/bin/env python
""" A "connective" module that takes a number of inputs and determines if 
    there exists a conceptual association between them.
    Ex: Given letters t, h, and e, determines that they form the concept "the".

    Module Structure:
        Persistence and output is handled by classlib.ModelHandler().

    Dependencies:
        Requests

    Usage: 
        See "__main__" for example usage.

    # TODO: 
        Noise Params
        Fix: All unique classes must be present in both training and val set
        Best check point persistence (for online learning)
        EMU: new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]


    Author: Dustin Fast, 2018
"""

import requests

MODEL_EXT = '.cn'

def is_noun(word, lang='en'):
    """ Returns True iff the given word is a noun in the US/English dictionary.
        Results are obtained via HTTP GET request to OxfordDictionaries.com. 
        API info may be found at: https://developer.oxforddictionaries.com/
        Accepts:
            word (str)      : The word of interest
            lang (str)      : Language. Ex: en = english, es = spanish, etc.
    """
    # Oxford dictionary API credentials
    app_id = 'fa088f2c'
    app_key = '71fca31a4ca067d4d3df45997ce78b0e'

    # URL, including language and word in question
    url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/'
    url += lang + '/'
    url += word
    url += '/lexicalCategory%3Dnoun'
    url += '/regions=us'

    # Query OxfordDictionaries.com
    try:
        r = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})
        if r.status_code == 200:
            return True
    except:
        raise Exception("HTTP GET failed - Connection error.")
