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
        Learn to associate metric from types of each element in given data

    Author: Dustin Fast, 2018
"""

import requests

MODEL_EXT = '.cn'

class Connective(object):
    def __init__(self, ID):
        self.mode = mode
        self.success = success
    
    def is_valid(self, data, try_funcs, succ_funcs=[]):
        """ Returns True iff any f(data['string']) returns true for any f in
            try_funcs. Additionally, for each f evaluating to True in this way, 
            calls f(data['success_args']) for every f in succ_funcs.
            Accepts:
                data (dict)       : Must contain at least two keys: 'string'
                                    and 'success_args', where 'string' is the
                                    string to be evaluated and 'success_args'
                                    are the arguments to any f in success_funcs.
                try_funcs (list)  : Functions for evaluating each 
                                    data['string'] (ex: Logical.is_noun).
                succ_funcs(list)  : Functions to call when any f in try_funcs
                                    returns True. Must accept a single argument
                                    of the type given by data['success_args'].
        """
        

    @staticmethod
    def is_python(string):
        """ Returns true iff the given string is a valid python string.
            Accepts:
                string
        """

    @staticmethod
    def is_noun(string, lang='en'):
        """ Returns True iff the given string is a noun in the english language.
            Results are obtained from OxfordDictionaries.com via HTTP GET.
            For API info, see: https://developer.oxforddictionaries.com
            Accepts:
                string (str)    : The word of interest
                lang (str)      : Language. Ex: en = english, es = spanish, etc.
        """
        # OxfordDictionaries.com API credentials
        creds = {'app_id': 'fa088f2c',
                 'app_key': '71fca31a4ca067d4d3df45997ce78b0e'}

        # URL, including language and word in question
        url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/'
        url += lang + '/'
        url += word
        url += '/lexicalCategory%3Dnoun'
        url += '/regions=us'

        # Query OxfordDictionaries.com
        try:
            r = requests.get(url, headers=creds)
            if r.status_code == 200:
                return True
        except:
            raise Exception("HTTP GET failed - Connection error.")
