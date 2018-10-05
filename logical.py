#!/usr/bin/env python
""" An object that takes a number of inputs and determines if there exists
    some logical or conceptual association between them, and if so persorms
    the specified actions

    Ex: Given a string such as "ball" and the function is_noun, determines if
        the string represents a noun.

    Module Structure:
        Persistence and output is handled by classlib.ModelHandler().

    Dependencies:
        Requests

    Usage: 
        See "__main__" for example usage.

    # TODO: 
        

    Author: Dustin Fast, 2018
"""

import requests
import multiprocessing

MODEL_EXT = '.log'

class Logical(object):
    def __init__(self, ID):
        """ Accepts:
                ID (str): The objects unique identifier
        """
        self.ID = ID
    
    def is_valid(self, data, try_func, succ_funcs=[], fail_funcs=[]):
        """ Returns True iff try_func(data['string']) returns True.
            Additionally, calls f(data['args']) for every f in succ_funcs if
            the try_func call returns True, else calls f(data['args']) for 
            every f in fail_funcs.
            f in try_funcs
            Accepts:
                data (dict)       : Must contain at least two keys - 'string'
                                    and 'args', where 'string' is the
                                    string to be evaluated and 'args'
                                    are the arguments (of any type) passed
                                    to f's in succ_funcs and fail_funcs.
                try_func (func)   : Function for evaluating each 
                                    data['string'] (ex: Logical.is_python).
                succ_funcs(list)  : Optional. Functions to call when try_func 
                                    returns True. Each must accept the args 
                                    given by data['args'].
                succ_funcs(list)  : Optional. Functions to call when try_func 
                                    returns False. Each must accept the args 
                                    given by data['args'].
        """
        for d in data:
            args = d['args']
            [[succ_f(args) for succ_f in succ_funcs] if try_func(d['string'])
             else [fail_f(args) for fail_f in fail_funcs]]
        
    @staticmethod
    def is_python(string):
        """ Returns true iff the given string is a valid python string, as
            determined by attempting to compile the string into byte code.
        """
        try:
            compile(string, '<string>', 'exec')
            return True
        except:
            return False
        
    @staticmethod
    def is_noun(string, lang='en'):
        """ Returns True iff the given string is a noun in the english language.
            Results are obtained from OxfordDictionaries.com via HTTP GET.
            For API info, see: https://developer.oxforddictionaries.com
            Accepts:
                string (str)    : The string of interest
                lang (str)      : Language. Ex: en = english, es = spanish, etc.
        """
        # OxfordDictionaries.com API credentials
        api_creds = {'app_id': 'fa088f2c',
                     'app_key': '71fca31a4ca067d4d3df45997ce78b0e'}

        # URL, including language and word in question
        url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/'
        url += lang + '/'
        url += string
        url += '/lexicalCategory%3Dnoun'
        url += '/regions=us'

        # Query OxfordDictionaries.com
        try:
            r = requests.get(url, headers=api_creds)
            if r.status_code == 200:  # API specific status code
                return True
        except:
            print("ERROR: HTTP GET failed for is_noun - Connection error.")


if __name__ == '__main__':
    # Define data
    data_valid_python1 = {'string': 'print("test")',
                          'args': 'string = "print("test")"'}
    data_valid_python2 = {'string': 'a = 3',
                          'args': 'string = "a = 3'}
    data_invalid_python = {'string': 'a++',
                           'args': 'string = "a++"'}

    data = [data_valid_python1, data_valid_python2, data_invalid_python]

    # Define success/fail funcs
    success_funcs = [lambda x: print('Valid: ' + str(x))]
    fail_funcs =  [lambda x: print('Invalid: ' + str(x))]

    # Init object and evaluate data
    logical = Logical('demo')
    logical.is_valid(data, Logical.is_python, success_funcs, fail_funcs)
