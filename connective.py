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
    def __init__(self):
        pass
    
    def is_connective(self, data):
        """ Returns True iff the given data elements can be verified as having
            some conceptual connection.
            Accepts:
                data (str or list)  : A string or list of elements
                metric (function)   : The function to use as a comparator
        """
        if type(data) is str:
            return self.is_noun(data)
        else:
            raise Exception("Unhandled data type encountered.")

    #TODO: def is_valid_python():
        # eval. If success, add to KB, by domain label = python

    @staticmethod
    def is_noun(word, lang='en'):
        """ Returns True iff the given word is a noun in the english language.
            Results are obtained from OxfordDictionaries.com via HTTP GET.
            For API info, see: https://developer.oxforddictionaries.com
            Accepts:
                word (str)      : The word of interest
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
