#!/usr/bin/env python
""" A module for determining logical/conceptual associatitivy. See object def
    for more info.

    Module Structure:
        Class Logical contains static methods for determining various kinds
        of logical associations.

    Dependencies:
        Requests

    Usage: 
        See "__main__" for example usage.

    TODO:
        Ensure string passed to is_python won't break anything


    Author: Dustin Fast, 2018
"""
import os
import sys
import queue
import requests
import multiprocessing

class Logical(object):
    """ Static methods for determining logical/conceptual associativity. Ex:
        Logical.is_noun("ball") returns True, where Logical.is_noun("kick")
        returns False.
    """
    @staticmethod
    def is_python(string, timeout=5):
        """ Returns true iff the given string is a valid python string, as
            determined by an eval/exec of the given string - this occurs in
            a seperate process, so the current applicaton is not affected.
            Accepts:
                string (str)    : The string to be checked
                timeout (int)   : Give up after "timeout" seconds (0=never).
                                  On timeout, False is returned.
        """
        # return_queue = multiprocessing.Queue()  # Subproc results queue
        subproc = _ExecProc()
        subproc.start()
        try:
            subproc.in_queue.put_nowait(string)
            results = subproc.out_queue.get(timeout=timeout)
            return results
        except queue.Empty:
            subproc.terminate()
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


class _ExecProc(multiprocessing.Process):
    """ A processes for performing an exec in a seperate memory space, with
        STDOUT redirected to null, so we don't see exec's output (if any).
        Accepts (passed in via self.in_queue): 
            A string to be exec'd.
        Returns (via self.out_queue):
            "True" if the given string can be succesfully exec'd, else "False".
    """
    def __init__(self):
        super(_ExecProc, self).__init__()
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()

    def run(self):
        string = self.in_queue.get()  # Wait for input string

        # "exec" the string with stdout to null so we don't see its output
        with open(os.devnull, 'w') as sys.stdout:
            try:
                exec(string)
                self.out_queue.put_nowait(True)
            except:
                self.out_queue.put_nowait(False)


if __name__ == '__main__':
    # Determine if 4 seperate strings are valid python
    test_str = "print('test')"                              # valid
    result = Logical.is_python(test_str)
    print('"%s" is valid python? %s' % (test_str, result))

    test_str = "a = 3; b = a"                               # valid
    result = Logical.is_python(test_str)
    print('"%s" is valid python? %s' % (test_str, result))

    test_str = "a = b"                                      # b is not defined
    result = Logical.is_python(test_str)
    print('"%s" is valid python? %s' % (test_str, result))

    test_str = "a = 3; a++"                                 # a++ is not python
    result = Logical.is_python(test_str)
    print('"%s" is valid python? %s' % (test_str, result))
    
