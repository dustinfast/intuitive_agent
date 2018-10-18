#!/usr/bin/env python
""" A module of static methods for determining logical "connectedness" of a
    set of symbols. For example, contains a method for checking if a set of
    letters represents a noun.

    Module Structure:
        Class Connector contains static methods for determining various kinds
        of logical associations/connectedness.

    Dependencies:
        Requests

    Usage: 
        See "__main__" for example usage.

    TODO:
        Ensure string passed to is_python won't break the process.
        Improve is_python perf by using a presistent process


    Author: Dustin Fast, 2018
"""
import os
import sys
import queue
import keyword
import requests
import multiprocessing

class Connector(object):
    """ Static methods for determining logical/conceptual connectedness. Ex:
        Connector.is_noun("ball") returns True, where Connector.is_noun("kick")
        returns False.
    """
    @staticmethod
    def is_python(string, is_kwd=False, timeout=5):
        """ Returns true iff the given string is valid python.
            Accepts:
                string (str)    : The string to be checked
                is_kwd (bool)   : True - denotes check for python keyword.
                                : False - denotes check for python "program".
                timeout (int)   : Give up after "timeout" seconds (0=never).
                                  On timeout, False is returned.
        """

        if is_kwd:
            return keyword.iskeyword(string)

        subproc = _ExecProc()
        subproc.start()
        try:
            # Test string in seperate process (protects current application)
            subproc.in_queue.put_nowait(string)
            results = subproc.out_queue.get(timeout=timeout)
            return results
        except queue.Empty:
            print('TIMEOUT OCCURRED!')
            subproc.terminate()
            return False
   
    @staticmethod
    def is_noun(string, lang='en'):
        """ Returns True iff the given string is a noun in the english language
            according to OxfordDictionaries.com (via HTTP GET - for API info,
            see: https://developer.oxforddictionaries.com).
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
    # Test the following set of string for Python attributes
    strings = ["if", "print('if')", "a = b", "a = 3; b = a", "a = 3; a++"]

    print('Is python program?')
    for s in strings:
        result = Connector.is_python(s)
        print('"%s": %s' % (s, result))

    print('\nIs python keyword?')
    for s in strings:
        result = Connector.is_python(s, is_kwd=True)
        print('"%s": %s' % (s, result))
    
