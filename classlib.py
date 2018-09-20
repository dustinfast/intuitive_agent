#!/usr/bin/env python
""" A collection of shared and/or parent classes for the intutive agent.

    Conventions:

    # TODO: 

    Author: Dustin Fast, 2018
"""

import os
import logging

LOG_LEVEL = logging.DEBUG
OUT_PATH = 'var/models'

class Model(object):
    """ The parent class for many of the inutive agent's classes, exposing 
        logging, saving, loading, and forward methods. 

        Console Output:
            Each child may output via self.log() with console_out enabled.
        
        Model Persistance:
            If persist mode enabled, any string passed to self.log() will also
            be logged to the model's logfile OUT_PATH/child_type/child_ID.log.
            Additionally, saving/loading of the model is enabled to/from file
            OUT_PATH/child_type/child_ID.model_ext w/ self.save() & self.load()

        Inheritance:
            If specified by any method definition below, inheriting classes
            must override that method in a way congruent with what that method
            is defined here to represent.
    """
    def __init__(self, child_ID, child_type, console_out, persist, **kwargs):
        """ Accepts the following parameters:
            child_ID (str)          : Denotes child's unqiue ID
            child_type (str)        : Denotes child type. Ex: 'ANN'  
            console_out (bool)      : Console output flag
            persist (bool)          : Persist mode flag

            If persist, kwargs must include:
            path (str)              : File in/out directory. Ex: 'var/model'
            model_ext (str)         : Model filename extension. Ex: '.pt'
            save_func (function)    : The save function to use
            save_args (tuple)       : A tuple of save_func arguments
            load_func (function)    : The load function to use
            load_args (tuple)       : A tuple of load_func arguments
            
            Save func/arg example: 
                To implement "torch.save(self.state_dict(), self.model_file)":
                Use save_func = torch.save
                and save_args = (self.state_dict(), self.model_file)

            Load func/arg example:
                To implement "load_state_dict(torch.load(self.model_file))":
                use load_func = load_state_dict
                and load_args = (torch.load(self.model_file),)

            Note: For save/load error msg purposes, self.model_file is assumed.
        """
        # Model properties
        self._console_out = console_out
        self._persist = persist
        self.model_file = None

        # Init persist mode, if specified
        if self._persist:
            path = kwargs.get('path')
            model_ext = kwargs.get('model_ext')
            self._save_func = kwargs.get('save_func')
            self._save_args = kwargs.get('save_args')
            self._load_func = kwargs.get('load_func')
            self._load_args = kwargs.get('load_args')

            # Ensure good accompanying args
            if not (path and model_ext and
                    self._save_func and self._save_args and
                    self._load_func and self._load_args):
                raise Exception(
                    'ERROR: Persistant mode set but missing attributes.')

            # Init persist mode and create output path if not exists
            self._persist = True
            if not os.path.exists(path):
                os.mkdir(path)
            file_prefix = path + '/' + child_type + '/' + child_ID

            # Init logger and output initialization statment
            logging.basicConfig(filename=file_prefix + '.log',
                                level=LOG_LEVEL,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self.log('*** Initialized ' + child_type + ' ***:\n' + str(self))

            # Denote model file and, if it exists, load the model from it
            self.model_file = file_prefix + model_ext
            if os.path.isfile(self.model_file):
                self.load()

    def log(self, log_str, level=logging.info):
        """ Logs the given string to the log file, iff persist enabled.
            Also outputs the string to the console, iff console_out enabled.
        """
        if self._persist:
            level(log_str)

        if self._console_out:
            print(log_str)

    def save(self):
        """ Saves the model to the file given by self.model_file.
        """
        try:
            self._save_func(*self._save_args)
            self.log('Saved model to: ' + self.model_file)
        except Exception as e:
            err_str = 'Error saving model: ' + str(e)
            self.log(err_str, level=logging.error)
            raise Exception(err_str)

    def load(self):
        """ Loads the model from the file given by self.model_file.
        """
        try:
            self._load_func(*self._load_args)
            self.log('Loaded model from: ' + self.model_file)
        except Exception as e:
            err_str = 'Error loading model: ' + str(e)
            self.log(err_str, level=logging.error)
            raise Exception(err_str)

    def forward(self):
        """ Represents the state-machine stepping forward one step.
            Each inheriting class is expected to override this method.
        """
        raise NotImplementedError
