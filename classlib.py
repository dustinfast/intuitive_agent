#!/usr/bin/env python
""" A collection of shared and/or parent classes for the intutive agent.

    Conventions:

    # TODO: 

    Author: Dustin Fast, 2018
"""

# Imports
import os
import torch    # Used in eval() - May throw "unused" linter error.
import logging

# Constants
OUT_PATH = 'var/models'
LOGFILE_EXT = '.log'
LOG_LEVEL = logging.DEBUG


class Model(object):
    """ The Model class, used by many of the inutive agent's classes. Exposes 
        logging, saving, and loading methods. A "child" is defined here as
        any module using this class, not necessarily inheriting from it.

        Console Output:
            Each child may output via self.log() with console_out enabled.
        
        Model Persistance:
            If persist mode enabled, any string passed to self.log() will also
            be logged to the model's logfile OUT_PATH/child_type/child_ID.log.
            Additionally, saving/loading of the model is enabled to/from file
            OUT_PATH/child_type/child_ID.model_ext w/ self.save() & self.load()
    """
    def __init__(self, child, console_out, persist, **kwargs):
        """ Accepts the following parameters:
            child (obj)             : A ref to the child object
            console_out (bool)      : Console output flag
            persist (bool)          : Persist mode flag

            If persist, kwargs must include:
            model_ext (str)    : Model filename extension. Ex: '.pt'
            save_func (str)    : Save function and args (see example)
            load_func (str)    : Load function and args (see example)
            
            Load/Save function example ('MODEL_FILE' placeholder required): 
                save_func = "torch.save(self.state_dict(), 'MODEL_FILE')"
                load_func = "load_state_dict(torch.load('MODEL_FILE'))"
                The func string will be used to save/load with eval()
                Note: Ensure you import any depencies of save/load funcs
        """
        # Model properties
        self._child = child
        self._console_out = console_out
        self._persist = persist
        self._model_file = None
        self._save_func = None
        self._load_func = None

        # Init persist mode, if specified
        if self._persist:
            model_ext = kwargs.get('model_ext')
            save_func = kwargs.get('save_func')
            load_func = kwargs.get('load_func')

            # Ensure good accompanying kwargs
            if not (model_ext and save_func and load_func):
                raise Exception(
                    'ERROR: Persistant mode set but missing attributes.')

            # Form filenames from kwargs
            child_type = child.__class__.__name__
            output_path = OUT_PATH + '/' + child_type
            file_prefix = output_path + '/' + child.ID
            log_file = file_prefix + LOGFILE_EXT
            self._model_file = file_prefix + model_ext

            # Replace 'MODEL_FILE' placeholder in load/save_func
            self._save_func = save_func.replace('MODEL_FILE', self._model_file)
            self._load_func = load_func.replace('MODEL_FILE', self._model_file)
            
            if self._save_func == save_func or self._load_func == load_func:
                raise Exception("ERROR: Missing 'MODEL_FILE' placeholder.")

            # Replace refs to self with refs to child obj
            self._save_func = self._save_func.replace('self.', 'self._child.')
            self._load_func = self._load_func.replace('self.', 'self._child.')

            # Create output path if not exists
            if not os.path.exists(OUT_PATH):
                os.mkdir(OUT_PATH)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            # Init logger and output initialization statment
            logging.basicConfig(filename=log_file,
                                level=LOG_LEVEL,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self.log('*** Initialized ' + child_type + ' ***:\n' + str(child))

            # Denote model file and, if it exists, load the model from it
            if os.path.isfile(self._model_file):
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
        """ Saves the model to the file given by self._model_file.
        """
        try:
            eval(self._save_func)
            self.log('Saved model to: ' + self._model_file)
        except Exception as e:
            err_str = 'Error saving model: ' + str(e)
            self.log(err_str, level=logging.error)
            raise Exception(err_str)

    def load(self):
        """ Loads the model from the file given by self._model_file.
        """
        try:
            eval(self._load_func)
            self.log('Loaded model from: ' + self._model_file)
        except Exception as e:
            err_str = 'Error loading model: ' + str(e)
            self.log(err_str, level=logging.error)
            raise Exception(err_str)
