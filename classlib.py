#!/usr/bin/env python
""" A collection of shared and/or parent classes for the intutive agent.

    # TODO: 

    Author: Dustin Fast, 2018
"""

# Imports
import os
import logging

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable as V
import pandas as pd


# Constants
OUT_PATH = 'var/models'
LOGFILE_EXT = '.log'
LOG_LEVEL = logging.DEBUG


class ModelHandler(object):
    """ The ModelHandler, used by many of the inutive agent's classes for
        logging, saving, and loading methods. A "child" is defined here as
        any module using this class, not necessarily inheriting from it.

        Console Output:
            Each child may output via self.log() with console_out enabled.
        
        Model Persistance:
            If persist mode enabled, any string passed to self.log() will also
            be logged to the model log file OUT_PATH/child_type/child_ID.log.
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


class DataFrom(Dataset):
    """ A set of inputs & targets (i.e. labels & features) as torch.tensors,
        populated from the given CSV file and normalized if specified.
        self.inputs = torch.FloatTensor, with a gradient for torch.optim.
        self.targets = torch.FloatTensors, converted from str->float if needed.
        Assumes CSV file w/no header with format: label, feat_1, ... , feat_n
    """

    def __init__(self, csvfile, normalize=False):
        """ Accepts the following parameters:
            csvfile (str)       : CSV filename
            normalize           : If True, inputs data is normalized
        """
        self.inputs = None                          # 3D Inputs tensor
        self.targets = None                         # 3D Targets tensor
        self.raw_inputs = None                      # Original inputs form
        self.raw_targets = None                     # Original targets form
        self.class_labels = None                    # Unique instance labels
        self.class_count = None                     # Num unique labels
        self.feature_count = None                   # Num input features
        self.row_count = None                       # Num data rows
        self.normalized = normalize                 # Denotes normalized data
        self.fname = csvfile                        # CVS file name

        # Load data
        data = pd.read_csv(csvfile, header=None)    # csvfile -> pd.DataFrame
        inputs = data.loc[:, 1:]                    # All cols but leftmost
        targets = data.loc[:, :0]                   # Only leftmost col

        # Populate non-tensor member info
        self.raw_inputs = inputs
        self.raw_targets = targets
        self.class_labels = sorted(list(data[0].unique()), key=lambda x: x)
        self.class_count = len(self.class_labels)
        self.row_count = len(inputs)

        # Load inputs and normalize if specified
        if normalize:
            self.norm_max = max(inputs.max())
            self.norm_min = min(inputs.min())
            inputs.apply(self._normalize)

        # Store inputs as a torch.tensor with gradients
        self.inputs = V(torch.FloatTensor(inputs.values), requires_grad=True)
        self.feature_count = self.inputs.size()[1]

        # Init targets
        targets = targets.apply(lambda t: self._map_outnode(t.iloc[0]), axis=1)
        self.targets = targets

        # Set raw input members

    def __str__(self):
        str_out = 'Classes: ' + str(self.class_labels) + '\n'
        str_out += 'Row 1 Target: ' + str(self.targets[0]) + '\n'
        str_out += 'Row 1 Inputs: ' + str(self.inputs[0])
        return str_out

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def _map_outnode(self, label):
        """ Given a class label, returns zeroed tensor with tensor[label] = 1.
            Facilitates mapping each class to its corresponding output node
        """
        tgt_width = len(self.class_labels)
        target = torch.tensor([0 for i in range(tgt_width)], dtype=torch.float)
        target[self.class_labels.index(label)] = 1
        return target

    def _normalize(self, t):
        """ Returns a normalized representation of the given tensor
        """
        return (t - self.norm_min) / (self.norm_max - self.norm_min)
