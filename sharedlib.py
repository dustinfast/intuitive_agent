#!/usr/bin/env python
""" A collection of shared classes/functions for the intutive agent.

    Dependencies:
        Pandas      (pip install pandas)
        PyTorch     (see https://pytorch.org)

"""
__author__ = "Dustin Fast"
__email__ = "dustin.fast@outlook.com"
__license__ = "GPLv3"


# Std lib
import os
import logging
import logging.handlers

# Third-party
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable as V
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# User configurable
OUT_PATH = 'var'                # Log and Model file output directories
LOGFILE_EXT = '.log'            # Log file extension
LOG_LEVEL = logging.DEBUG       # Log file level
LOG_SIZE = 1 * 1048576          # Log file size (x * bytes in a mb)
LOG_FILES = 2                   # Concurrent locating log file count

# Global logger
g_logger = None


#################
#   Class Lib   #
#################

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
            
            Load/Save function example: 
                save_func = "torch.save(self.state_dict(), 'MODEL_FILE')"
                load_func = "load_state_dict(torch.load('MODEL_FILE'))"
                The func string will be used to save/load with eval()
                Note: The 'MODEL_FILE' placeholder is required iff func !=None
                Note: Ensure depencies of save/load funcs are imported
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
            child_type = child.__class__.__name__.lower()
            output_path = OUT_PATH + '/' + child_type
            file_prefix = output_path + '/' + child.ID
            log_file = file_prefix + LOGFILE_EXT
            self._model_file = file_prefix + model_ext

            # Replace the 'MODEL_FILE' placeholder in load/save_func
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

            # Init global logger (if not already) - it gets first callers ID
            global g_logger
            if not g_logger:
                g_logger = Logger(child.ID, log_file, False)
            
            self.log('*** Initialized ' + child_type + ': ' + str(child))

            # Denote model filename and, if it exists, load the model from it
            if os.path.isfile(self._model_file):
                self.load()

    def log(self, log_str, level=logging.info):
        """ Logs the given string to the log file, iff persist enabled.
            Also outputs the string to the console, iff console_out enabled.
        """
        if self._persist:
            g_logger.info(log_str)

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
        # try:
        eval(self._load_func)
        self.log('Loaded model from: ' + self._model_file)
        # except Exception as e:
        #     err_str = 'Error loading model: ' + str(e)
        #     self.log(err_str, level=logging.error)
        #     raise Exception(err_str)


class DataFrom(Dataset):
    """ A set of inputs & targets (i.e. labels & features) as torch.tensors,
        populated from the given CSV file and normalized if specified.
        self.inputs = torch.FloatTensor, with a gradient for torch.optim.
        self.targets = torch.FloatTensors, converted from str->float if needed.
        Assumes CSV file w/no header with format: label, feat_1, ... , feat_n
    """
    def __init__(self, csvfile, normalize=True):
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


class Logger(logging.Logger):
    """ An extension of Python's logging.Logger. Implements log file rotation
        and optional console output.
    """
    def __init__(self,
                 name,
                 fname,
                 console_output=False,
                 level=LOG_LEVEL,
                 num_files=LOG_FILES,
                 max_filesize=LOG_SIZE):
        """"""
        logging.Logger.__init__(self, name, level)

        # Define output formats
        log_fmt = '%(asctime)s - %(levelname)s: %(message)s'
        log_fmt = logging.Formatter(log_fmt + '')

        # Init log file rotation
        rotate_handler = logging.handlers.RotatingFileHandler(
            fname, max_filesize, num_files)
        rotate_handler.setLevel(level)
        rotate_handler.setFormatter(log_fmt)
        self.addHandler(rotate_handler)

        if console_output:
            console_fmt = '%(asctime)s - %(levelname)s:'
            console_fmt += '\n%(message)s'
            console_fmt = logging.Formatter(console_fmt)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(console_fmt)
            self.addHandler(console_handler)


class AttrIter(object):
    """ An iterable set of parrallel attribute stacks (one set per label).
        The iterator returns the next set of all attributes for every label
        as a dict: { ID_1: {attr_1: val, attr_n: val}, ... }
    """
    def __init__(self):
        self._results = {}  # {ID: {attr_1: [val_x], attr_n: [val_x]}, ..}
        
    def __str__(self):
        return str(self._results)

    def push(self, label, attr, value):
        """ Push the given results attribute to its stack for the given label. 
            Note: To keep stacks balanced, each push() for a labels's attribute
            should be followed by the rest of the attributes for that label.
        """
        try:
            node = self._results[label]
        except KeyError:
            self._results[label] = {}
            node = self._results[label]

        try:
            val_list = node[attr]
        except KeyError:
            node[attr] = []
            val_list = node[attr]

        val_list.append(value)

    def rm_empties(self, attr):
        """ Removes all sets/labels having no attributes of the given name.
        """
        self._results = {k: v for k, v in self._results.items() if v[attr]}

    def is_empty(self, attr):
        """ Returns True if the given attr exists for any label. Else False.
        """
        if {k: v for k, v in self._results.items() if v[attr]}:
            return False
        return True

    def keys(self):
        """ Returns a list of this object's keys/labels.
        """
        return [k for k in self._results.keys()]

    def __iter__(self):
        return self

    def __next__(self):
        """ Dequeues and returns a single set of attributes for each label.
        """
        attributes = {}
        good_results = False
        for label, attrs in self._results.items():
            attributes[label] = {}
            for k, v in attrs.items():
                try:
                    attributes[label][k] = v.pop()
                    good_results = True
                except IndexError:
                    attributes[label][k] = None
        
        if not good_results:
            raise StopIteration
        return attributes


class Queue:
    """ A Queue data structure.
        Exposes reset, push, pop, shove, cut, peek, top, is_empty, item_count, 
        and get_items.
        TODO: contains. use list.pop()
    """

    def __init__(self, maxsize=None):
        self.items = []             # Container
        self.maxsize = maxsize      # Max size of self.items

        # Validate maxsize and populate with defaultdata
        if maxsize and maxsize < 0:
                raise Exception('Invalid maxsize parameter.')

    def __len__(self):
        return len(self.items)

    def reset(self):
        """ Clear/reset queue items.
        """
        self.items = []

    def push(self, item):
        """ Adds an item to the back of queue.
        """
        if self.is_full():
            raise Exception('Attempted to push item to a full queue.')
        self.items.append(item)

    def shove(self, item):
        """ Adds an item to the back of queue. If queue already full, makes
            room for it by removing the item at front. If an item is removed
            in this way, is returned.
        """
        removed = None
        if self.is_full():
            removed = self.pop()
        self.items.append(item)
        return removed

    def cut(self, n, item):
        """ Inserts an item at the nth position from queue front. Existing
            items are moved back to accomodate it.
        """
        length = len(self)
        if length < n + 1:
            raise Exception('Attempted to cut at an out of bounds position.')
        if length >= self.maxsize:
            raise Exception('Attempted to cut into a full queue.')
        self.items = self.items[:n] + item + self.items[n:]  # TODO: Test cut

    def pop(self):
        """ Removes front item from queue and returns it.
        """
        if self.is_empty():
            raise Exception('Attempted to pop from an empty queue.')
        d = self.items[0]
        self.items = self.items[1:]
        return d

    def peek(self, n=0):  # TODO: Test safety of peek
        """ Returns the nth item from queue front. Leaves queue unchanged.
        """
        if len(self) < n + 1:
            raise Exception('Attempted to peek at an out of bounds position.')
        if self.is_empty():
            raise Exception('Attempted to peek at an empty.')
        return self.items[n]

    def is_empty(self):
        """ Returns true iff queue empty.
        """
        return len(self) == 0

    def is_full(self):
        """ Returns true iff queue at max capacity.
        """
        return self.maxsize and len(self) >= self.maxsize

    def get_items(self):
        """ Returns queue contents as a new list.
        """
        return [item for item in self.items]


################
# Function Lib #
################

def negate(x):
    """ Returns a negated version of the given string or digit.
    """
    # String negate
    if x.islower():
        return x.upper()
    elif x.isupper():
        return x.lower()

    # Numerical negate
    return (x * -1)
    