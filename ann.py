#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using the PyTorch and 
    Pandas libraries.

    The ANN functions as a classifier, with the output classification denoted
    by the active (argmax(y)) output node.
    
    Dataset extraction from CSV and mapping of features and classes to each
    output node is facilitated by DataFromCSV().
    
    If persistent mode enabled, ANN state persists between exectutions via file
    PERSIST_PATH/ID.MODEL_EXT and status msgs logged to PERSIST_PATH/ID.LOG_EXT

    Dependencies:
        PyTorch

    Usage: 
        See __main__ for example usage.

    Conventions:
        x = Input layer (i.e. the set of input-layer nodes)
        h = Hidden layer
        y = Output layer
        t = A tensor

    # TODO: 
        Noise Params
        Fails if some class types missing between training and validation set.
        Expand ANN to allow an arbitrary number of hidden layers
        Implement option to use a PyTorch.utils.data.DataLoader as data source
        Ensure self.classes is persistent


    Author: Dustin Fast, 2018
"""

import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable as V

# import sys; sys.path.append('modules/') # add directory 'modules' to the current path

PERSIST_PATH = 'var/ann/'   # ANN model and log file path
MODEL_EXT = '.pt'           # ANN model file extension
LOG_EXT = '.log'            # ANN log file extension


class DataFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features) for ANN(),
        populated from the given CSV file.
    """
    def __init__(self, csvfile, norm_range=None):
        """ csvfile (str):        CSV file of form: label, feat_1, ... , feat_n
            norm_range (2-tuple): Normalization range, as (min, max). None OK.
        """
        self.classes = None                         # Unique instance labels
        self.class_count = None                     # Num unique labels
        self.feature_count = None                   # Num input features
        self.inputs = None                          # 3D Inputs tensor
        self.targets = None                         # 3D Targets tensor
        self.norm = norm_range                      # Normalization range
        self.fname = csvfile                        # CVS file name
        
        data = pd.read_csv(csvfile, header=None)    # csvfile -> pd.DataFrame

        # Populate class info
        self.classes = list(data[0].unique())
        self.class_count = len(self.classes)

        # Init inputs, normalizing as specified
        inputs = data.loc[:, 1:]
        if self.norm:
            inputs.apply(self.normalize)
        self.inputs = V(torch.FloatTensor(inputs.values), requires_grad=True)
        self.feature_count = self.inputs.size()[1]

        # Init targets
        targets = data.loc[:, :0]
        targets = targets.apply(lambda t: self._map_outnode(t.iloc[0]), axis=1)
        self.targets = targets

    def __str__(self):
        str_out = 'Classes: ' + str(self.classes) + '\n'
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
        tgt_width = len(self.classes)
        target = torch.tensor([0 for i in range(tgt_width)], dtype=torch.float)
        target[self.classes.index(label)] = 1
        return target

    def normalize(self, t):
        """ Returns a normalized representation of the given tensor. 
        """
        return (t - self.norm[0]) / (self.norm[1] - self.norm[0])


class ANN(nn.Module):
    """ An artificial neural network with 3 fully connected layers x, y, and z,
        with each layer represented as a tensor.
        """
    def __init__(self, ID, dims, f_act=nn.Sigmoid, f_loss=nn.MSELoss, **kwargs):
        """ ID (str)                    :   This ANNs unique ID number
            dims (3-tuple)              :   Node counts by layer (x, y, z)
            f_act (nn.Layer)            :   Node activation function
            f_loss (nn.LossFunction)    :   Node Loss function

            **kwargs:
                persist (bool)          :   Persit mode flag
                console_out (bool)      :   Output log stmts to console flag
        """
        super(ANN, self).__init__()
        self.ID = ID
        self.model_file = None
        self.logger = None
        self.persist = False
        self.console_out = False
        self.classes = None  # Set on self.train()
        
        # Layer defs
        self.x_sz = dims[0]
        self.y_sz = dims[2]
        self.x = torch.randn(dims[0])
        self.y = torch.randn(dims[2])
        
        # Layer activation functions
        self.f_x = nn.Linear(dims[0], dims[1], bias=True)
        self.f_h = nn.Linear(dims[1], dims[2], bias=True)
        self.f_y = nn.Linear(dims[2], dims[2], bias=True)

        # Node activation/loss functions
        self.f_act = f_act()
        self.f_loss = f_loss()

        # kwarg handlers...
        if kwargs.get('console_out'):
            self.console_out = True
            
        if kwargs.get('persist'):
            self.persist = True
            if not os.path.exists(PERSIST_PATH):
                os.mkdir(PERSIST_PATH)

            # Init logger and output init statment
            logging.basicConfig(filename=PERSIST_PATH + ID + LOG_EXT,
                                level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s: %(message)s')
            self._log('*** ANN initialized ***:\n' + str(self))

            # Init, and possibly load, model file
            self.model_file = PERSIST_PATH + ID + MODEL_EXT
            if os.path.isfile(self.model_file):
                self.load()
            
    def __str__(self):
        str_out = 'ID = ' + self.ID + '\n'
        str_out += 'x = ' + str(self.f_x) + '\n'
        str_out += 'h = ' + str(self.f_h) + '\n'
        str_out += 'z = ' + str(self.f_y)
        return str_out

    def _log(self, log_str, level=logging.info):
        """ Logs the given string to the ANN's log file, iff in persist mode.
            Also outputs the string to the console, iff in console_out mode.
        """
        if self.persist:
            level(log_str)

        if self.console_out:
            print(log_str)

    def _label_from_outputs(self, outputs):
        """ Given an outputs tensor, returns the mapped classification label.
        """
        _, idx = torch.max(outputs, 0)
        return self.classes[idx]

    def save(self):
        """ Saves a model of the ANN.
        """
        try:
            torch.save(self.state_dict(), self.model_file)
            self._log('Saved ANN model to: ' + self.model_file)
        except Exception as e:
            self._log('Error saving model: ' + str(e), level=logging.error)

    def load(self):
        """ Loads a model of the ANN.
        """
        try:
            self.load_state_dict(torch.load(self.model_file), strict=False)
            self._log('Loaded ANN model from: ' + self.model_file)
        except Exception as e:
            self._log('Error loading model: ' + str(e), level=logging.error)
            exit(0)

    def forward(self, t):
        """ Feeds the given tensor through the ANN, thus updating output layer.
        """
        self.x = self.f_act(self.f_x(t))            # Update input layer
        h = self.f_act(self.f_h(self.x))            # Update hidden layer
        self.y = self.f_act(self.f_y(h))            # Update output layer
        self.y = F.relu(self.y)                     # Rectify output layer
        return self.y

    def train(self, data, epochs=100, lr=.1, alpha=.3, stats_at=10, noise=None):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        info_str = '{} epochs @ lr={}, alpha={}, file={}.'
        info_str = info_str.format(epochs, lr, alpha, data.fname)
        self._log('Training started: ' + info_str)

        # Do training
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=alpha)        
        for epoch in range(epochs):
            for row in data:
                inputs, target = iter(row)
                optimizer.zero_grad()
                outputs = self(inputs)
                curr_loss = self.f_loss(outputs, target)
                curr_loss.backward()
                optimizer.step()

            # Output status as specified by stats_at
            if stats_at and epoch % stats_at == 0:
                self._log('Epoch {} - loss: {}'.format(epoch, curr_loss))

        # If no class labels yet, use the dataset's.
        if not self.classes:
            self.classes = data.classes
            self._log('Set self.classes:' + str(self.classes))
        else:
            self._log('Still using old self.classes:' + str(self.classes))

        self._log('Training Completed: ' + info_str + '\n')

        # Save the updated model, including the class labels, to the model file
        if self.persist:
            self.save()

    def validate(self, data, verbose=False):
        """ Validates the ANN against the given data set.
        """
        info_str = 'file={}.'.format(data.fname)
        self._log('Validation started: ' + info_str)

        total = 0
        corr = 0
        class_total = {c: 0 for c in self.classes}
        class_corr = {c: 0 for c in self.classes}

        with torch.no_grad():
            for row in data:
                inputs, target = iter(row)
                outputs = self(inputs)
                target_class = self._label_from_outputs(target)
                pred_class = self._label_from_outputs(outputs)
                
                total += 1
                class_total[target_class] += 1
                if target_class == pred_class:
                    corr += 1
                    class_corr[pred_class] += 1

        log_str = 'Validation Completed: Accuracy=%d%%\n' % (100 * corr / total)

        # If verbose, show detailed accuracy info
        if verbose:
            log_str += 'Correct: %d\n' % corr
            log_str += 'Total: %d\n' % total
            for c in self.classes:
                log_str += '%s : %d / %d ' % (c, class_corr[c], class_total[c])
                if class_total[c] > 0:
                    log_str += '(%d%%)' % (100 * class_corr[c] / class_total[c])
                else:
                    log_str += '(0%)'
                log_str += '\n'
        
        self._log(log_str)

    def classify(self, inputs):
        """ Returns the ANN's classification of the given input tensor.
        """
        return self._label_from_outputs(self(inputs))


if __name__ == '__main__':
    # Define and load training and validation sets
    trainfile = 'static/datasets/letter_train.data'
    # trainfile = 'static/datasets/test.data'  # debug
    valfile = 'static/datasets/letter_val.data'
    # valfile = 'static/datasets/test.data'  # debug
    train_data = DataFromCSV(trainfile, (0, 15))
    val_data = DataFromCSV(valfile, (0, 15))

    # The ANN's layer sizes. In/out sz is based on the dataset's dimennsions
    x_sz = train_data.feature_count
    h_sz = 14
    y_sz = train_data.class_count
    ann_dimens = (x_sz, h_sz, y_sz)

    # Init, train, and subsequently validate the ANN
    ann = ANN('ann_1_2', ann_dimens, persist=True, console_out=True)
    ann.train(train_data, epochs=1700, lr=.1, alpha=.2, stats_at=50, noise=None)
    ann.validate(val_data, verbose=True)

    # Example of a classification request, given a feature vector for "b"
    b_inputs = torch.tensor(
        [4, 2, 5, 4, 4, 8, 7, 6, 6, 7, 6, 6, 2, 8, 7, 10], dtype=torch.float)
    prediction = ann.classify(b_inputs)
    print('Test Classification: ' + prediction)  # Should print "B"
