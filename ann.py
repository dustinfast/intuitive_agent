#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using PyTorch.

    The ANN functions as a classifier, with the output classification denoted
    by the active (argmax(y)) output node.
    
    Dataset extraction from CSV and mapping of features and classes to each
    output node is facilitated by DataFromCSV().
    
    If persistent mode enabled, ANN state is saved and status is loggged to 
    "ID.pt" and "ID.pt.log" on completion of each training/validation.

    Usage: See __main__ for example usage.

    Conventions:
        x - Input layer (i.e. the set of input-layer nodes)
        h - Hidden layer
        y - Output layer
        t - Some arbitrary tensor

    # TODO: 
        Noise Params
        Fails if some class types missing between  training and validation set.
        Expand ANN to allow an arbitrary number of hidden layers
        Implement use of PyTorch.utils.data.DataLoader


    Author: Dustin Fast, 2018
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset
import pandas as pd


PERSIST_PATH = 'var/ann/'   # ANN model and log file path
MODEL_EXT = '.pt'           # ANN model file extension
LOG_EXT = '.log'            # ANN log file extension


class DataFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features) for ANN(),
        populated from the given CSV file.
    """

    def __init__(self, csvfile, norm=None):
        """ csvfile (str):      CSV file of form: label, feat_1, ... , feat_n
            norm (2-tuple):     Normalization range, as (min, max). None OK.
        """
        self.classes = None                         # Unique instance labels
        self.class_count = None                     # Num unique labels
        self.feature_count = None                   # Num input features
        self.inputs = None                          # 3D Inputs tensor
        self.targets = None                         # 3D Targets tensor
        self.norm = norm                            # Normalization range
        
        data = pd.read_csv(csvfile, header=None)    # Load CSV data w/pandas

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
        targets = targets.apply(
            lambda t: self._class_to_node(t.iloc[0]), axis=1)
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

    def _class_to_node(self, label):
        """ Given a class label, returns zeroed tensor with tensor[label] = 1.
            Facilitates mapping each class to its corresponding output node
        """
        tgt_width = len(self.classes)
        target = torch.tensor([0 for i in range(tgt_width)], dtype=torch.float)
        target[self.classes.index(label)] = 1
        return target

    def class_from_node(self, t):
        """ Given an output level tensor, returns the mapped classification.
        """
        _, idx = torch.max(t, 0)
        return self.classes[idx]

    def dataloader(self, batch_sz=4, workers=2):
        """ Returns a torch.utils.Data.DataLoader representation of the set.
        """
        # TODO: return DataLoader(self, batch_size=batch_sz, num_workers=workers)
        raise NotImplementedError

    def normalize(self, t):
        """ Returns a normalized representation of the given tensor. 
        """
        return (t - self.norm[0]) / (self.norm[1] - self.norm[0])


class ANN(nn.Module):
    """ An artificial neural network with 3 fully connected layers x, y, and z,
        with each layer represented as a tensor.
        """
    def __init__(self, ID, dims, f_act=nn.Sigmoid, f_loss=nn.MSELoss, persist=False):
        """ ID (int)                    :   This ANNs unique ID number
            dims (3-tuple)              :   Node counts by layer (x, y, z)
            f_act (nn.Layer)            :   Node activation function
            f_loss (nn.LossFunction)    :   Loss function
            persist (bool)              :   Persit mode flag
        """
        super(ANN, self).__init__()
        self.ID = ID
        
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

        # Init persist mode as needed
        if persist:
            if not os.path.exists(PERSIST_PATH):
                os.mkdir(PERSIST_PATH)  # Ensure output folder exists
            self.log_file = PERSIST_PATH + str(ID) + LOG_EXT
            self.model_file = PERSIST_PATH + str(ID) + MODEL_EXT
            self.load()

    def __str__(self):
        str_out = 'ID = ' + str(self.ID) + '\n'
        str_out += 'x = ' + str(self.f_x) + '\n'
        str_out += 'h = ' + str(self.f_h) + '\n'
        str_out += 'z = ' + str(self.f_y)
        return str_out

    def _log(self, log_string):
        """ Logs the given string to the ANN's log file.
        """
        raise NotImplementedError

    def forward(self, t):
        """ Feeds the given tensor through the ANN, thus updating output layer.
        """
        # Apply node activation functions to each node of each layer
        # (Note rectification function applied to self.y)
        self.x = self.f_act(self.f_x(t))            # Update input layer
        h = self.f_act(self.f_h(self.x))            # Update hidden layer
        self.y = F.relu(self.f_act(self.f_y(h)))    # Update output layer
        return self.y

    def train(self, data, epochs=10000, lr=.01, alpha=.2, stats_at=10, noise=None):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        # Status info
        train_str = '{} epochs @ lr={}, alpha={}...'.format(epochs, lr, alpha)
        if stats_at:
            print('Training w/ ' + train_str)

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
            if stats_at:
                if epoch % stats_at == 0:
                    print('Epoch {} - loss: {}'.format(epoch, curr_loss))
                elif epoch == epochs:
                    print('Training Complete w/' + train_str)

    def validate(self, data, verbose=False):
        """ Validates the ANN against the given data set.
        """
        total = 0
        correct = 0
        class_total = {c: 0 for c in data.classes}
        class_correct = {c: 0 for c in data.classes}

        with torch.no_grad():
            for row in data:
                inputs, target = iter(row)
                outputs = self(inputs)
                target_class = data.class_from_node(target)
                pred_class = data.class_from_node(outputs)
                
                total += 1
                class_total[target_class] += 1
                if target_class == pred_class:
                    correct += 1
                    class_correct[pred_class] += 1

        print('Validaton Complete - Accuracy: (%d %%)' % (100 * correct / total))

        if verbose:
            print('Correct : %d' % correct)
            print('Total   : %d' % total)
            print('By class:')
            for c in data.classes:
                print('%s : %d / %d' % (c, class_correct[c], class_total[c]), end=' ')
                if class_total[c] > 0:
                    print('(%2d %%)' % (100 * class_correct[c] / class_total[c]))
                else:
                    print('(0%)')
                
    def save(self):
        """ Saves a model of the ANN.
        """
        pass
        torch.save(self.state_dict(), self.model_file)
        # torch.save(self.optimizer.state_dict(), fnames[1])

    def load(self):
        """ Loads a model of the ANN.
        """
        try:
            self.load_state_dict(torch.load(self.model_file))
            # self.optimizer.load_state_dict(torch.load(self.opt_file))
        except Exception as e:
            print('WARN: No model file currently exists for ' + str(self.ID))


if __name__ == '__main__':
    # The ID of this demonstration's ANN
    ann_ID = 43770

    # Training and validation sets
    trainfile = 'datasets/letter_train.data'
    trainfile = 'static/datasets/test3.data'  # debug
    valfile = 'static/datasets/letter_val.data'
    valfile = 'static/datasets/test3.data'  # debug

    train_data = DataFromCSV(trainfile, (0, 15))
    val_data = DataFromCSV(valfile, (0, 15))

    print('Training set        : ' + trainfile)
    print('Validation set      : ' + valfile)

    # The ANN's layer sizes, based on the dataset
    x_sz = train_data.feature_count
    # h_sz = train_data.class_count - train_data.feature_count
    h_sz = 14
    y_sz = train_data.class_count

    # Init, train, and subsequently validate the ANN
    ann = ANN(ann_ID, (x_sz, h_sz, y_sz), persist=True)
    print('Using ANN:\n' + str(ann))
    ann.train(train_data)
    ann.validate(val_data, verbose=True)

    # TODO: Example of a single classification request
