#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using PyTorch.

    The ANN functions as a classifier, with the output classification denoted
    by the active (y=1) output node.
    Data set extraction from CSV and mapping of classes to each output node is
    facilitated by DataFromCSV().

    Usage: See __main__ for example usage.

    Conventions:
        x - Input layer (i.e. the set of input-layer nodes)
        h - Hidden layer
        y - Output layer
        t - Some arbitrary tensor

    TODO: 
        Expand ANN to allow an arbitrary number of hidden layers


    Author: Dustin Fast, Fall, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ANN(nn.Module):
    """ An artificial neural network with fully connected layers x, y, and z,
        with each layer represented as a tensor.
        """
    def __init__(self, dimens, f_activation=nn.Sigmoid, f_loss=nn.MSELoss):
        """ dimens (3-tuple):           Node counts by layer (x, y, z)
            f_activation (nn.Layer):    Node activation function
            f_loss (nn.LossFunction):   Loss function
        """
        super(ANN, self).__init__()

        # Layer defs
        self.x_sz = dimens[0]
        self.y_sz = dimens[2]
        self.x = torch.randn(dimens[0])
        self.y = torch.randn(dimens[2])
        
        # Layer activation functions
        self.f_x = nn.Linear(dimens[0], dimens[1], bias=True)
        self.f_h = nn.Linear(dimens[1], dimens[2], bias=True)
        self.f_y = nn.Linear(dimens[2], dimens[2], bias=True)

        # Node activation/loss functions
        self.f_act = f_activation()
        self.f_loss = f_loss()

    def __str__(self):
        str_out = str(ann.f_x) + '\n'
        str_out += str(ann.f_h) + '\n'
        str_out += str(ann.f_y)
        return str_out

    def forward(self, t):
        """ Feeds the given tensor through the ANN, thus updating output layer.
        """
        # Apply node activation functions to each node of each layer
        # Note: rectification function applied to self.y results
        self.x = self.f_act(self.f_x(t))            # Update input layer
        h = self.f_act(self.f_h(self.x))            # Update hidden layer
        self.y = F.relu(self.f_act(self.f_y(h)))    # Update output layer

        return self.y

    def train(self, data, epochs=100, lr=.1, alpha=.7, stats_at=10):
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
                # print('Train - I: ' + str(inputs))  # debug
                # print('Train - T: ' + str(target))  # debug
                optimizer.zero_grad()
                outputs = self(inputs)
                # print('Train - P: ' + str(outputs))  # debug
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
        correct = 0
        total = 0
        class_acc = {c[0]: (0, 0) for c in data.classes}  # {label: (correct, outof)}


        with torch.no_grad():
            for row in data:
                inputs, target = iter(data)
                print('Val - I: ' + str(inputs))  # debug
                print('Val - T: ' + str(target))  # debug
                outputs = self(inputs)
                print('Val - P: ' + str(outputs))  # debug

                
                # _, predicted = torch.max(outputs, 0)
                # total += targets.size(0)
                # if outputs.data == targets:
                #     correct += 1

        if verbose:
            p = (100 * correct / total)
            print('Validaton Complete - Accuracy (%d %%):' % p)
            print(class_acc)

    def save_model(self, filename):
        """ Saves a model of the ANN to the given file.
        """
        # TODO
        raise NotImplementedError

    def load_model(self, filename):
        """ Loads the ANN model from the given file.
        """
        # TODO
        raise NotImplementedError


class DataFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features) for ANN(),
        populated from the given CSV file.
    """
    def __init__(self, csvfile, norm=None):
        """ csvfile (str):      CSV file of form: label, feat_1, ... , feat_n
            norm (2-tuple):     Input normalization range, as (min, max)
        """
        data = pd.read_csv(csvfile, header=None)    # Load CSV data w/pandas
        
        self.classes = list(data[0].unique())       # Unique feature classes
        self.inputs = None                          # 3D Inputs tensor
        self.targets = None                         # 3D Targets tensor
        self.norm = norm                            # Normalization range
        
        # Init inputs, normalizing as specified
        inputs = data.loc[:, 1:]
        if self.norm:
            inputs = self.normalize(inputs)  # TODO: inputs.apply(self.normalize)
        self.inputs = V(torch.FloatTensor(inputs.values), requires_grad=True)

        # Init targets
        targets = data.loc[:, :0] 
        targets = targets.apply(lambda t: self._map_to_node(t.iloc[0]), axis=1)
        self.targets = targets

        # print('DEBUG: DataFromCSV loaded...\n' + str(self))  # debug

    def __str__(self):
        str_out = 'Classes: ' + str(self.classes) + '\n'
        str_out += 'Row 1 Target: ' + str(self.targets[0]) + '\n'
        str_out += 'Row 1 Inputs: ' + str(self.inputs[0])
        return str_out

    def _map_to_node(self, label):
        """ Given a class label, returns zeroed tensor with tensor[label] = 1.
            Facilitates mapping each class to its corresponding output node
        """
        tgt_width = len(self.classes)
        target = torch.tensor([0 for i in range(tgt_width)], dtype=torch.float)
        target[self.classes.index(label)] = 1
        # print(target)  # debug
        return target

    def _map_from_node(self, idx):
        """ Given an output level tensor, returns the mapped classification.
        """
        # TODO: _, predicted=torch.max(outputs, 0)
        of_class = self.classes[idx]
        print(of_class)  # debug
        return of_class

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def dataloader(self, batch_sz=4, workers=2):
        return DataLoader(self, batch_size=batch_sz, num_workers=workers)

    def normalize(self, t):
        """ Returns a normalized representation of the given tensor. 
        """
        return (t - self.norm[0]) / (self.norm[1] - self.norm[0])


if __name__ == '__main__':
    # TODO: Usage example w/ data = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)]

    trainfile = 'datasets/letter_train.data'
    # trainfile = 'datasets/test.data'
    train_data = DataFromCSV(trainfile, (0, 15)) 
    valfile = 'datasets/test3.data'
    val_data = DataFromCSV(valfile, (0, 15))

    print('Using training set  : ' + trainfile)
    print('Using validation set: ' + valfile)

    # TODO: in_nodes = len(train_data.inputs)
    print(len(train_data.inputs))
    out_nodes = len(train_data.classes)

    ann = ANN((16, 14, out_nodes))
    print('Using ANN w/layers:   ' + str(ann))

    ann.train(train_data)

    print('Test row results:')
    print(ann(V(torch.Tensor([[2, 14, 12, 8, 5, 9, 10, 4, 3, 5, 10, 7, 10, 12, 2, 6]]))))  # W
    ann.validate(val_data, True)
    exit(0)
