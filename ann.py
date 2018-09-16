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
        If some class types missing from training or validation set, errors.
        Expand ANN to allow an arbitrary number of hidden layers


    Author: Dustin Fast, Fall, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class DataFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features) for ANN(),
        populated from the given CSV file.
    """

    def __init__(self, csvfile, norm=None):
        """ csvfile (str):      CSV file of form: label, feat_1, ... , feat_n
            norm (2-tuple):     Normalization range, as (min, max). None OK.
        """
        data = pd.read_csv(csvfile, header=None)    # Load CSV data w/pandas

        self.classes = list(data[0].unique())       # Unique instance classes
        self.class_count = len(self.classes)        # Num unique classes
        self.feature_count = None                   # Num input features
        self.inputs = None                          # 3D Inputs tensor
        self.targets = None                         # 3D Targets tensor
        self.norm = norm                            # Normalization range

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
    """ An artificial neural network with fully connected layers x, y, and z,
        with each layer represented as a tensor.
        """
    def __init__(self, ID, dimens, f_activation=nn.Sigmoid, f_loss=nn.MSELoss):
        """ If ID, the model will 
            ID (int)        :           This ANNs unique ID
            dimens (3-tuple):           Node counts by layer (x, y, z)
            f_activation (nn.Layer):    Node activation function
            f_loss (nn.LossFunction):   Loss function
        """
        super(ANN, self).__init__()
        self.ID = ID
        
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

    def train(self, data, epochs=500, lr=.1, alpha=.4, stats_at=10, noise=None):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        # TODO: Noise param
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
        class_acc = {c: [0, 0] for c in data.classes}  # c: [correct, total]
        
        with torch.no_grad():
            for row in data:
                inputs, target = iter(row)
                outputs = self(inputs)
                target_class = data.class_from_node(target)
                pred_class = data.class_from_node(outputs)
                
                total += 1
                class_acc[pred_class][1] += 1
                if target_class == pred_class:
                    correct += 1
                    class_acc[pred_class][0] += 1

        print('Validaton Complete - Accuracy (%d %%):' % (100 * correct / total))

        if verbose:
            for c in ((k, v) for k, v in class_acc.items() if v[0]):
                print('Accuracy of %s : %2d %%' % (c[0], 100 * c[1][0] / c[1][1]))
                
    @staticmethod
    def get_filenames(self, filename, pt_ext=True):
        """ Returns a 2-tuple of a model & optimizer filename with .pt extension
        """
        opt_file = filename + '.opt'
        if pt_ext:
            filename += '.pt'
            opt_file += '.pt'

        return filename, opt_file

    def save_model(self, filename, pt_ext=True):
        """ Saves as a model of the ANN and its optimizer to the given file.
            If pt_ext, the filename is appended with '.pt'.
        """
        raise NotImplementedError
        fnames = get_filenames(filename, pt_ext=pt_ext)
        torch.save(self.state_dict(), fnames[0])
        torch.save(self.optimizer.state_dict(), fnames[1])

    def load_model(self, filename, pt_ext=True):
        """ Loads a model of the ANN and its optimizer from the given file.
            If pt_ext, the filename is appended with '.pt'.
        """
        raise NotImplementedError
        fnames = get_filenames(filename, pt_ext=pt_ext)
        self.load_state_dict(torch.load(fnames[0]))
        self.optimizer.load_state_dict(torch.load(fnames[1]))


if __name__ == '__main__':
    # The ID of this demonstrations ID
    ann_ID = 43770

    # Check for a model file of this ANN's name
    # if os.path.isfile(ANN.get_filenames(ann_ID)):

    trainfile = 'datasets/letter_train.data'
    # trainfile = 'datasets/test.data'
    valfile = 'datasets/letter_val.data'

    train_data = DataFromCSV(trainfile, (0, 15))
    val_data = DataFromCSV(valfile, (0, 15))

    print('Training set        : ' + trainfile)
    print('Validation set      : ' + valfile)

    # The ANN's layer sizes, based on the CSV file dimensions
    x_sz = train_data.feature_count
    h_sz = 14
    y_sz = train_data.class_count

    ann = ANN(ann_ID, (x_sz, h_sz, y_sz))
    print('Using ANN w/ dimens :\n' + str(ann))

    ann.train(train_data)
    ann.validate(val_data, verbose=True)
