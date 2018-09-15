#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using PyTorch.

    Author: Dustin Fast, Fall, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ANN(nn.Module):
    """ An abstraction of an artificial neural network with 3 fully connected
        layers: input layer x, hidden layer h, and output layer y.
        # TODO: n hidden layers
    """
    def __init__(self, dimens, f_activation=nn.Sigmoid, f_loss=nn.MSELoss):
        """ dimens (3-tuple):           Denotes node counts for layers x, y, z
            f_activation (nn.Layer):    Node activation function
            f_loss (nn.LossFunction):   Loss function
        """
        super(ANN, self).__init__()
        self.input_len = dimens[0]
        self.output_len = dimens[2]
        self.output = F.softmax(torch.randn(dimens[1]))  # softmax(y)
        
        # Activaton and loss functions
        self.f_act = f_activation()
        self.f_loss = f_loss()

        # Layer summation functions
        self.f_x = nn.Linear(dimens[0], dimens[1], bias=True)
        self.f_h = nn.Linear(dimens[1], dimens[2], bias=True)
        self.f_y = nn.Linear(dimens[2], dimens[2], bias=True)

    def __str__(self):
        """ Returns a string representation of the ANN.
        """
        str_out = str(ann.f_x) + '\n'
        str_out += str(ann.f_h) + '\n'
        str_out += str(ann.f_y)
        return str_out

    def forward(self, t):
        """ Feeds the given tensor through the ANN, updating self.output.
        """
        # TODO: Validate size
        x = self.f_act(self.f_x(t))     # Update input layer
        h = self.f_act(self.f_h(x))     # Update hidden layer
        y = self.f_act(self.f_y(h))     # Update output layer
        self.output = F.softmax(y)      # "Classify" output layer
        return y

    def train(self, data, epochs=1000, lr=.01, alpha=.4, stats_at=100):
        """ Trains the ANN according to the given parameters.
            data:               Training data, as a Dataloader or csv file name
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        # Load dataset from file, if needed
        if type(data) is str:
            features = Dataset(data)

        # Optimization function
        optimizer = torch.optim.SGD(ann.parameters(), lr=lr, momentum=alpha)

        # Do training epochs
        for epoch in range(epochs):
            for feature in features:
                inputs, label = iter(feature)
                inputs = V(torch.FloatTensor([inputs]), requires_grad=True)
                label = V(torch.FloatTensor([label]), requires_grad=False)
                optimizer.zero_grad()
                pred_label = self(inputs)
                curr_loss = self.loss(pred_label, label)
                curr_loss.backward()
                optimizer.step()
            if (stats_at and epoch % stats_at == 0):
                print("Epoch {} - loss: {}".format(epoch, curr_loss))

    def save_model(self, filename):
        print(list(self.parameters()))

    def load_model(self, filename):
        raise NotImplementedError


class Featureset(Dataset):
    """ A set of label to feature-vector associations, populated from CSV file.
    """
    def __init__(self, csv_file, header=None):
        """ csv_file (string):  CSV file of form: "label, feat_1, ... , feat_n"
        """
        self.featureset = pd.read_csv(csv_file, header=header)

        # debug
        print('Featureset loaded - First row  = ')
        print(self[0]['label'], self[0]['features'])

    def __len__(self):
        return len(self.featureset)

    def __getitem__(self, idx):
        label = self.featureset.iloc[idx, 0]
        features = self.featureset.iloc[idx, 1:].values
        features = features.astype('float').reshape(1, -1)
        feature = {'label': label, 'features': features}
        return feature


ann = ANN((16, 14, 26))
# ann = ANN((16, 14, 26), f_activation=nn.ReLU)

print('Layers:')
print(str(ann))

print('\nLoading Featureset...')
datafile = 'datasets/test.data'
# datafile = 'datasets/letter_train.data'
data = Featureset(datafile)  

