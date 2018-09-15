#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using PyTorch.

    The ANN functions as a classifier, with the output classification denoted
    by the active (y=1) output node.
    Data set extraction from CSV and mapping of classes to each output node is
    facilitated by DataFromCSV()

    Author: Dustin Fast, Fall, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ANN(nn.Module):
    """ An artificial neural network with 3 fully connected layers: thne 
        input layer X, hidden layer H, and output layer Y.
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
        self.X = torch.randn(dimens[0])   # TODO: set in forwards
        self.H = torch.randn(dimens[1])   # TODO: set in forwards
        self.Y = torch.randn(dimens[2])
        
        # Activaton and loss functions
        self.f_act = f_activation()
        self.f_loss = f_loss()

        # Layer summations
        self.f_x = nn.Linear(dimens[0], dimens[1], bias=True)
        self.f_h = nn.Linear(dimens[1], dimens[2], bias=True)
        self.f_y = nn.Linear(dimens[2], dimens[2], bias=True)

    def __str__(self):
        str_out = str(ann.f_x) + '\n'
        str_out += str(ann.f_h) + '\n'
        str_out += str(ann.f_y)
        return str_out

    def forward(self, t):
        """ Feeds the given tensor through the ANN, updating self.Y.
        """
        self.X = self.f_act(self.f_x(t))                # Update input layer
        self.H = self.f_act(self.f_h(self.X))           # Update hidden layer
        self.Y = F.relu(self.f_act(self.f_y(self.H)))   # Update output layer
        # self.Y = F.relu(y)  # TODO: Test w no relu
        # TODO: Test w/ self.Y = F.softmax(y, 0)
        return self.Y

    def train(self, data, epochs=1000, lr=.1, alpha=.4, stats_at=5):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        print("Training {} epochs @ lr={}, alpha={}...".format(epochs, lr, alpha))
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=alpha)        

        # Do training epochs
        for epoch in range(epochs):
            for feature in data:
                inputs, target = iter(feature)
                # print('Train - I: ' + str(inputs))  # debug
                # print('Train - T: ' + str(target))  # debug
                optimizer.zero_grad()
                outputs = self(inputs)
                # print('Train - P: ' + str(outputs))  # debug
                curr_loss = self.f_loss(outputs, target)
                curr_loss.backward()
                optimizer.step()

            # Output status, per stats_at param
            if (stats_at and epoch % stats_at == 0):
                    print("Epoch {} - loss: {}".format(epoch, curr_loss))

        print('DEBUG: Training Complete')  # debug

    def validate(self, data):
        """ Validates the ANN against the given data set.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for feature in data:
                inputs, targets = iter(feature)
                # print('Val - I: ' + str(inputs))  # debug
                # print('Val - T: ' + str(target))  # debug
                outputs = self(inputs)
                # print('Val - P: ' + str(outputs))  # debug
                _, predicted = torch.max(outputs, 0)
                total += targets.size(0)
                if outputs.data == targets:
                    correct += 1

        print('Validaton Complete. Accuracy: %d %%' % (100 * correct / total))

    def save_model(self, filename):
        """ Saves a model of the ANN to the given file.
        """
        print(list(self.parameters()))

    def load_model(self, filename):
        """ Loads the ANN model from the given file.
        """
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
        targets = targets.apply(lambda x: self._map_to_node(x.iloc[0]), axis=1)
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
        target = torch.tensor([0 for x in range(tgt_width)], dtype=torch.float)
        target[self.classes.index(label)] = 1
        # print(target)  # debug
        return target

    def _map_from_node(self, idx):
        """ Given an output node index, returns the mapped classification.
        """
        of_class = self.classes[idx]
        print(of_class)  # debug
        return of_class

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def dataloader(self, batch_sz=4, workers=2):
        return DataLoader(self, batch_size=batch_sz, num_workers=workers)

    def normalize(self, x):
        """ Returns a normalized representation of the given tensor. 
        """
        return (x - self.norm[0]) / (self.norm[1] - self.norm[0])


if __name__ == '__main__':
    trainfile = 'datasets/letter_train.data'
    # trainfile = 'datasets/test.data'
    train_data = DataFromCSV(trainfile, (0, 15)) 
    valfile = 'datasets/test.data'
    val_data = DataFromCSV(valfile, (0, 15))

    # TODO: in_nodes = len(train_data.inputs)
    print(len(train_data.inputs))
    out_nodes = len(train_data.classes)

    ann = ANN((16, 14, out_nodes))

    print('DEBUG: Layers:')
    print(str(ann))

    ann.train(train_data)  # TODO: pass csv or model file

    # Test ff
    # print(ann(V(torch.Tensor([[2,14,12,8,5,9,10,4,3,5,10,7,10,12,2,6]]))))  # W
    # ann.validate(val_data)    # TODO: pass csv or model file
    exit(0)
