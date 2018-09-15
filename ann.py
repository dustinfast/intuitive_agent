#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using PyTorch.

    Author: Dustin Fast, Fall, 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim
import pandas as pd


class ANN(nn.Module):
    """ An abstraction of an artificial neural network with 3 fully connected
        layers: input layer x, hidden layer h, and output layer y.
        # TODO: N hidden layers
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

    def forward(self, tensor):
        """ Feeds the given tensor through the ANN, updating self.output.
        """
        # TODO: Validate size
        x = self.f_act(self.f_x(t))     # Update input layer
        h = self.f_act(self.f_h(x))     # Update hidden layer
        y = self.f_act(self.f_y(h))     # Update output layer
        self.output = F.softmax(y)      # "Classify" output layer
        return y

    def do_train(self, data, epochs=1000, lr=.01, alpha=.4, status_at=100):
        """ Trains the ANN according to the given parameters.
            data:   Training data, of form [ label [ inputs ] ] 
        """
        # Optimization function
        optimizer = optim.SGD(ann.parameters(), lr=lr, momentum=alpha)

        # Do training epochs
        for epoch in range(epochs):
            for row in data:
                inputs, label = iter(row)
                inputs = V(torch.FloatTensor([inputs]), requires_grad=True)
                label = V(torch.FloatTensor([label]), requires_grad=False)
                optimizer.zero_grad()
                pred_label = self(inputs)
                curr_loss = self.loss(pred_label, label)
                curr_loss.backward()
                optimizer.step()
            if (status_at and epoch % status_at == 0):
                print("Epoch {} - loss: {}".format(epoch, curr_loss))

    def save_model(self, filename):
        print(list(self.parameters()))

    def load_model(self, filename):
        raise NotImplementedError


class CSVFeatureSet(Dataset):
    """A set of label to feature-vector associations, populated from CSV file.
    """
    def __init__(self, csv_file):
        """ csv_file (string): 
        """
        self.features = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

ann = ANN((16, 14, 26))
# ann = ANN((16, 14, 26), f_activation=nn.ReLU)

print('Layers:\n')
print(str(ann))

print('\nLoading data...')
# Load data from file
datafile = 'datasets/letter_train.data'
pd.read_csv(csv)

data = [ None, [] ]
try:
    with open(datafile, 'r') as f:
        for ln in f:
            if ln != '\n':
                print(ln)
except:
    print('Error reading data file: ' + datafile)
    exit(0)

print('\nDone. Data size: ' + 

data = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18)]

for epoch in range(1000):
    for i, data2 in enumerate(data):
        inputs, label = iter(data2)
        inputs = Variable(torch.FloatTensor([inputs]), requires_grad=True)
        label = Variable(torch.FloatTensor([label]), requires_grad=False)
        optimizer.zero_grad()  # zero the parameter gradients
        y_pred = ann(inputs)
        # print('Given: ' + str(label))
        # print('Pred: ' + str(y_pred))
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
    if (epoch % 20 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, loss))


exit(0)

    

