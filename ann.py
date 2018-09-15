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
        self.intput = torch.randn(dimens[0])
        self.output = torch.randn(dimens[2])
        # self.output = F.softmax(torch.randn(dimens[1]))  # softmax(y)
        
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
        self.output = F.softmax(y, 0)
        return self.output

    def train(self, data, epochs=1, lr=.01, alpha=.4, stats_at=1000):
        """ Trains the ANN according to the given parameters.
            data (DataLoader):  Training dataset, as a PyTorch DataLoader
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        # Optimization function
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=alpha)
        
        # Do training epochs
        for epoch in range(epochs):
            for feature in data:
                inputs, target = iter(feature)
                optimizer.zero_grad()
                print(inputs[0])
                pred = self(inputs)
                print('Target: ' + str(target))
                print('Pred T: ' + str(pred))
                print('Pred M: ' + str(pred.max(0)))
                curr_loss = self.f_loss(pred.max(0)[0], target)
                curr_loss.backward()
                optimizer.step()

            # Output status according to stats_at
            if (stats_at and epoch % stats_at == 0):
                    print("Epoch {} - loss: {}".format(epoch, curr_loss))

        print('DEBUG: Training Complete')

    def save_model(self, filename):
        print(list(self.parameters()))

    def load_model(self, filename):
        raise NotImplementedError


class FeaturesFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features), populated from
        the given CSV file.
    """
    def __init__(self, csvfile, char_targets=False, norm=None):
        """ csvfile (str):      CSV file of form: label, feat_1, ... , feat_n

        """
        data = pd.read_csv(csvfile, header=None)    # Load CSV data w/pandas
        
        self.classes = list(data[0].unique())   # Unique feature classes
        self.targets = None                     # 2D Targets tensor
        self.inputs = None                      # 3D Inputs tensor
        self.norm = norm                        # Normalization range
        
        # Init targets, converting to strings if necessary
        targets = data.loc[:, :0] 
        if char_targets:
            targets = targets.apply(lambda x: self.map_str(x.iloc[0]), axis=1)

        # Init inputs, normalizing if necessary
        inputs = data.loc[:, 1:]
        if self.norm:
            inputs = self.normalize(inputs)

        # Set up tensors, with the input gradients tracked by PyTorch
        self.targets = V(torch.FloatTensor(targets), requires_grad=False)
        self.inputs = V(torch.FloatTensor(inputs.values), requires_grad=True)

        print('DEBUG: FeaturesFromCSV loaded...\n' + str(self))  # debug

    def __str__(self):
        str_out = 'Classes: ' + str(self.classes) + '\n'
        str_out += 'Row 1 Target: ' + str(self.targets[0]) + '\n'
        str_out += 'Row 1 Inputs: ' + str(self.inputs[0])
        return str_out

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]

    def dataloader(self, batch_sz=4, workers=2):
        return DataLoader(self, batch_size=batch_sz, num_workers=workers)

    def normalize(self, x):
        return (x - self.norm[0]) / (self.norm[1] - self.norm[0])

    def map_str(self, string):
        return self.classes.index(string)


if __name__ == '__main__':
    ann = ANN((16, 14, 3))
    # ann = ANN((16, 14, 26), f_activation=nn.ReLU)

    print('DEBUG: Layers:')
    print(str(ann))

    datafile = 'datasets/test3.data'
    # datafile = 'datasets/letter_train.data'
    data = FeaturesFromCSV(datafile, True, (0, 15)) 

    ann.train(data)
    exit(0)


    # ###############################################
    # Validation

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('Given: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' %
                                  classes[predicted[j]] for j in range(4)))

    if DO_WHOLE_SET:
        # ###################################################
        # Test against whole data set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        # Determine accuracy by class
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
