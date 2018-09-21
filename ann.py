#!/usr/bin/env python
""" An Artificial Neural Network (ANN) implemented using the PyTorch and 
    Pandas libraries.

    The ANN functions as a classifier, with the output classification denoted
    by the active (argmax(y)) output node.
    
    Dataset extraction from CSV and mapping of features and classes to each
    output node is facilitated by DataFromCSV().
    
    If persistent mode enabled, ANN state persists between exectutions via file
    PERSIST_PATH/ID.MODEL_EXT and status msgs logged to PERSIST_PATH/ID.LOG_EXT

    Structure:
        ANN is the main interface, with DataFromCSV and Model as helpers.

    Usage: 
        See __main__ for example usage.

    Conventions:
        x = Input layer (i.e. the set of input-layer nodes)
        h = Hidden layer
        y = Output layer
        t = A tensor

    # TODO: 
        Noise Params
        Move DataFromCSV to classlib
        Remove norm param from DataFromCSV
        Test if some class types missing between training and validation set
        Ability to use PyTorch.utils.data.DataLoader as data source


    Author: Dustin Fast, 2018
"""

# Imports
import logging

import torch
import torch.nn as nn

from classlib import Model, DataFromCSV


# Constants
MODEL_EXT = '.pt'           # ANN model file extension


class ANN(nn.Module):
    """ An artificial neural network with 3 fully connected layers x, y, and z,
        with each layer represented as a tensor.

        Layer/Node properties:
            Each layer is summed according to torch.nn.Linear()
            Each Node is activated according torch.nn.Sigmoid()
            Error is computed via torch.nn.MSELoss()
        """
    def __init__(self, ID, dims, console_out, persist, start_bias=-1):
        """ Accepts the following parameters
            ID (str)                : This ANNs unique ID number
            dims (3-tuple)          : Node counts by layer (x, y, z)
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
            start_bias (float)      : Initial node bias
        """
        super(ANN, self).__init__()
        self.ID = ID
        self.persist = persist
        self.class_labels = None
        
        # Layer defs
        self.x_sz = dims[0]
        self.y_sz = dims[2]
        self.x = torch.randn(dims[0])
        self.y = torch.randn(dims[2])
        
        # Layer activation functions
        self.f_x = nn.Linear(dims[0], dims[1], bias=True)
        self.f_h = nn.Linear(dims[1], dims[2], bias=True)
        self.f_y = nn.Linear(dims[2], dims[2], bias=True)

        # # Node activation/loss functions
        # self.f_act = nn.Sigmoid()
        self.f_loss = nn.MSELoss()

        # Set initial node bias
        # self.f_x.bias.data.fill_(start_bias)
        # self.f_h.bias.data.fill_(start_bias)
        # self.f_y.bias.data.fill_(start_bias)

        self.test = nn.Sequential(nn.Linear(dims[0], dims[1]),
                                            nn.ReLU(),
                                            nn.Linear(dims[1], dims[2]),
                                            nn.Sigmoid())

        # Init the Model obj, which handles load, save, log, and console output
        save_func = "torch.save(self.state_dict(), 'MODEL_FILE')"
        load_func = "self.load_state_dict(torch.load('MODEL_FILE'), strict=False)"
        self.model = Model(self,
                           console_out,
                           persist,
                           model_ext=MODEL_EXT,
                           save_func=save_func,
                           load_func=load_func)

    def __str__(self):
        str_out = 'ID = ' + self.ID + '\n'
        str_out += 'x = ' + str(self.f_x) + '\n'
        str_out += 'h = ' + str(self.f_h) + '\n'
        str_out += 'z = ' + str(self.f_y)
        return str_out

    def _label_from_outputs(self, outputs):
        """ Given an outputs tensor, returns the mapped classification label.
        """
        try:
            _, idx = torch.max(outputs, 0)
            return self.class_labels[idx]
        except TypeError:
            self.model.log('Must set class labels first.', logging.error)

    def set_labels(self, classes):
        """ Sets the class labels from the given list of classes.
        """
        self.class_labels = classes

    def forward(self, t):
        """ Feeds the given tensor through the ANN, thus updating output layer.
        """
        # self.x = self.f_act(self.f_x(t))            # Update input layer
        # h = self.f_act(self.f_h(self.x))            # Update hidden layer
        # self.y = self.f_act(self.f_y(h))            # Update output layer
        # # self.y = F.relu(self.y)
        self.y = self.test(t)
        return self.y

    def train(self, data, epochs=100, lr=.1, alpha=.9, stats_at=10, noise=None):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        info_str = '{} epochs @ lr={}, alpha={}, file={}.'
        info_str = info_str.format(epochs, lr, alpha, data.fname)
        self.model.log('Training started: ' + info_str)

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
                self.model.log('Epoch {} - loss: {}'.format(epoch, curr_loss))

        # If no class labels yet, use the dataset's.
        if not self.class_labels:
            self.set_labels(data.class_labels)
            self.model.log('Set class labels: ' + str(self.class_labels))
        else:
            self.model.log('Using existing labels: ' + str(self.class_labels))

        self.model.log('Training Completed: ' + info_str + '\n')
        self.model.log('Last epcoh loss: {}'.format(curr_loss))

        # Save the updated model, including the class labels, to the model file
        if self.persist:
            self.model.save()

    def validate(self, data, verbose=False):
        """ Validates the ANN against the given data set.
        """
        if not self.class_labels:
            self.model.log('No class labels defined.', logging.error)
            return

        info_str = 'file={}.'.format(data.fname)
        self.model.log('Validation started: ' + info_str)

        total = 0
        corr = 0
        class_total = {c: 0 for c in self.class_labels}
        class_corr = {c: 0 for c in self.class_labels}

        with torch.no_grad():
            for row in data:
                inputs, target = iter(row)
                pred_class = self.classify(inputs)
                target_class = self._label_from_outputs(target)
                
                total += 1
                class_total[target_class] += 1
                if target_class == pred_class:
                    corr += 1
                    class_corr[pred_class] += 1

        log_str = 'Validation Completed: Accuracy=%d%%\n' % (100 * corr / total)

        # If verbose, output detailed accuracy info
        if verbose:
            log_str += 'Correct: %d\n' % corr
            log_str += 'Total: %d\n' % total
            for c in self.class_labels:
                log_str += '%s : %d / %d ' % (c, class_corr[c], class_total[c])
                if class_total[c] > 0:
                    log_str += '(%d%%)' % (100 * class_corr[c] / class_total[c])
                else:
                    log_str += '(0%)'
                log_str += '\n'
        
        self.model.log(log_str)

    def classify(self, inputs):
        """ Returns the ANN's classification of the given input tensor.
        """
        return self._label_from_outputs(self(inputs))


if __name__ == '__main__':
    # Load the training and validation data sets
    trainfile = 'static/datasets/letter_train.data'
    trainfile = 'static/datasets/test.data'  # debug
    valfile = 'static/datasets/letter_val.data'
    valfile = 'static/datasets/test.data'  # debug
    train_data = DataFromCSV(trainfile, normalize=True)
    val_data = DataFromCSV(valfile, normalize=True)

    # Define the ANN's layer sizes.
    x_sz = train_data.feature_count
    h_sz = 14
    y_sz = train_data.class_count

    # Init the ann
    ann = ANN('ann_test', (x_sz, h_sz, y_sz), console_out=True, persist=False)

    # Train the ann with the training set
    ann.train(train_data, epochs=1000, lr=.01, alpha=.9, stats_at=10, noise=None)
    
    # Set the classifier labels (only really necessary if loading pre-trained)
    ann.set_labels(train_data.class_labels)

    # Validate the ann against the validation set
    ann.validate(val_data, verbose=True)

    # Example of a classification request, given a feature vector for "B"
    inputs = torch.tensor([4,2,5,4,4,8,7,6,6,7,6,6,2,8,7,10], dtype=torch.float)
    prediction = ann.classify(inputs)
    print('Test Classification: ' + str(prediction))
