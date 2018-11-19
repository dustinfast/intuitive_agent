#!/usr/bin/env python
""" A non-binary classifier implemented as an artificial neural network (ANN)
    using the PyTorch library.
    
    Interface:
        ANN() is the main interface. It expects training/validation data as
        an instance object of type sharedlib.DataFrom(). 
        ANN persistence and output is handled by sharedlib.ModelHandler().
        After training, ann.classify(inputs) gets a classification from inputs.

    Usage: 
        See "__main__" for example usage.

    Dependencies:
        PyTorch (see https://pytorch.org)

"""
__author__ = "Dustin Fast"
__email__ = "dustin.fast@outlook.com"
__license__ = "GPLv3"


# Std lib
import logging

# Third-party
import torch
import torch.nn as nn

# Custom
from sharedlib import ModelHandler, DataFrom

MODEL_EXT = '.pt'  # ANN model file extension


class Classifier(nn.Module):
    """ An artificial neural network classifier with 3 fully connected 
        layers x, y, and z. Each layer is represented internally as a tensor.
    """
    def __init__(self, ID, dims, console_out, persist):
        """ Accepts the following parameters
            ID (str)                : This ANNs unique ID number
            dims (3-tuple)          : Node counts by layer (x, y, z)
            console_out (bool)      : Output log stmts to console flag
            persist (bool)          : Persit mode flag
            start_bias (float)      : Initial node bias
        """
        super(Classifier, self).__init__()
        self.ID = ID
        self.persist = persist
        self.inputs_sz = dims[0]
        self.outputs_sz = dims[2]
        self.outputs = torch.randn(dims[2])
        self.class_labels = None

        # Statistics containers
        self.train_epoch = 0
        self.train_loss = 0
        self.train_acc = 0
        self.train_acc_v = ''

        # Define loss function 
        self.loss_func = nn.MSELoss()

        # Define sequential layers: Linear->ReLU->Linear->ReLU->Linear->Sigmoid
        self.seq_layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.Sigmoid())

        # Init the load, save, log, and console output handler
        f_save = "torch.save(self.state_dict(), 'MODEL_FILE')"
        f_load = "self.load_state_dict(torch.load('MODEL_FILE'), strict=False)"
        self.model = ModelHandler(self, console_out, persist,
                                  model_ext=MODEL_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID + '\n'
        str_out += 'Layers = ' + str(self.seq_layers)
        return str_out

    def set_labels(self, classes):
        """ Sets the class labels from the given list of classes.
        """
        self.class_labels = classes

    def forward(self, t):
        """ Feeds the given tensor through the ANN, thus updating output layer.
        """
        self.outputs = self.seq_layers(t)
        return self.outputs

    def train(self, data, epochs=100, lr=.1, alpha=.9, stats_at=10):
        """ Trains the ANN according to the given parameters.
            data (iterable):    Training dataset
            epochs (int):       Learning iterations
            lr (float):         Learning rate
            alpha (float):      Learning gain/momentum
            stats_at (int):     Print status every stats_at epochs (0=never)
        """
        info_str = '{} epochs @ lr={}, alpha={}, file={}.'
        info_str = info_str.format(epochs, lr, alpha, data.fname)
        self.model.log('Training started: ' + info_str)

        # If no class labels set yet, assign the dataset's
        if not self.class_labels:
            self.set_labels(data.class_labels)
            self.model.log('Set class labels: ' + str(self.class_labels))
        else:
            self.model.log('Using existing labels: ' + str(self.class_labels))

        # Do training
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=alpha)        
        for self.train_epoch in range(epochs):
            for row in data:
                inputs, target = iter(row)
                optimizer.zero_grad()
                outputs = self(inputs)
                self.train_loss = self.loss_func(outputs, target)
                self.train_loss.backward()
                optimizer.step()
            if self.train_loss == 0.0: break

            # Output status as specified by stats_at
            if stats_at and self.train_epoch % stats_at == 0:
                self.model.log(
                    'Epoch {} - loss: {}'.format(
                        self.train_epoch, self.train_loss))

        self.model.log('Training Completed: ' + info_str + '\n')
        self.model.log('Last epcoh loss: {}'.format(self.train_loss))

        # If persisting, save the updated model
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
        self.train_acc = 0
        class_total = {c: 0 for c in self.class_labels}
        class_corr = {c: 0 for c in self.class_labels}

        with torch.no_grad():
            for row in data:
                inputs, target = iter(row)
                pred_class = self.classify(inputs)
                target_class = self.classify_outputs(target)
                
                # Aggregate accuracy
                total += 1
                class_total[target_class] += 1
                if target_class == pred_class:
                    corr += 1
                    class_corr[pred_class] += 1

                self.train_acc = corr / total

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
        
        self.train_acc_v = log_str
        self.model.log(log_str)

    def max_index(self, t):
        """ Returns the element index w/the highest value in the given tensor.
            Ex: If t = tensor(1, 2, 4, 2), returns 2
        """
        _, idx = torch.max(t, 0)
        return idx

    def classify_outputs(self, outputs):
        """ Returns the classification label of the given outputs tensor.
            I.e. outputs is the tensor produced by the output layer.
        """
        try:
            return self.class_labels[self.max_index(outputs)]
        except TypeError:
            self.model.log('Must set class labels first.', logging.error)

    def classify(self, inputs):
        """ Returns the classification label of the given inputs tensor.
            I.e. inputs is the initial input-layers input.
        """
        return self.classify_outputs(self(inputs))


if __name__ == '__main__':
    # Load the training and validation data sets
    trainfile = 'static/datasets/letter_train.csv'
    # trainfile = 'static/datasets/test/test3x2.csv'  # debug
    valfile = 'static/datasets/letter_val.csv'
    # valfile = 'static/datasets/test/test3x2.csv'  # debug
    train_data = DataFrom(trainfile, normalize=True)
    val_data = DataFrom(valfile, normalize=True)

    # Define the ANN's layer sizes from the training set
    x_sz = train_data.feature_count
    h_sz = int((train_data.feature_count + train_data.class_count) / 2)
    y_sz = train_data.class_count

    # Init the ann
    ann = Classifier('ann', (x_sz, h_sz, y_sz), console_out=True, persist=False)
    
    # Train the ann with the training set
    ann.train(train_data, epochs=500, lr=.001, alpha=.9, stats_at=10)
    
    # Set the classifier labels (only really necessary if loading pre-trained)
    ann.set_labels(train_data.class_labels)

    # Validate the ann against the validation set
    ann.validate(val_data, verbose=True)

    # Example of a classification request, given a feature vector for "B"
    inputs = torch.tensor([4,2,5,4,4,8,7,6,6,7,6,6,2,8,7,10], dtype=torch.float)
    # prediction = ann.classify(inputs)
    # print('Test Classification: ' + str(prediction))
