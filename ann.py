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
        Iniital bias = -1, initial bias wt = .5 
        Encapsulate main contents into ANN.from_csv()
        Noise Params
        Fails if some class types missing between training and validation set.
        Expand ANN to allow an arbitrary number of hidden layers
        Implement ability to use a PyTorch.utils.data.DataLoader as data source
        Ensure self.classes is persistent


    Author: Dustin Fast, 2018
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable as V
import pandas as pd

from classlib import Model

# Constants
MODEL_EXT = '.pt'           # ANN model file extension


class DataFromCSV(Dataset):
    """ A set of inputs and targets (i.e. labels and features) for ANN(),
        populated from the given CSV file and normalized if specified.
    """
    def __init__(self, csvfile, norm_range=None):
        """ Accepts the following parameters:
            csvfile (str):        CSV file of form: label, feat_1, ... , feat_n
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

        Layer/Node properties:
            Each layer is summed according to torch.nn.Linear()
            Each Node is activated according torch.nn.Sigmoid()
            Error is computed via torch.nn.MSELoss()
            Output layer is rectified with torch.nn.functional.relu()
        """
    def __init__(self, ID, dims, console_out, persist):
        """ Accepts the following parameters
            ID (str)                :   This ANNs unique ID number
            dims (3-tuple)          :   Node counts by layer (x, y, z)
            console_out (bool)      :   Output log stmts to console flag
            persist (bool)          :   Persit mode flag
        """
        super(ANN, self).__init__()
        self.ID = ID
        self.persist = persist
        self.classes = None  # Unique class labels, set on self.train()
        
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
        self.f_act = nn.Sigmoid()
        self.f_loss = nn.MSELoss()

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
        _, idx = torch.max(outputs, 0)
        return self.classes[idx]

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
        if not self.classes:
            self.classes = data.classes
            self.model.log('Set self.classes:' + str(self.classes))
        else:
            self.model.log('Still using old self.classes:' + str(self.classes))

        self.model.log('Training Completed: ' + info_str + '\n')

        # Save the updated model, including the class labels, to the model file
        if self.persist:
            self.model.save()

    def validate(self, data, verbose=False):
        """ Validates the ANN against the given data set.
        """
        info_str = 'file={}.'.format(data.fname)
        self.model.log('Validation started: ' + info_str)

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

        # If verbose, output detailed accuracy info
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
        
        self.model.log(log_str)

    def classify(self, inputs):
        """ Returns the ANN's classification of the given input tensor.
        """
        return self._label_from_outputs(self(inputs))


if __name__ == '__main__':
    # Define and load training and validation sets
    # trainfile = 'static/datasets/letter_train.data'
    trainfile = 'static/datasets/test.data'  # debug
    # valfile = 'static/datasets/letter_val.data'
    valfile = 'static/datasets/test.data'  # debug
    train_data = DataFromCSV(trainfile, (0, 15))
    val_data = DataFromCSV(valfile, (0, 15))

    # The ANN's layer sizes. In/out sz is based on the dataset's dimennsions
    x_sz = train_data.feature_count
    h_sz = 14
    y_sz = train_data.class_count
    ann_dimens = (x_sz, h_sz, y_sz)

    # Init, train, and subsequently validate the ANN
    ann = ANN('ann.1.1', ann_dimens, console_out=True, persist=True)
    ann.train(train_data, epochs=3000, lr=.1, alpha=.4, stats_at=100, noise=None)
    ann.validate(val_data, verbose=True)

    # Example of a classification request, given a feature vector for "b"
    b_inputs = torch.tensor(
        [4, 2, 5, 4, 4, 8, 7, 6, 6, 7, 6, 6, 2, 8, 7, 10], dtype=torch.float)
    prediction = ann.classify(b_inputs)
    print('Test Classification: ' + prediction)  # Should print "B"
