#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for a description of the agent/application as a whole.

    Interface:
        Agent() is the main interface and expects training/validation/input data as
        an instance obj of type sharedlib.DataFrom(). 
        Persistence and log/console output handled by sharedlib.ModelHandler().
        Performance data is displayed graphically.

        If PERSIST = True:
            Agent saves to file PERSIST_PATH/ID.AGENT_FILE_EXT between runs.
            Log statments are logged to PERSIST_PATH/ID.LOG_EXT. 

        If CONSOLE_OUT = True:
            Agent (and its sub-modules) output log statements to stdout.

    Usage: 
        Run from the terminal with './agent.py'.
        To pre-train layer 1, run with './agent.py -l1_train'.
    
    Dependencies:
        KarooGP         (lib/karoo_gp)
        Matplotlib      (pip install matplotlib)
        Numpy           (pip install numpy)
        Pandas          (pip install pandas)
        PyTorch         (see https://pytorch.org)
        Requests        (pip install requests
        Scikit-learn    (pip install scikit-learn)
        Sympy           (pip install sympy)
        TensorFlow      (See https://www.tensorflow.org/install)
        Scipy           (pip install scipy)

    # TODO: 
        Log file does not auto-rotate
        Add usage instr to readme, including dep installation
        Check private notation for all members
        Refactor stats out
        L2 does not output node count in multimode
        in_data must contain all labels, otherwise L1 inits break 

"""
__author__ = "Dustin Fast"
__email__ = "dustin.fast@outlook.com"
__license__ = "GPLv3"


# Std lib
import time
import logging
from math import sqrt
from datetime import datetime
from itertools import groupby
from threading import Thread
from optparse import OptionParser as opt

# Custom
from classifier import Classifier   # Used by Layer one (one instance/node)
from genetic import Genetic         # Used by layer two (one instance/agent)
from connector import Connector     # Used by layer three (one instance/agent)
from sharedlib import ModelHandler  # Handles file persistence/logging/output
from sharedlib import DataFrom      # Loads datasets from file into memory
from sharedlib import MultiPlotAnimated  # Handles the agent's output graphs

# Output toggles
PERSIST = True                  # File persistence
CONSOLE_OUT = False             # Log statement output to console
STATS_OUT = True                # Statistics output to console

# Top-level user configurables
AGENT_NAME = 'agent1_memdepth1'  # Log file prefix
AGENT_FILE_EXT = '.agent'        # Log file extension
AGENT_ITERS = 10                 # Num times to iterate AGENT_INPUTFILES

# Layer 1 user configurables
L1_EPOCHS = 1000                # Num L1 training epochs (per node)
L1_LR = .001                    # Classifier learning rate (all nodes)
L1_ALPHA = .9                   # Classifier lr momentum (all nodes)

# Layer 2 user configurables
L2_EXT = '.lyr2'                # Layer two model-file extension
L2_KERNEL_MODE = 1              # 1 = no case flip, 2 = w/case flip
L2_MUT_REPRO = 0.10             # Genetic mutation ration: Reproduction
L2_MUT_POINT = 0.40             # Genetic mutation ration: Point
L2_MUT_BRANCH = 0.10            # Genetic mutation ration: Branch
L2_MUT_CROSS = 0.40             # Genetic mutation ration: Crossover
L2_MAX_DEPTH = 5                # Max is 10, per KarooGP (has perf affect)
L2_GAIN = .75                   # Fit/random ratio of the genetic pool
L2_MEMDEPTH = 1                 # Working mem depth, an iplier of L1's input sz
L2_MAX_POP = 50                 # Genetic population size (has perf affect)
L2_POOLSZ = int(L2_MAX_POP * .25)  # Genetic pool size

# Layer 3 user configurables
L3_EXT = '.lyr3'                # Layer three model-file extension
L3_CONTEXTMODE = Connector.is_python_func  # Agent's context mode

# Agent input data set - length denotes the number of layer-one nodes
AGENT_INPUTFILES = ['static/datasets/letters0.csv',
                    'static/datasets/letters1.csv',
                    'static/datasets/letters2.csv',
                    'static/datasets/letters3.csv',
                    'static/datasets/letters4.csv',
                    'static/datasets/letters5.csv',
                    'static/datasets/letters6.csv',
                    'static/datasets/letters7.csv']

# Layer 1 training data (by node) - length must match len(AGENT_INPUTFILES)
L1_TRAINFILES = ['static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv',
                 'static/datasets/letter_train.csv']

# Layer 1 validation data (by node) - length must match len(AGENT_INPUTFILES)
L1_VALIDFILES = ['static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv',
                 'static/datasets/letter_val.csv']

# Output graph labels
GRAPH_LEGEND_AGENT = (['Avg Try Len'], 
                      ['Learns'],
                      ['Encounters'],
                      ['ReEncounters'],
                      ['ReE Std Dev'])
GRAPH_LEGEND_L1TRAIN = (['Training Loss'], ['Validation Acc'])

# Module-level globals
g_start_time = datetime.now()    # Application start time
g_graph_out = None               # Output graph handler


class ClassificationLayer(object):
    """ An abstraction of the agent's classification layer (i.e. layer one), 
        representing our ability to recognize pre-learned symbols in our
        environement. Contains layer 1 nodes and exposes their outputs.
        Nodes at this level are classifiers, each representing a single
        input processing channel - the input to each channel is a feature
        vector of data representing some subset of the agent's environment. 
        Its output, then, is its classification of that input.
        Persistence:
            On init, each node loads from file via class Classifer iff PERSIST.
        Training:
            This layer must be trained offline via self.train(). After 
            training, each node saves its model to file iff PERSIST.
        """
    def __init__(self, ID, depth, dims, inputs):
        """ Accepts:
                ID (str)        : This layers unique ID
                depth (int)     : How many nodes this layer contains
                dims (list)     : 3 ints - classfier x/h/y layer sizes
                inputs (list)   : The agent's input data - a list of vectors
        """
        self._nodes = []        # A list of nodes, one for each layer 1 depth
        self._depth = depth     # This layer's depth. I.e., it's node count
        
        for i in range(depth):
            nodeID = ID + '_node_' + str(i)
            self._nodes.append(Classifier(nodeID, dims[i], CONSOLE_OUT, PERSIST))
            self._nodes[i].set_labels(inputs[i].class_labels)
    
    def train(self, train_data, val_data, epochs=L1_EPOCHS,
              lr=L1_LR, alpha=L1_ALPHA):
        """ Trains each node from the given training/validation data.
            Accepts:
                train_data (list)       : A list of DataFrom objects
                val_data (list)         : A list of DataFrom objects
                epochs (int)            : Number of training iterations
                lr (float)              : Learning rate
                alpha (float)           : Learning gain/momentum
        """
        print('Training layer 1 nodes...')
        self._t_stats = datetime.now()
        self._base_epoch = 0
        self._t_i = 0
        self._val_stats = []

        for i in range(self._depth):
            node_str = 'Node %d' % (i + 1)
            g_graph_out.annotate('<-%s' % node_str)
            self._t_node = self._nodes[i]
            self._t_i = i + i
            self._t_node.train(train_data[i], epochs=epochs, lr=lr, alpha=alpha)
            if self._t_i != self._depth:
                self._base_epoch += self._t_node.train_epoch + 1
            self._t_node.validate(val_data[i], verbose=True)
            self._val_stats.append(
                '%s: %d%%' % (node_str, self._t_node.train_acc))
        print('Done training layer 1 nodes.')
        self._val_stats.append(
            ' Done. %ds' % (datetime.now() - g_start_time).seconds)
        time.sleep(.5)       # Ensure no pause before val stats get refreshed
        g_graph_out.pause()  # Pause stats graph

    def forward(self, inputs):
        """ Moves the given inputs through the layer.
            Accepts:
                inputs (list)   : A list of tensors, one per node.
            Returns:
                A list of outputs with one element (a list) for each node.
        """
        outputs = []
        for i in range(self._depth):
            outputs.append(self._nodes[i].classify(inputs[i]))
        return outputs

    def stats_graphdata(self):
        """ Returns training statistics as a tuple for graph display.
        """
        try:
            ret = (self._t_node.train_loss,
                   self._t_node.train_epoch + self._base_epoch)
        except AttributeError:
            ret = (0, 0)  # No training data yet

        return ret

    def stats_graphtxt(self):
        """ Returns training text field data for graph display.
        """
        epoch = 0
        results = ''
        try:
            epoch = self._t_node.train_epoch
            results = ', '.join(self._val_stats)
        except AttributeError:
            pass  # No training data yet

        s = 'Training/Validating node %d: %d/%d epochs' 
        s1 = (s % (self._t_i, epoch, L1_EPOCHS))
        s2 = 'Results: %s' % results 
        return s2, s1

    
class EvolutionaryLayer(object):
    """ An abstraction of the agent's second layer, representing our ability to
        intuitively form new symbols composed of connections between known
        symbols in the environemnts (including working memory).
    """
    def __init__(self, ID, size, mem_depth):
        """ Accepts:
                ID (str)            : This layers unique ID
                size (int)          : Num input terminals
                mem_depth (int)     : Num prev inputs to keep in memory
        """
        self.ID = ID            
        self._size = size
        self._mem_depth = mem_depth       
        self._node = None               # The gentic programming (GP) obj
        self._nodeID = ID + '_node'     # GP node's unique ID

        # Init the layer's node - a genetically evolving tree of expressions
        self._node = Genetic(ID=self._nodeID,
                             kernel=L2_KERNEL_MODE,
                             max_pop=L2_MAX_POP,
                             max_depth=L2_MAX_DEPTH,
                             max_inputs=self._size,
                             mem_depth=self._mem_depth,
                             tourn_sz=L2_POOLSZ,
                             console_out=CONSOLE_OUT,
                             persist=False)

        # Set default node mutation ratios (overwritten if loading from file)
        self._node.set_mratio(repro=L2_MUT_REPRO,
                              point=L2_MUT_POINT,
                              branch=L2_MUT_BRANCH,
                              cross=L2_MUT_CROSS)

        # Init the model handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L2_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nSize = ' + str(self._size)
        str_out += '\nMem Depth = ' + str(self._mem_depth)
        return str_out

    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write the node's ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')
            savestr = self._node._save()
            f.write('"' + self._nodeID + '": """' + savestr + '""", ')
            f.write('}')

    def _load(self, filename):
        """ Loads the layer from file. For use by ModelHandler.
            Note: The node ID in the file is ignored
        """
        with open(filename, 'r') as f:
            data = f.read()

        loadme = next(iter(eval(data).values()))
        ID = self._nodeID
        node = Genetic(ID, 2, 0, 0, 0, 0, 0, CONSOLE_OUT, False)  # skeleton
        node._load(loadme, not_file=True)
        self._node = node

    def forward(self, inputs):
        """ Moves the given inputs through the layer and returns the output.
            Accepts: 
                inputs (list)       : Data elements, one for each node
            Returns:
                dict: { TreeID: {'output': [], 'in_context':[]}, ... }
        """
        return self._node.apply(inputs=inputs)

    def update(self, fitness):
        """ Updates the layer's node according to the given fitness data.
            Accepts:
                fitness (dict) : { treeID: {'fitness': x, 'in_context':[]} }
        """
        self._node.update(fitness)


class LogicalLayer(object):
    """ An abstraction of the agent's Logical layer (i.e. layer three).
        Represents our ability to validate ideas against the environment.
        At this level, fitness of it's input are evaluated according to the
        given context mode.
        Performance statistics are also tracked/exposed by this level.
    """
    def __init__(self, ID, mode):
        """ Accepts:
                ID (str)            : This layers unique ID
                mode (function)     : A bool-returning func accepting L2 output
        """
        self.ID = ID
        self._context = mode

        # Statistics
        self.epoch_time = datetime.now()    # Start of current stats epoch
        self.epoch = 1                      # Stats epoch count
        self.kb_lifetime = []               # Items learned, lifetime
        self.learned = []                   # New items learned this epoch
        self.learned_t = []                 # Times each item was learned
        self.encounters = []                # Seen for first time this epoch
        self.encounters_t = []              # Times of each encounter
        self.re_encounters = []             # Items encounters 2+ times 
        self.re_encounters_t = []           # Times for each re-encounter
        self.input_lens = 0                 # Sum(Len(all inputs received))
        self.input_count = 0                # Count of all inputs received

        # Save/Load/Loghandler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L3_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nContext Mode = ' + self._context.__name__
        return str_out

    def _save(self, filename):
        """ Saves a model of the current population. For use by ModelHandler.
            Iff no filename given, does not save to file but instead returns
            the string that would have otherwise been written.
        """
        # Build model params in dict form
        writestr = "{ '_context': 'Connector." + self._context.__name__ + "'"
        writestr += ", 'kb': " + str(self.kb_lifetime)
        writestr += "}"

        if not filename:
            return writestr

        with open(filename, 'w') as f:
            f.write(writestr)

    def _load(self, filename, not_file=False):
        """ Loads model from file. For use by ModelHandler.
            Iff not_file, does not load from file but instead loads from the
            given string as if it were file contents.
        """
        if not_file:
            loadfrom = filename
        else:
            with open(filename, 'r') as f:
                loadfrom = f.read()

        data = eval(loadfrom.replace('\n', ''))
        self.kb_lifetime = data['kb']
        self._context = eval(data['_context'])

    def forward(self, results):
        """ Checks each result in results against the current context mode and 
            qauntifies its fitness.
            Accepts:
                fitness (AttrIter) : Keyed by tree ID with 'ouput' label
            Returns:
                dict: { treeID: FitnessScore }
        """
        fitness = {k: 0.0 for k in results.keys()}

        for trees in results:
            for treeID, attrs in trees.items():
                item = attrs['output']
                self.model.log('L3 TRYING: ' + item)

                # Update stats
                self.input_count += 1
                self.input_lens += len(item)

                # If item is valid in the current context
                if self._context(item):
                    sec_in = (datetime.now() - self.epoch_time).seconds
                    in_ep = '\n(%ss into epoch %s)' % (sec_in, str(self.epoch))

                    # If seeing this item for the very first time this epoch
                    if item not in self.learned:
                        self.learned.append(item)
                        self.learned_t.append(sec_in)

                        # If seeing item for the very first time ever
                        if item not in self.kb_lifetime:
                            fitness[treeID] += 100

                            self.kb_lifetime.append(item)
                            
                            self.model.log('L3 LEARNED: ' + item + in_ep)
                            print('L3 LEARNED: ' + item + in_ep)

                        # Else, seeing it for the first time this epoch
                        else:
                            fitness[treeID] += 10

                            self.encounters.append(item)
                            self.encounters_t.append(sec_in)

                            self.model.log('L3 Encountered: ' + item)
                            # print('L3 Encountered: ' + item + in_ep)

                    # Else we've seen this item this epoch
                    else:
                        fitness[treeID] += 1

                        self.re_encounters.append(item)
                        self.re_encounters_t.append(sec_in)

                        self.model.log('L3 Re-encountered: ' + item)
                        # print('L3 RE-encountered: ' + item + in_ep)

        return fitness

    def stats_clear(self):
        """ Clears the current statistics / Starts a new statistics epoch.
        """
        self.epoch += 1
        self.epoch_time = datetime.now()
        self.learned = []
        self.encounters = []
        self.re_encounters = []
        self.input_lens = 0
        self.input_count = 0

    def _stats_dict(self, stime=g_start_time):
        """ Returns statistics in dict form.
            Accepts:
                stime (datetime)    : Application start time
            Returns:
                stats (dictionary)  : See stats dict below for fields
        """
        stats = {'epoch'        : self.epoch,
                 'epoch_time'   : (datetime.now() - self.epoch_time).seconds,
                 'run_time'     : (datetime.now() - stime).seconds,
                 'learns'       : len(self.learned),
                 'encounters'   : len(self.encounters),
                 're_encounters': len(self.re_encounters),
                 're_std_dev'  : -1,
                 'avg_len'      : -1}

        avg_len = 0     # Average length of all inputs this epoch
        if self.input_count:
            avg_len = self.input_lens / self.input_count
        stats['avg_len'] = avg_len

        re_std = 0      # Variance among re-encounters this epoch
        if self.learned:
            res_sorted = sorted(self.re_encounters)
            dist = [len(list(group)) for _, group in groupby(res_sorted)]
            re_len = len(self.learned)
            avg = sum(dist) / re_len
            re_var = sum((x - avg) ** 2 for x in dist) / re_len
            re_std = sqrt(re_var)
        stats['re_std_dev'] = re_std

        return stats

    def stats_graphdata(self):
        """ Returns plottable performance metrics as a tuple.
        """
        stats = self._stats_dict()
        return (stats['avg_len'], 
                stats['learns'],
                stats['encounters'],
                stats['re_encounters'],
                stats['re_std_dev'],
                stats['run_time'])
    
    def stats_graphtxt(self):
        """ Returns plottable text field values as a tuple
        """
        res_sorted = sorted(self.re_encounters)
        keys = [key for key, _ in groupby(res_sorted)]
        dist = [len(list(group)) for _, group in groupby(res_sorted)]
        
        s1 = 'KB          : %s' % str(self.kb_lifetime)
        s2 = 'Encounters  : %s' % str(self.learned)
        s3 = 'ReEncounters: %s' % str(list(zip(keys, dist)))

        return s3, s2, s1
        
    def stats_str(self, stime=g_start_time, clear=False):
        """ Returns the formatted statistics output string.
            Accepts:
                stime (datetime)    : Application start time
                clear (bool)        : Also reset stats / start new stats epoch
        """
        stats = self._stats_dict()

        ret = '\n-- Epoch %s Statistics --\n' % stats['epoch']
        ret += 'Input count: %d\n' % self.input_count
        ret += 'Avg input length: %d\n' % stats['avg_len']

        ret += '\nTotal learns: %d\n' % stats['learns']
        ret += 'Total encounters: %d\n' % stats['encounters']
        ret += 'Total re-encounters: %d\n' % stats['re_encounters']
        ret += 'Re-encounter variance: %d\n' % stats['re_std_dev']

        ret += '\nLearned (lifetime):\n%s\n' % str(self.kb_lifetime)
        ret += 'Learned (this epoch):\n%s\n' % str(self.learned)

        ret += '\nEpoch run time: %ds\n' % stats['epoch_time']
        ret += 'Total run time: %ds\n' % stats['run_time']

        if clear:
            self.stats_clear()

        return ret

    def _run_benchmark(self, width=4, epochs=3):
        """ A debug function for testing satistics functionality and getting
            baseline performance metrics by brute forcing strings generated
            combinatorily against the current context mode. Benchmark 
            statistics are output via console and graph, depending on consts.
            Accepts:
                width (int)     : Max string width to generate
                epochs (int)    : Benchmark revolutions
        """
        from itertools import product
        from string import ascii_lowercase

        test_kb = []        # Fresh kb 
        self.epoch = 0      # Ensure fresh epoch number
        self.stats_clear()  # Ensure fresh stats containers

        # Query every combination of lcase strings against agent's context
        print('Running benchmark queries...')
        t_start = datetime.now()
        for _ in range(epochs):
            for _ in range(epochs):
                for i in range(1, width + 1):
                    for item in (''.join(s) for s in product(ascii_lowercase, repeat=i)):
                        # TODO: Refactor the below w/self.forward()
                        # Update stats
                        self.input_count += 1
                        self.input_lens += len(item)

                        # If item is valid in the current context
                        if self._context(item):
                            sec_in = (datetime.now() - self.epoch_time).seconds

                            # If seeing this item for the very first time this epoch
                            if item not in self.learned:
                                self.learned.append(item)
                                self.learned_t.append(sec_in)

                                # If seeing item for the very first time ever
                                if item not in test_kb:
                                    test_kb.append(item)
                                    # print('L3 LEARNED: ' + item)

                                # Else, seeing it for the first time this epoch
                                else:
                                    self.encounters.append(item)
                                    self.encounters_t.append(sec_in)
                                    # print('L3 Encountered: ' + item)

                            # Else we've seen this item this epoch
                            else:
                                self.re_encounters.append(item)
                                self.re_encounters_t.append(sec_in)
                                # print('L3 Re-encountered: ' + item)
                
                g_graph_out.annotate('<-epoch')
                print(self. stats_str(t_start))

        g_graph_out.pause()


class Agent(Thread):
    """ The intutive agent.
        The constructor accepts the agent's "sensory input" data, from which
        the layer dimensions are derived. After init, start the agent from 
        the terminal with 'agent start', which runs the agent as a seperate
        thread (running does not cause the agent to block, so the user may
        stop it from the command line, etc.
        Persistence: On each iteration of the input data, the agent is saved 
        to a file.
    """
    def __init__(self, ID, inputs):
        """ Accepts the following parameters:
            ID (str)            : The agent's unique ID
            inputs (list)       : Agent input data, one for each L1 node
        """
        Thread.__init__(self)
        self.ID = ID
        self.depth = None           # Layer 1 depth/node count
        self.model = None           # The model handler
        self.running = False        # Agent thread running flag (set on start)
        self.inputs = inputs        # The agent's "sensory input" data
        self.max_iters = None       # Num inputs iters, set on self.start
        self.L2_nodemap = None      # Pop via ModelHandler, for loading L2
        self.depth = len(inputs)    # L1 node count and L2 inputs count

        # Init the load, save, log, and console output handler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=AGENT_FILE_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

        # Determine layer 1 dimensions from input data
        l1_dims = []
        for i in range(self.depth):
            l1_dims.append([])                              # New L1 dimension
            l1_dims[i].append(inputs[i].feature_count)      # x sz
            l1_dims[i].append(0)                            # h sz placeholder
            l1_dims[i].append(inputs[i].class_count)        # y sz
            l1_dims[i][1] = int(
                (l1_dims[i][0] + l1_dims[i][2]) / 2)        # h sz is xy avg

        # Init layers
        id_prefix = self.ID + '_'
        ID = id_prefix + 'L1'
        self.l1 = ClassificationLayer(ID, self.depth, l1_dims, inputs)
        ID = id_prefix + 'L2'
        self.l2 = EvolutionaryLayer(ID, self.depth, L2_MEMDEPTH)
        ID = id_prefix + 'L3'
        self.l3 = LogicalLayer(ID, L3_CONTEXTMODE)

    def __str__(self):
        return 'ID = ' + self.ID

    def _save(self, filename):
        """ Saves a model of the agent. For use by ModelHandler.
            Also causes saves to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        # Note: L1 handles its own persistence
        self.l2.model.save() 
        self.l3.model.save() 

    def _load(self, filename):
        """ Loads the agent model from file. For use by ModelHandler.
            Also causes loads to occur for each agent layer (and associated
            nodes) that perform online learning.
        """
        # Note: L1 handles its own persistence
        self.l2.model.load() 
        self.l3.model.load()

    def _step(self, data_row):
        """ Steps the agent forward one step with the given data row: A list
            of tuples (one for each depth) of inputs and targets. Each tuple,
            by depth, is fed to layer one, who's ouput is fed to layer 2, etc.
            Note: At this time, the 'targets' in the data row are not used.
            Accepts:
                data_row (list)     : [inputs, ... ]
        """
        # Ensure well formed data_row
        if len(data_row) != self.depth:
            err_str = 'Bad data_row size - expected sz ' + str(self.depth)
            err_str += ', recieved size' + str(len(data_row))
            self.model.log(err_str, logging.error)
            return

        # --------------------- Step Layer 1 -------------------------------
        # L1_outputs[i] becomes L1._nodes[i]'s classification of data_row[i]
        # ------------------------------------------------------------------
        l1_row = [d[0] for d in data_row]  # TODO: Fix [0] in run()

        for i in range(self.depth):
            self.model.log('-- Feeding L1[%d]:\n%s' % (i, str(l1_row[i])))

        l1_outputs = self.l1.forward(l1_row)
        
        # --------------------- Step Layer 2 -------------------------------
        # L2.outputs are the "masked" versions of all L1.outputs w/recurrance
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L2:\n %s' % (l1_outputs))
        l2_outputs = self.l2.forward(list([l1_outputs]))
        
        # --------------------- Step Layer 3 -------------------------------
        # L3 evals fitness of it's input from L2 and backpropagates signals
        # ------------------------------------------------------------------
        self.model.log('-- Feeding L3:\n%s' % str(l2_outputs))
        l3_outputs = self.l3.forward(l2_outputs)

        self.model.log('-- L2 Backprop:\n%s' % str(l3_outputs))
        self.l2.update(l3_outputs)

    def start(self, max_iters=AGENT_ITERS):
        """ Starts the agent thread.
            Accepts:
                max_iters (int)     : Max times to iterate data set (0=inf)
        """
        self.max_iters = max_iters
        Thread.start(self)

    def run(self):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
        """
        self.model.log('Agent thread started.')
        min_rows = min([data.row_count for data in self.inputs])
        self.running = True
        iters = 0

        # Step the agent forward with each row of each dataset
        while self.running:
            for i in range(min_rows):
                row = []
                for j in range(self.depth):
                    row.append([row for row in iter(self.inputs[j][i])])
        
                self.model.log('\n** STEP - iter: %d depth:%d **' % (iters, i))
                self._step(row)

            # End of iteration...
            self.l2._node.clear_mem()  # Keep data consistent across iterations
            
            # Stat keeping
            stats = self.l3. stats_str(g_start_time, clear=True)
            self.model.log(stats)
            g_graph_out.annotate('<-epoch')
            
            if STATS_OUT:
                print(stats)        # Print statistics to console

            if PERSIST:
                self.model.save()   # Save the model to file

            if self.max_iters and iters >= self.max_iters - 1:
                self.stop('Agent stopped: max_iters reached.')
                g_graph_out.pause()
            
            iters += 1

    def stop(self, output_str='Agent stopped.'):
        """ Stops the thread. May be called from the REPL, for example.
            Also logs the given output string (if any) and saves the agent to
            file (iff PERSIST).
        """
        self.running = False
        self.model.log(output_str)
        print(output_str)


if __name__ == '__main__':
    """ This is the main drive for the intutive agent application. 
        The agent thread (or another, depending on cmd line args) runs 
        concurrently with the graph output.
    """
    # Parse cmd line options
    opts = opt()
    opts.add_option('--bmark', action='store_true', dest='bmark',
                    help='Run a benchmark session in the current context.')
    opts.add_option('--l1_train', action='store_true', dest='l1_train',
                    help='Train layer one classifiers from current data sets.')
    opts.add_option('--nograph', action='store_true', dest='nograph',
                    help='Runs without the graph display.')
    (options, args) = opts.parse_args()

    # Instantiate the agent (Note: agent shape is derived from input data)
    agent = Agent(AGENT_NAME, [DataFrom(f) for f in AGENT_INPUTFILES])
    
    # Depending on cmd line args...
    # Run the layer-one training/validation routine, with its graph
    if options.l1_train:
        runthread = Thread(target=agent.l1.train,
                           args=([DataFrom(f) for f in L1_TRAINFILES],
                                 [DataFrom(f) for f in L1_VALIDFILES]))
        g_graph_out = MultiPlotAnimated(1, agent.l1.stats_graphdata,
                                        2, agent.l1.stats_graphtxt,
                                        interval=300, lim_y=0.03,
                                        lock_y_lim=True,
                                        legend=GRAPH_LEGEND_L1TRAIN,
                                        title_txt=AGENT_NAME + ' L1 Training')
    # Or...
    else:
        # Run the benchmark tool
        if options.bmark:
            runthread = Thread(target=agent.l3._run_benchmark)

        # Or run the agent itself
        else:
            runthread = agent

        # With the main graph
        g_graph_out = MultiPlotAnimated(5, agent.l3.stats_graphdata,
                                        3, agent.l3.stats_graphtxt,
                                        interval=1000, lim_y=10,
                                        legend=GRAPH_LEGEND_AGENT,
                                        title_txt=AGENT_NAME)

    # Start the selected thread
    print('Starting %s...' % AGENT_NAME)
    runthread.start()
    
    # Display staistics graphs
    if not options.nograph:
        g_graph_out.play()
        g_graph_out.show()  # Blocks until window closed

    runthread.join()

    print('Quitting... Goodbye.')
