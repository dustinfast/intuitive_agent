#!/usr/bin/env python
""" The top-level module for the intuitive agent application. 
    See README.md for a description of the agent/application as a whole.

    Interface:
        Agent() is the main interface. It expects training/validation data as
        an instance obj of type sharedlib.DataFrom(). 
        Persistence and output is handled by sharedlib.ModelHandler().

        If PERSIST = True:
            Agent saves to file PERSIST_PATH/ID.AGENT_FILE_EXT between runs.
            Log statments are logged to PERSIST_PATH/ID.LOG_EXT. 

        If CONSOLE_OUT = True:
            Agent (and its sub-modules) output log statements to stdout.

        If STATS_OUT = True:
            Statistics info output to console

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
        Brute force benchmark
        Add usage instr to readme, including dep installation
        Check private notation for all members
        Statistic graph output through model obj
        L2 does not output node count in multimode
        in_data must contain all labels, otherwise ANN inits break 

"""
__author__ = "Dustin Fast"
__email__ = "dustin.fast@outlook.com"
__license__ = "GPLv3"


# Std lib
import sys
import logging
import threading
from math import sqrt
from itertools import groupby
from datetime import datetime

# Third-party
import numpy as np

# Custom
from ann import ANN
from genetic import Genetic
from connector import Connector
from sharedlib import ModelHandler, DataFrom

PERSIST = True
CONSOLE_OUT = False
STATS_OUT = True

# Top-level user configurables
AGENT_NAME = 'agent1_memdepth1'  # Log file prefix
AGENT_FILE_EXT = '.agnt'         # Log file extension
AGENT_ITERS = 5                  # Num times to iterate AGENT_INPUTFILES

# Agent input data. Length denotes the agent's L1 and L2 depth.
AGENT_INPUTFILES = [DataFrom('static/datasets/letters1.csv'),
                    DataFrom('static/datasets/letters2.csv'),
                    DataFrom('static/datasets/letters3.csv'),
                    DataFrom('static/datasets/letters4.csv'),
                    DataFrom('static/datasets/letters5.csv'),
                    DataFrom('static/datasets/letters6.csv'),
                    DataFrom('static/datasets/letters7.csv')]

# Layer 1 user configurables
L1_EPOCHS = 1000            # Num L1 training epochs (per node)
L1_LR = .001                # ANN learning rate (all nodes)
L1_ALPHA = .9               # ANN learning rate momentum (all nodes)

# Layer 1 training data (per node). Length must match len(AGENT_INPUTFILES)
L1_TRAINFILES = [DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv')]

# Layer 1 validation data (per node). Length must match len(AGENT_INPUTFILES)
L1_VALIDFILES = [DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv')]

# Layer 2 user configurables
L2_EXT = '.lyr2'
L2_KERNEL_MODE = 1      # 1 = no case flip, 2 = w/case flip
L2_MUT_REPRO = 0.10     # Genetic mutation ration: Reproduction
L2_MUT_POINT = 0.40     # Genetic mutation ration: Point
L2_MUT_BRANCH = 0.10    # Genetic mutation ration: Branch
L2_MUT_CROSS = 0.40     # Genetic mutation ration: Crossover
L2_MAX_DEPTH = 5        # Max is 10, per KarooGP (has perf affect)
L2_GAIN = .75           # Fit/random ratio of the genetic pool
L2_MEMDEPTH = 1         # Working mem depth, an iplier of L1's input sz
L2_MAX_POP = 50         # Genetic population size (has perf affect)
L2_TOURNYSZ = int(L2_MAX_POP * .25)  # Genetic pool size

# Layer 3 user configurables
L3_EXT = '.lyr3'
L3_CONTEXTMODE = Connector.is_python_kwd


class ConceptualLayer(object):
    """ An abstraction of the agent's conceptual layer (i.e. layer one), which
        represents its "sensory input". 
        Provides interfaces to each layer-node and a its current output.
        Nodes at this level are ANN's representing a single sensory 
        input processing channel, where the input to each channel is a sample
        of some subset of the agent's environment. Its output is then its
        "classification" of what that input represents.
        On init, each node is loaded by the ANN object from file iff PERSIST.
        Note: This layer must be trained offline via self.train(). After 
        training, each node saves its model to file iff PERSIST.
        """
    def __init__(self, ID, depth, dims, inputs):
        """ Accepts:
                ID (str)        : This layers unique ID
                depth (int)     : How many nodes this layer contains
                dims (list)     : 3 ints - ANN in/hidden/output layer sizes
        """
        self._nodes = []        # A list of nodes, one for each layer 1 depth
        self._depth = depth     # This layer's depth. I.e., it's node count
        
        for i in range(depth):
            nodeID = ID + '_node_' + str(i)
            self._nodes.append(ANN(nodeID, dims[i], CONSOLE_OUT, PERSIST))
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
        for i in range(self._depth):
            self._nodes[i].train(
                train_data[i], epochs=epochs, lr=lr, alpha=alpha, noise=None)
            self._nodes[i].validate(val_data[i], verbose=True)

    def forward(self, inputs):
        """ Moves the given inputs through the layer, setting self.outputs
            appropriately.
            Accepts:
                inputs (list)   : A list of tensors, one per node.
            Returns:
                A list of outputs with one element (a list) for each node.
            
        """
        outputs = []
        for i in range(self._depth):
            outputs.append(self._nodes[i].classify(inputs[i]))
        return outputs

    
class IntuitiveLayer(object):
    """ An abstraction of the agent's second layer, representing the ability to
        intuitively form new connections between symbols, as well as its
        recurrent memory (i.e. it's last L2_MEMDEPTH)
    """
    def __init__(self, ID, size, mem_depth):
        """ Accepts:
                ID (str)        : This layers unique ID
                size (int)      : Num input terminals
        """
        self.ID = ID            
        self._size = size
        self._mem_depth = mem_depth       
        self._node = None               # The GP element
        self._nodeID = ID + '_node'     # GP node's unique ID

        # Init the layer's node - a genetically evolving tree of expressions
        self._node = Genetic(ID=self._nodeID,
                             kernel=L2_KERNEL_MODE,
                             max_pop=L2_MAX_POP,
                             max_depth=L2_MAX_DEPTH,
                             max_inputs=self._size,
                             mem_depth=self._mem_depth,
                             tourn_sz=L2_TOURNYSZ,
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
    """ An abstraction of the agent's Logical layer (i.e. layer three), which
        evaluates the fitness of it's input according to the given context mode.
        Statistics are also tracked here.
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
        self.encounters = []       # Not new but seen first time this epoch
        self.encounters_t = []              # Times of each encounter
        self.re_encounters = []    # Items for which 1 or more encounters occur
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
                    in_ep = '\n( %ss into epoch %s)' % (sec_in, str(self.epoch))

                    # If LEARNED / seeing this item for the very first time
                    if item not in self.kb_lifetime:
                        fitness[treeID] += 100

                        # Update stat containers
                        self.kb_lifetime.append(item)
                        self.learned.append(item)
                        self.learned_t.append(sec_in)

                        self.model.log('L3 LEARNED: ' + item + in_ep)
                        print('L3 LEARNED: ' + item + in_ep)

                    # If ENCOUNTERED / seen item before but not this epoch
                    elif item not in self.encounters:
                        fitness[treeID] += 10

                        # Update stat containers
                        self.encounters.append(item)
                        self.encounters_t.append(sec_in)

                        self.model.log('L3 Encountered: ' + item + in_ep)
                        print('L3 Encountered: ' + item + in_ep)

                    # Else RENCOUNTERED / we've seen this item this epoch
                    else:
                        fitness[treeID] += 1

                        # Update stat containers
                        self.re_encounters.append(item)
                        self.re_encounters_t.append(sec_in)

                        self.model.log('L3 Re-encountered: ' + item)

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

    def stats_get(self, stime, clear=False, graph=False):
        """ Generates statistics from the current state and returns the results
            as a string.
            Accepts:
                stime (datetime)    : A datetime representing run start time
                clear (bool)        : Also reset stats / start new stats epoch
                graph (bool)        : Also display stats in graph form
        """
        # Misc stats
        epoch = self.epoch
        epoch_time = (datetime.now() - self.epoch_time).seconds
        run_time = (datetime.now() - stime).seconds
        learns = len(self.learned)
        encounters = len(self.encounters)
        re_encounters = len(self.re_encounters)

        avg_try_len = 0     # Average length of all inputs
        if self.input_count:
            avg_try_len = self.input_lens / self.input_count

        # re_var = 0          # Variance among re-encounters
        # if self.encounters:
        #     dist = [len(list(group))
        #             for _, group in groupby(self.re_encounters)]
        #     re_len = len(self.encounters)
        #     avg = sum(dist) / re_len
        #     re_var = sum((x - avg) ** 2 for x in dist) / re_len
            
        ret_str = '-- Epoch %s Statistics --\n' % epoch
        ret_str += ' Epoch run time: %d\n' % epoch_time
        ret_str += '   Total inputs: %d\n' % self.input_count
        ret_str += '   Avg try length: %d\n' % avg_try_len

        ret_str += '   Total learns: %d\n' % learns
        ret_str += '   Total encounters: %d\n' % encounters
        ret_str += '   Total re-encounters: %d\n' % re_encounters
        # ret_str += '   Re-encounter variance: %d\n' % re_var

        ret_str += 'Total run time: %ds\n' % run_time

        if clear:
            self.stats_clear()

        return ret_str

    def run_benchmark(self, width=4, epochs=3):
        """ A function for establishing baseline performance metrics by brute
            forcing strings against the current context mode. Benchmark 
            statistics are output to the console.
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
        t_start = datetime.now()
        for _ in range(epochs):
            for _ in range(epochs):
                for i in range(1, width + 1):
                    for item in (''.join(s) for s in product(ascii_lowercase, repeat=i)):
                        self.input_count += 1
                        self.input_lens += len(item)

                        if L3_CONTEXTMODE(item):
                            sec_in = (datetime.now() - self.epoch_time).seconds

                            # If seeing item for the very first time
                            if item not in test_kb:
                                print('L3 LEARNED: ' + item)
                                test_kb.append(item)
                                self.learned.append(item)
                                self.learned_t.append(sec_in)

                            # If seen item before but not this epoch
                            elif item not in self.encounters:
                                print('L3 ENCOUNTERED: ' + item)
                                self.encounters.append(item)
                                self.encounters_t.append(sec_in)

                            # Else we've seen this item this epoch
                            else:
                                print('L3 Re-encountered: ' + item)
                                self.re_encounters.append(item)
                                self.re_encounters_t.append(sec_in)
                
                print(self.stats_get(t_start))

            self.stats_get(t_start, clear=True)


class Agent(threading.Thread):
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
        threading.Thread.__init__(self)
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
        self.l1 = ConceptualLayer(ID, self.depth, l1_dims, inputs)
        ID = id_prefix + 'L2'
        self.l2 = IntuitiveLayer(ID, self.depth, L2_MEMDEPTH)
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

    def start(self, max_iters=10):
        """ Starts the agent thread.
            Accepts:
                max_iters (int)     : Max times to iterate data set (0=inf)
        """
        self.max_iters = max_iters
        threading.Thread.start(self)

    def run(self):
        """ Starts the agent thread, stepping the agent forward until stopped 
            externally with self.stop() or (eof reached AND stop_at_eof)
        """
        self.model.log('Agent thread started.')
        min_rows = min([data.row_count for data in self.inputs])
        self.running = True
        stime = datetime.now()
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

            if STATS_OUT:
                print(self.l3.stats_get(stime, clear=True))  # Output stats

            if PERSIST:
                self.model.save()  # Save the model to file

            if self.max_iters and iters >= self.max_iters - 1:
                self.stop('Agent stopped: max_iters reached.')
            
            iters += 1

    def stop(self, output_str='Agent stopped.'):
        """ Stops the thread. May be called from the REPL, for example.
            Also logs the given output string (if any) and saves the agent to
            file (iff PERSIST).
        """
        self.running = False
        self.model.log(output_str)


if __name__ == '__main__':
    # Instantiate the agent (Note: agent shape derived from input data)
    agent = Agent(AGENT_NAME, AGENT_INPUTFILES)

    # Train and validate each layer 1 node, if specified by cmd line arg
    if len(sys.argv) > 1 and sys.argv[1] == '-l1_train':
            print('Training layer 1...', sep=' ')
            agent.l1.train(L1_TRAINFILES, L1_VALIDFILES)
            print('Done.')

    if len(sys.argv) > 1 and sys.argv[1] == '-benchmark':
        agent.l3.run_benchmark()
        exit()

    # Start the agent thread
    print('Running ' + AGENT_NAME)
    agent.start(AGENT_ITERS)
    agent.join()
