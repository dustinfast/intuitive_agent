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
from datetime import datetime

# Custom
from ann import ANN
from genetic import Genetic
from connector import Connector
from sharedlib import ModelHandler, DataFrom

PERSIST = True
CONSOLE_OUT = False

# Agent user configurables
AGENT_NAME = 'agent1_memdepth1'  # Log file prefix
AGENT_FILE_EXT = '.agnt'         # Log file extension
AGENT_ITERS = 1                  # Num times to iterate AGENT_INPUTFILES as input

# Agent input data. Length denotes the agent's L1 and L2 depth.
AGENT_INPUTFILES = [DataFrom('static/datasets/letters1.csv'),
                    DataFrom('static/datasets/letters1.csv'),
                    DataFrom('static/datasets/letters2.csv'),
                    DataFrom('static/datasets/letters3.csv')]



# Layer 1 user configurables
L1_EPOCHS = 1000            # Num L1 training epochs (per node)
L1_LR = .001                # ANN learning rate (all nodes)
L1_ALPHA = .9               # ANN learning rate momentum (all nodes)

# Layer 1 training data (per node). Length must match len(AGENT_INPUTFILES)
L1_TRAINFILES = [DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv'),
                 DataFrom('static/datasets/letter_train.csv')]

# Layer 1 validation data (per node). Length must match len(AGENT_INPUTFILES)
L1_VALIDFILES = [DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv'),
                 DataFrom('static/datasets/letter_val.csv')]

# Layer 2 user configurables
L2_EXT = '.lyr2'
L2_KERNEL_MODE = 1      # 1 = no case flip, 2 = w/case flip
L2_MUT_REPRO = 0.10     # Genetic mutation ration: Reproduction
L2_MUT_POINT = 0.40     # Genetic mutation ration: Point
L2_MUT_BRANCH = 0.00    # Genetic mutation ration: Branch
L2_MUT_CROSS = 0.50     # Genetic mutation ration: Crossover
L2_MAX_DEPTH = 5        # Max is 10, per KarooGP (has perf affect)
L2_GAIN = .75           # Fit/random ratio of the genetic pool
L2_MEMDEPTH = 1         # Working mem depth, an iplier of L1's input sz
L2_MAX_POP = 50         # Genetic population size (has perf affect)
L2_TOURNYSZ = int(L2_MAX_POP * .25)  # Genetic pool size

# L2_MUT_REPRO = 0.15   # Default genetic mutation ration: Reproduction
# L2_MUT_POINT = 0.15   # Default genetic mutation ration: Point
# L2_MUT_BRANCH = 0.0   # Default genetic mutation ration: Branch
# L2_MUT_CROSS = 0.7    # Default genetic mutation ration: Crossover

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
    """
    def __init__(self, ID, mode):
        """ Accepts:
                ID (str)            : This layers unique ID
                mode (function)     : A bool-returning func accepting L2 output
        """
        self.ID = ID
        self._mode = mode

        # Heuristics variables
        self.last_learnhit = datetime.now()     # New item learned event time
        self.last_encounter = datetime.now()    # Last saw a prev learned item
        self.learnhits = []                     # Number of new items learned
        self.encounterhits = []                 # Num encounters w/items in kb 
        self.len_total = 0                      # Cumulative len of inputs fed
        self.len_count = 0                      # Count of all inputs fed

        # Persistent containers and metrics
        self.kb = []                # Lifetime list of all learned items
        self._life_learnhits = 0    # Lifetime "new item learned" hits count
        self._life_encounters = 0   # Lifetime encounters w/items in kb
        self._life_learn_t = 0      # Lifetime "new item learned" hits count
        self._life_enc_t = 0        # Lifetime avg time btwn encounters

        # Save/Load/Loghandler
        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L3_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nMode = ' + self._mode.__name__
        str_out += '\nLifetime learning hits: ' + str(self._life_learnhits)
        str_out += '\n  Avg time between learn hits: ' + str(self._life_learn_t)
        str_out += '\nLifetime encounters: ' + str(self._life_encounters)
        str_out += '\n  Avg time between encounters: ' + str(self._life_enc_t)
        return str_out

    def _save(self, filename):
        """ Saves a model of the current population. For use by ModelHandler.
            Iff no filename given, does not save to file but instead returns
            the string that would have otherwise been written.
        """
        # Build model params in dict form
        writestr = "{ '_mode': 'Connector." + self._mode.__name__ + "'"
        writestr += ", 'kb': " + str(self.kb)
        writestr += ", '_life_learnhits': " + str(self._life_learnhits)
        writestr += ", '_life_learn_t': " + str(self._life_learn_t)
        writestr += ", '_life_encounters': " + str(self._life_encounters)
        writestr += ", '_life_enc_t': " + str(self._life_enc_t)
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
        self.kb = data['kb']
        self._mode = eval(data['_mode'])
        self._life_learnhits = data['_life_learnhits']
        self._life_learn_t = data['_life_learn_t']
        self._life_encounters = data['_life_encounters']
        self._life_enc_t = data['_life_enc_t']

    def forward(self, results):
        """ Checks each result in results and determines fitness.
            Accepts:
                fitness (AttrIter) : Keyed by tree ID with 'ouput' label
            Returns:
                dict: { treeID: FitnessScore }
        """
        fitness = {k: 0.0 for k in results.keys()}

        for trees in results:
            for treeID, attrs in trees.items():
                tryme = attrs['output']
                self.len_count += 1
                self.len_total += len(tryme)

                self.model.log('L3 TRYING: ' + tryme)
                if self._mode(tryme):

                    if tryme not in self.kb:
                        lasthit = (datetime.now() - self.last_learnhit).seconds
                        self.model.log('L3 LEARNED: ' + tryme)
                        self.model.log('(Last was: ' + str(lasthit) + 's ago)')
                        print('L3 LEARNED: ' + tryme)
                        print('(Last was: ' + str(lasthit) + 's ago)')
                        self.kb.append(tryme)
                        self.learnhits.append(lasthit)
                        self.encounterhits.append(lasthit)
                        self.last_learnhit = datetime.now()
                        self.last_encounter = datetime.now()
                        fitness[treeID] += 10
                    else:
                        lasthit = (datetime.now() - self.last_encounter).seconds
                        self.model.log('L3 Encountered: ' + tryme)
                        self.model.log('(Last was: ' + str(lasthit) + 's ago)')
                        self.encounterhits.append(lasthit)
                        self.last_encounter = datetime.now()
                        fitness[treeID] += .5

        return fitness

    def stats(self, clear=False):
        """ Updates the lifetime statistics and returns a string representing 
            performance statistics.
            Accepts:
                clear (bool)    : Denotes stats to be reset after generating
                # TODO: (bool)  : Denotes results returned as dict vs str
        """
        # Build statistics
        iters = self.len_count / L2_MAX_POP
        l_hits = len(self.learnhits)
        e_hits = len(self.encounterhits)

        l_hits_time = 0
        avg_hit_len = 0
        if l_hits:
            l_hits_time = sum(self.learnhits) / l_hits
            avg_hit_len = sum([len(h) for h in self.kb]) / l_hits

        e_hits_time = 0
        if e_hits:
            e_hits_time = sum(self.encounterhits) / e_hits

        avg_len = 0
        if self.len_count:
            avg_len = str(self.len_total / self.len_count)

        # Update lifetime stats
        self._life_learnhits += l_hits
        self._life_learn_t += l_hits_time
        self._life_encounters += e_hits
        self._life_enc_t += e_hits_time

        ret_str = 'Total iterations: ' + str(iters) + '\n'
        ret_str += ' Avg try length: ' + str(avg_len) + '\n'

        ret_str += 'Total learn hits: ' + str(l_hits) + '\n'
        ret_str += ' Avg learn hit length: ' + str(avg_hit_len) + '\n'
        ret_str += ' Avg time btwn learn hits -\n'
        ret_str += '    This run: ' + str(l_hits_time) + '\n'
        ret_str += '    Lifetime (NEEDS FIXED): ' + str(self._life_learn_t) + '\n'

        ret_str += 'Total encounters: ' + str(e_hits) + '\n'
        ret_str += ' Avg time btwn encounters (NEEDS FIXED): ' + str(e_hits_time) + '\n'

        ret_str += 'Learned: \n' + str(self.kb) + '\n'

        if clear:
            self.learnhits = []
            self.encounterhits = []
            self.len_total = 0
            self.len_count = 0

        return ret_str


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

    def get_stats(self, reset=False):
        pass

    def do_bruteforce_benchmark(self):
        """ A function for establishing a baseline performance metric by brute
            forcing strings against the current context mode.
        """
        max_width = len(self.inputs)                # Max string width
        charset = [chr(i) for i in range(97, 123)]  # Ascii chars a-z
        t_start = datetime.now()                    # Start time

        print('Establishing benchmark performance by brute force...', sep=' ')
        
        # Query every combination of characters for the given charset and 

        print('Done.')
        print('  Run time: ' + str((datetime.now() - stime).seconds) + 's\n')


            

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

            # Minimal console output
            print('-- Epoch', iters, 'complete --')
            print(self.l3.stats(clear=True))
            print('Run time: ' + str((datetime.now() - stime).seconds) + 's\n')
            self.l2._node.clear_mem()

            if PERSIST:
                self.model.save()

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

    # Start the agent thread
    print('Running ' + AGENT_NAME)
    agent.start(AGENT_ITERS)
    agent.join()
