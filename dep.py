""" Deprecated/test functionality - save for possible re-use

    Author: Dustin Fast
"""

# Multiple-node L2: One node for each unique input
class IntuitiveLayer(object):
    """ An abstration of the agent's Intuitive layer (i.e. layer two), which
        represents the intutive "bubbling up" of "pertinent" information to
        layers above it.
        Nodes at this level represent a single genetically evolving population
        of expressions. They are created dynamically, one for each unique
        input the layer receives (from layer-one), with each node's ID then
        being that unique input (as a string).
        A population's expressions represent a "mask" applied to the layer's
        input as it passes through it.
        On init, each previously existing node is loaded from file iff PERSIST.
        Note: This layer is trained in an "online" fashion - as the agent runs,
        its model file is updated for every call to node.update() iff PERSIST.
    """

    def __init__(self, ID, size):
        """ Accepts:
            ID (str)            : This layer's unique ID
            id_prefix (str)     : Each node's ID prefix. Ex: 'Agent1_L2_'
        """
        self.ID = ID
        self._size = size       # This layer's size. I.e., it's input count
        self._curr_node = None  # The node for the current unique input
        self._nodes = {}        # Nodes, as: { nodeID: (obj_instance, output) }
        self._prev_inputs = []  # Previous input values, for recurrance
        self.id_prefix = ID + '_node_'

        f_save = "self._save('MODEL_FILE')"
        f_load = "self._load('MODEL_FILE')"
        self.model = ModelHandler(self, CONSOLE_OUT, PERSIST,
                                  model_ext=L2_EXT,
                                  save_func=f_save,
                                  load_func=f_load)

    def __str__(self):
        str_out = '\nID = ' + self.ID
        str_out += '\nNodes = ' + str(len(self._nodes))
        return str_out

    def _save(self, filename):
        """ Saves the layer to file. For use by ModelHandler.
        """
        # Write each node ID and asociated data as { "ID": ("save_string") }
        with open(filename, 'w') as f:
            f.write('{')
            for k, v in self._nodes.items():
                savestr = v[0].save()
                f.write('"' + k + '": """' + savestr + '""", ')
            f.write('}')

    def _load(self, filename):
        """ Loads the layer from file. For use by ModelHandler.
        """
        # Restore the layer nodes, one at a time
        self._nodes = {}
        i = 0

        with open(filename, 'r') as f:
            data = f.read()

        for k, v in eval(data).items():
            nodeID = self.id_prefix + str(i)
            node = Genetic(nodeID, 2, 0, 0, 0, 0, CONSOLE_OUT, False)
            node.load(v, not_file=True)
            self._nodes[k] = (node, None)
            i += 1

    def forward(self, inputs, is_seq=False):
        """ Returns the layer's output after moving the given inputs
            through it and setting the currently active node according to it
            (the node is created first, if it doesn't already exist).
            Accepts:
                inputs (list)
        """
        inputs_str = ''.join(inputs)
        if not self._nodes.get(inputs_str):
            # Init new node ("False", because we'll handle its persistence)
            sz = len(inputs)
            pop_sz = sz * 5
            tourny_sz = int(pop_sz * .25)
            node = Genetic(ID=inputs_str,  # data string is node ID
                           kernel=2,
                           max_pop=pop_sz,
                           max_depth=L2_MAX_DEPTH,
                           max_inputs=4,  # debug
                           tourn_sz=tourny_sz,
                           console_out=CONSOLE_OUT,
                           persist=False)
            self._nodes[inputs_str] = (node, None)
        else:
            node = self._nodes[inputs_str][0]

        self._curr_node = node
        return node.apply(inputs=list([inputs]), is_seq=is_seq)

    def update(self, fitness):
        """ Updates the currently active node w/ the given fitness data (dict).
        """
        self._curr_node.update(fitness)


class WeightedValues(object):
    """ A collection of weighted numeric values, by label.
    """

    def __init__(self):
        self._values = {}  # { label: [ value, weight ], ... }

    def __str__(self):
        str_out = ''
        for k, v in self._values.items():
            str_out += str(k) + ': ' + str(v) + '\n'
        return str_out[:-1]

    def __len__(self):
        return len(self._values.keys())

    def set(self, label, value=0, default_weight=1.0):
        """ Sets the value for the given label. If the label does not already
            exist, it is created with the given value and default_weight.
        """
        try:
            self._values[label][0] = value
        except KeyError:
            new_pair = [value, default_weight]
            self._values[label] = new_pair

    def adjust(self, label, value):
        """ Updates the given label's value by adding the given value to it.
        """
        try:
            self._values[label][0] += value
        except KeyError:
            print("ERROR: Attempted to adjust a non-existent label.")

    def set_wt(self, label, wt):
        """ Sets the given label's weight to the specified value.
        """
        try:
            self._values[label][1] = wt
        except KeyError:
            print("ERROR: Attempted to weight a non-existent label.")

    def set_wts(self, wts):
        """ Sets each weight according to the ordered list given, where
            wts[i] maps to label_i: weight
        """
        try:
            for k in self._values.keys():
                self._values[k][1] = wts.pop(0)
        except IndexError:
            print("ERROR: Mismatched len(wts) to len(values).")

    def keys(self):
        """ Returns a list of heuristics labels
        """
        return [label for label in self._values.keys()]

    def get(self, label):
        """ Returns the weighted value associated with specified label.
        """
        return self._values[label][0] * self._values[label][1]

    def get_list(self, normalize=True):
        """ Returns a new list of all weighted values.
        """
        ps = [v[0] * v[1] for v in self._values.values()]

        if normalize:
            normmax = max(ps)
            normmin = min(ps)
            ps = [(p - normmin) / (normmax - normmin) for p in ps]

        return ps
