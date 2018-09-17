### Definitions
Context - 
Focus - 
Consciousness - 
Sensory Input - 
Intuition - black box, selects current context. May lead to "aha" moments or simply 

### Assumptions

Assumes consciousness to be an emergent property of the brain.
Assumes that conscious agents do not exert control over their current context, rather the current context is dictated by the current brain-body state according to current and previous sensory input and selected context.

## Challenges

Determine training/validation and problem to apply to

## Data 

Extracted w/Pandas
Why this set? Familiar with it. Deep learnable. No conv or pool layers

Data Set desc:
https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

## Learning variables

LEARNING_ITERATIONS
LEARNING_RATE
BIAS
BIAS_WEIGHT
NETWORK_LAYERS
NETWORK_LAYER_COUNT
TRAINING_DATAFILE
VALIDATION_DATAFILE

## Memory

 Each queue, q_i, receives input from ANN a_i. The queue implementation represents the limited working memory of a human !!!!!!as each element in the queue is used as input to each ANN. A set of n FIFO queues of depth d. 

### Analytical layer

The analytical layer examines the output-nodes of the attentive layer to draw concl
Analyst - Accepts highest weighted input. What does it do with the input? Can it override the weights? We can't choose to focus on one context or another, rather, the environment dictates

## Notes
The difference between this model and a single ANN is the extensibility of memory, and introduction of error into the connection-forming process. More recurrance
Note: With A arranged in this way, each ANN's output is dependent on current sensory input as well as the agent's current context.
Note: An additional set of input nodes for each ANN may eventually be explored, to represent short/long term working memory.
and the realization that moving from one context to the next is often (if not always) completely subconscious.
The attenuator pops an element from each input-layer pipe, assigns it a weight, and the weighted elements to the output layer. The weight assigned is dictated by an evolutionary function implemented via a genetic algorithm. HEURISTIC??
Expand intutive layer to include a shove queue representing working memeory
Why only 3 systems?
Further, it appears we can only go one level "up" - a person has the ability to think about themselves, but they cannot think about themselves thinking, therefore, there appears again to be two two primary "systems" at work - the data "filter", or "pattern recognizer" and the analytical agent.

It attempts to model as either contributing to homeostasis, or not

Apply to:
	Each ANN is an i_th letter classifier: get attentive layer to find new words
	a_0 equals 0th digit, a_1 = 1th digit. Attentive layer discovers digits such as pi.
	maze navigation
	Pick out patterns from noise
	Give two forms of an equation and determine which is "better"
	Each ann is a mode of operation
	Agent decides some binary conceptual comparison-based decision
IDEA: intution vs analytical system - genetic alg decides allocation between two systems.
IDEA: threading process allocation - genetic alg determines optimal allocation
IDEA: we can only focus on one context at a time, but how do we weight each one?
How are the inputs weighted? Biased by "amount" of input and log(type)? I.e. Fire vs. TV - Fire is big and new.
Context - an amalgamation of input from the five-senses input - we listen and see at the same time?
