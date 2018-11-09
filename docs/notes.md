
### Definitions
Context - 
Focus - 
Consciousness - 
Sensory Input - 
Intuition - black box, selects current context. May lead to "aha" moments or simply 



## What's next

* Add a temporal decay to the evolving string?
* Add a "recent connection beween two sub-ann's give'sadjacent ann's higher priority"
* The evoling string is a tensor of floats. Tee floatsrepresent connectoins between each sub-ann. i.e."adjacent anns"


 ## Prev model, w/ANN's at both ends
* Each queue, q_i, receives input from ANN a_i. The queue implementation represents the limited working memory of a human as each element in the queue is used as input to each ANN. A set of n FIFO queues of depth d.
* An additional set of input nodes for each ANN may eventually be explored, to represent short/long term working memory.
* The attenuator pops an element from each input-layer pipe, assigns it a weight, and the weighted elements to the output layer.
* The weight assigned is dictated by an evolutionary function implemented via a genetic algorithm. HEURISTIC??
* Expand intutive layer to include a shove queue representing working memeory
* Analyst - Accepts highest weighted input. What does it do with the input? Can it override the weights? We can't choose to focus on one context or another, rather, the environment dictates

Possible applications to:
	Each ANN is an i_th letter classifier: get attentive layer to find new words
	a_0 equals 0th digit, a_1 = 1th digit. Attentive layer discovers digits such as pi.
	maze navigation
	Pick out patterns from noise
	Give two forms of an equation and determine which is "better"
	Each ann is a mode of operation
	Agent decides some binary conceptual comparison-based decision
	threading process allocation - genetic alg determines optimal allocation


