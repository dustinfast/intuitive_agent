
### Definitions
Context - 
Focus - 
Consciousness - 
Sensory Input - 
Intuition - black box, selects current context. May lead to "aha" moments or simply 



## What's next
 rim mutation for paring down large trees when acc is low
Change DataFrom to look at rightmost col for labels
L2.node_map[].weight (logarithmic decay over time/frequency)
* L2.node_map[].kb/correct/solution strings
* REPL vs. Flask interface?
* GP auto-tuna - mutation ratios, pop sizes, etc
* Adapt ann.py to accept dataloader and use MNIST (or similiar)
* Refactor all save/load funcs into ModelHandler.get_savestring?
* "provides feedback" connector - ex: True if the action returns a value
* Seperate log output option and persist
* Add heuristics output to log handler
* L3 kb save/load w/decaying fitness if already seen to encourage newness
* Expand l2 nodes - one for each sub-class (ex: py func, py kwd, etc) as
* Expand l2 nodes - as soon as some local mimima reached
* L2 - One node per input, but only the first dimension
* Add a temporal decay to the evolving string?
* Add a "recent connection beween two sub-ann's give'sadjacent ann's higher priority"
* The evoling string is a tensor of floats. Tee floatsrepresent connectoins between each sub-ann. i.e."adjacent anns"

Why only 3 systems? – Can we use two under one umbrella system? After all, we can only go one layer up (it appears we can only go one level "up" - a person has the ability to think about themselves, but they cannot think about themselves thinking, therefore, there appears again to be two two primary "systems" at work - the data "filter", or "pattern recognizer" and the analytical agent.)

Bootstrapping:
•	Each sub-agent represents a single context. Example:	
o	2 agents = py kwd and py func
o	3 agents = first two plus is_python
Branching/Bootstrapping:
1.	Branch into a hierarichal structure of agents (faciliated by reflection) – each representing one context. Contexts are not necessarily predefined.
2.	L2 branch after some number of increases of the last x iters
3.	Each agent represents a single "concept" such as a letter, or a python kwd
4.	If a branch agent does not learn enough over some t, rm it (log to kb?) - it's inputs do not form any concepts
5.	One node has one goal, a clique has another goal, a network has an even still more encompassing goal (ex: find free memory?)
L2 Tuning:
1.	Monitor accuracy heuristics over time -
1.	increase max pop size after some accuracy threshold
2.	decrease max pop size if proprtion of unfit to fit outputs calls for it
More heuristics:
•	What heuristics might drive it and its exploration - neophilia? Self-actualization? Guilt?
•	We encounter many heuristics in life
•	OBSERVATION: We are not always compelled to act rationally - in addition to logical reasoning, emotional (irrational) reasoning strongly influences our behavior. 

The heuristics guiding our intuition’s evolutionary journey may be associated with irrational drivers such as guilt, loneliness, boredom, etc. 
o	Innate drive based on environmental queues/heuristics comples us toward the new, and 
o	How are the inputs wighted? Biased by "amount" of input and log(type)? I.e. Fire vs. TV - Fire is big and new.
o	What drives the shifts?
o	The genetic alg decides attentive allocation between input data elements. Heuristics?

•	In addition, noting the hierarchal nature of information, the agent was designed to scale from a single agent, to a node-agent in a network of such agents, thereby facilitating the bootstrapping of an increasingly advanced intuition.
•	



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


