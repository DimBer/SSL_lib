

Programm __SSL__ implements experiments for different __semi-supervised learing__ methods on __multiclass
or multilabel__ graphs with available groundtruth labels. 


Input files format.

__SSL__ loads the graph in __adjacency list__ format from a .txt file that contains edges as tab separated pairs of node indexes in the format: node1_index \tab node2_index. Node indexes should be in range [1 , 2^64 ]. 

For __multiclass__ graphs, the labels are loaded from a .txt file where each line is of the format: node_index \tab label ( for now node_indexes have to be sorted but I will allow for any indexing sequence). Labels have to be integers in [-127,127]. 

For __multilabel__ graphs, labels are loaded from a txt file in compressed __one-hot-matrix__ form (see graphs/HomoSapiens/class.txt for example).


Command line optional arguments with values:

ARGUMENT | VALUES | DEFAULT
-------- | ------ | -------
--method |  Tuned_RwR <br/> AdaDIF <br/> AdaDIF_LOO <br/> PPR <br/> HK | Tuned_RwR
	   	
--graph_file | (edges).txt | graphs/cora_adj.txt

--label_file | (labels).txt | graphs/cora_label.txt

--num_seeds | [1, 2^16] | 100

--walk_length | [1, 2^16] | 20

--lambda_trwr | >=0.0 | 1.0

--lambda_addf | >=0.0 | 50.0

--lambda_loo | >=0.0 |     

--num_iters | [1, 2^16] | 1




Command line optional arguments without values:

ARGUMENT | RESULT
-------- | ------
--unconstrained | switches AdaDIF and AdaDIF_LOO to unconstrained mode

--single_thread | forces single thread execution

--multilabel | specifies multilabel input / output















