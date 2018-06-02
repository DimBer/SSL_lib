////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains interfaces for (multi-threaded) Adaptive Diffusions (AdaDIF) method,
 and for (single-threaded) Personalize-PageRank 
 
 INPUT:  1) Graph as edgelist
 	 
 	 2) Seed indexes
 	 
 	 3) Seed labels as abstract_label type (one-hot-mat or list of labels depending on 
                        		        multi-class or multi-label; see defs.h for more)	 
 
 	 4) Algorithm parameters and options
 
 OUTPUT: 1) Writes label predictions as abstract_label type 
	 2) Returns number of nodes (usefull for cross-checking with)	

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

#include "AdaDIF.h"
#include "csr_handling.h"
#include "comp_engine.h"
#include "parameter_opt.h"
#include "my_IO.h"
#include "my_defs.h"
#include "my_utils.h"

// Adaptive - Diffusions method
uint64_t AdaDIF( abstract_label_output* label_out , const uint64_t** edge_list, uint64_t num_edges,
		 const uint64_t* seed_indices, abstract_labels labels , cmd_args args )
{
		
	uint16_t num_seeds = args.num_seeds;
	uint16_t walk_length = args.walk_length;
	double lambda = args.lambda_addf;
	uint8_t no_constr = args.no_constr;	

	uint64_t* seeds=malloc(num_seeds*sizeof(uint64_t));	
	for(uint16_t i=0;i<num_seeds;i++) seeds[i]=seed_indices[i]-1;

        //Create CSR graph from edgelist 

        csr_graph graph = csr_create(edge_list,num_edges);

	assert_all_nodes_present(graph,seed_indices,num_seeds);

	//Normalize csr_value to column stochastic

	make_CSR_col_stoch(&graph);


	///////////////////////////////////////////////////////////////////////////////////////////////
	// HANDLE LABELS

	uint8_t num_class; 
	uint16_t* num_per_class;
	uint8_t* class_ind;
	int8_t* class;

	num_class = abstract_handle_labels( &num_per_class, &class_ind, &class, labels, num_seeds);


	///////////////////////////////////////////////////////////////////////////////////////////////


	clock_t begin = clock();

	double* soft_labels=malloc(num_class*graph.num_nodes*sizeof(double));


        AdaDIF_core_multi_thread( soft_labels, graph, num_seeds, seeds, num_class, class_ind,
        			   num_per_class, walk_length, lambda, no_constr, args.single_thread);
        
        //prepare label output	
	if(labels.is_multilabel){
		label_out->mlabel = (double*) malloc(graph.num_nodes*num_class*sizeof(double));
		memcpy(label_out->mlabel, soft_labels, graph.num_nodes*num_class*sizeof(double));		
	}else{
		label_out->mclass = (int8_t*) malloc(graph.num_nodes*sizeof(int8_t));
		predict_labels_type2(label_out->mclass, soft_labels, class, graph.num_nodes, num_class );
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Graph size: %"PRIu64"\n", graph.num_nodes);

	printf("Num edges: %"PRIu64"\n", graph.nnz);

	printf("Runtime: %lf\n ", time_spent);

	#if DEBUG          
	double sum=0.0f;
	for(uint64_t i=0;i<graph.num_nodes;i++){
		for(int j=0;j<num_class;j++)
			sum+=soft_labels[j*graph.num_nodes + i];	
	}
	printf("Check sum: %lf \n",sum);
  	        
	printf("\n");
	for(int j=0;j<num_class;j++) printf("%"PRId8", ",class[j]);
	printf("\n");
	#endif		
	
	//free buffers
	free(soft_labels);
	free(num_per_class);
	free(class_ind);
	free(class);
	free(seeds);
        csr_destroy(graph);

	return graph.num_nodes;
}





//Personalized Pagerank method
uint64_t my_PPR( abstract_label_output* label_out , const uint64_t** edge_list, uint64_t num_edges, 
		const uint64_t* seed_indices, abstract_labels labels, cmd_args args )
{
	
	
	uint16_t num_seeds = args.num_seeds;
	uint16_t walk_length = args.walk_length;
	double tel_prob = args.tel_prob;

	uint64_t* seeds=malloc(num_seeds*sizeof(uint64_t));	
	for(uint64_t i=0;i<num_seeds;i++) seeds[i]=seed_indices[i]-1;

        //Create CSR graph from edgelist 

        csr_graph graph = csr_create(edge_list,num_edges);
	assert_all_nodes_present(graph,seed_indices,num_seeds);

	//Normalize csr_value to column stochastic

	make_CSR_col_stoch(&graph);

	//////////////////////////////////////////////////////////////////////////////////////////////
	// HANDLE LABELS

	uint8_t num_class; 
	uint16_t* num_per_class;
	uint8_t* class_ind;
	int8_t* class;
	
	num_class = abstract_handle_labels(&num_per_class, &class_ind, &class, labels, num_seeds);


	//////////////////////////////////////////////////////////////////////////////////////////////


	clock_t begin = clock();

	double* soft_labels=malloc(num_class*graph.num_nodes*sizeof(double));
	

        my_PPR_single_thread( soft_labels, graph, num_seeds, seeds, num_class,
        		      class_ind, num_per_class, walk_length, tel_prob);
        
	//prepare label output
	if(labels.is_multilabel){
		label_out->mlabel = (double*) malloc(graph.num_nodes*num_class*sizeof(double));
		memcpy(label_out->mlabel, soft_labels, graph.num_nodes*num_class*sizeof(double));		
	}else{
		label_out->mclass = (int8_t*) malloc(graph.num_nodes*sizeof(int8_t));		
		predict_labels_type2(label_out->mclass, soft_labels, class , graph.num_nodes, num_class );
	}

	

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Graph size: %"PRIu64"\n", graph.num_nodes);

	printf("Num edges: %"PRIu64"\n", graph.nnz);

	printf("Runtime: %lf\n ", time_spent);
	
	#if DEBUG          
	double sum=0.0f;
	for(uint64_t i=0;i<graph.num_nodes;i++){
		for(int j=0;j<num_class;j++)
			sum+=soft_labels[j*graph.num_nodes + i];
	}
	printf("Check sum: %lf \n",sum);
  	        
	printf("\n");
	for(int j=0;j<num_class;j++) printf("%"PRId8", ",class[j]);
	printf("\n");
	#endif	
	
	//free buffers
	free(soft_labels);
	free(num_per_class);
	free(class_ind);
	free(class);
	free(seeds);
        csr_destroy(graph);

	return graph.num_nodes;
}








