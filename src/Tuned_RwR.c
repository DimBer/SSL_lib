////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Implementation of Tuned Random-walks-with-restarts method for classification over graphs
 
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

//////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#include "csr_handling.h"
#include "comp_engine.h"
#include "parameter_opt.h"
#include "my_IO.h"
#include "my_defs.h"
#include "my_utils.h"


uint64_t Tuned_RwR( abstract_label_output label_out , const uint64_t** edge_list, uint64_t num_edges,
		    const uint64_t* seed_indices, abstract_labels labels , cmd_args args){
	
	uint16_t num_seeds = args.num_seeds;
	double tel_prob = args.tel_prob;
	double lambda = args.lambda_trwr;	
	uint64_t i;
	uint64_t* seeds=malloc(num_seeds*sizeof(uint64_t));


	for(i=0;i<num_seeds;i++){seeds[i]=seed_indices[i]-1;}

	//Create CSR graph from edgelist 

	csr_graph graph = csr_create(edge_list,num_edges);
	assert_all_nodes_present(graph,seed_indices,num_seeds);

	//Normalize csr_value to column stochastic

	make_CSR_col_stoch(&graph);

	/////////////////////////////////////////////////////////////////////////////////////////////
	// HANDLE LABELS

	uint8_t num_class; 
	uint16_t* num_per_class;
	uint8_t* class_ind;
	int8_t* class;
	
	num_class = abstract_handle_labels( &num_per_class, &class_ind, &class, labels, num_seeds);

	/////////////////////////////////////////////////////////////////////////////////////////////

	uint16_t iters; //Number of iters to extract G slice
	double* G_s=malloc(graph.num_nodes*num_seeds*sizeof(double));

	clock_t begin = clock();

	//Obtaining slice of G AND the square matrix G_LL 

	iters = get_slice_of_G( G_s, seeds, num_seeds, tel_prob, graph, args.single_thread);


	double* G_ll=malloc(num_seeds*num_seeds*sizeof(double));

	extract_G_ll(G_ll,G_s,seeds,num_seeds);


	//Multiply with parameter vectors

	double* theta=malloc(num_seeds*num_class*sizeof(double));

	double* soft_labels=malloc(graph.num_nodes*num_class*sizeof(double));

	tune_all_parameters(theta,G_ll,class_ind,num_seeds,num_class,lambda,num_per_class);

	matrix_matrix_product(soft_labels, G_s, theta, graph.num_nodes, num_seeds, num_class);


	if(labels.multi_label){
		label_out.mlabel = (double*) malloc(graph.num_nodes*num_class*sizeof(double));
		for(uint8_t i=0;i<num_class;i++){
			for(uint64_t j=0;j<graph.num_nodes;j++)
				label_out.mlabel[i*graph.num_nodes + j] = soft_labels[j*num_class +i ];
		}
	}else{
		label_out.mclass = (int8_t*) malloc(graph.num_nodes*sizeof(int8_t));		
		predict_labels(label_out.mclass, soft_labels, class , graph.num_nodes, num_class );		
	}



	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;


	printf("Graph size: %"PRIu64"\n", graph.num_nodes);

	printf("Num edges: %"PRIu64"\n", graph.nnz);

	printf("Runtime: %lf\n ", time_spent);

	printf("Number of iterations: %"PRIu16"\n ", iters);

        #if DEBUG
	double sum=0.0f;
	for(i=0;i<graph.num_nodes;i++){
		int j=0;
		for(;j<num_class;j++){
			printf("%lf  ",soft_labels[i*num_class + j]);
			sum+=soft_labels[i*num_class + j];
		}		
		printf("\n");
	}

	printf("Check sum: %lf \n",sum);
        #endif	

        //free
	free(G_s);
	free(G_ll);
	free(theta);
	free(soft_labels);
	free(num_per_class);
	free(class_ind);
	free(class);
	free(seeds);

	csr_destroy(graph);

	return graph.num_nodes;
}










