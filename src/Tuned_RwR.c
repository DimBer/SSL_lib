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
#include <stdbool.h>

#include "Tuned_RwR.h"
#include "csr_handling.h"
#include "comp_engine.h"
#include "parameter_opt.h"
#include "my_IO.h"
#include "my_defs.h"
#include "my_utils.h"


sz_long Tuned_RwR( abstract_label_output* label_out , const sz_long** edge_list, sz_long num_edges,
		    const sz_long* seed_indices, abstract_labels labels , cmd_args args){
	
	sz_med num_seeds = args.num_seeds;
	double tel_prob = args.tel_prob;
	double lambda = args.lambda_trwr;	
	sz_long* seeds=malloc(num_seeds*sizeof(sz_long));

	for(sz_med i=0;i<num_seeds;i++) seeds[i] = seed_indices[i]-1;

	//Create CSR graph from edgelist 

	csr_graph graph = csr_create(edge_list,num_edges);
	assert_all_nodes_present(graph,seed_indices,num_seeds);

	//Normalize csr_value to column stochastic

	make_CSR_col_stoch(&graph);

	/////////////////////////////////////////////////////////////////////////////////////////////
	// HANDLE LABELS

	sz_short num_class; 
	sz_med* num_per_class;
	sz_short* class_ind;
	class_t* class;
	
	num_class = abstract_handle_labels( &num_per_class, &class_ind, &class, labels, num_seeds);

	/////////////////////////////////////////////////////////////////////////////////////////////

	sz_med iters; //Number of iters to extract G slice
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

	//prepare label output
	if(labels.is_multilabel){
		label_out->mlabel = (double*) malloc(graph.num_nodes*num_class*sizeof(double));
		for(sz_short i=0;i<num_class;i++){
			for(sz_long j=0;j<graph.num_nodes;j++)
				label_out->mlabel[i*graph.num_nodes + j] = soft_labels[j*num_class +i ];
		}
	}else{
		label_out->mclass = (class_t*) malloc(graph.num_nodes*sizeof(class_t));		
		predict_labels(label_out->mclass, soft_labels, class , graph.num_nodes, num_class );		
	}



	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;


	printf("Graph size: %"PRIu64"\n", (uint64_t) graph.num_nodes);

	printf("Num edges: %"PRIu64"\n", (uint64_t) graph.nnz);

	printf("Runtime: %lf\n ", time_spent);

	printf("Number of iterations: %"PRIu32"\n ", (uint32_t) iters);

        #if DEBUG
	double sum=0.0f;
	for(sz_long i=0;i<graph.num_nodes;i++){
		for(sz_short j=0;j<num_class;j++){
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










