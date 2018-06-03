///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains SSL program that implements and tests graph-based semi-spurevised methods.

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

#include "Tuned_RwR.h"
#include "AdaDIF.h"
#include "my_IO.h"
#include "my_defs.h"
#include "my_utils.h"


int SSL_predict( cmd_args );
int SSL_test( cmd_args );

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//MAIN
int main( int argc, char **argv )
{
	cmd_args args;
	
	//Parse arguments using argument parser

	parse_commandline_args(argc,argv,&args);

	// Switch between operational (prediction) mode and test mode
	if(strcmp(args.mode,"test")==0){
		int err = SSL_test(args);
		if(!err) printf( "SSL in test mode finished succesfully.\n" );
	}else if(strcmp(args.mode,"predict") == 0){
		int err = SSL_predict(args);
		if(!err) printf( "SSL in prediction mode finished succesfully.\n" );		
	}else{
	   	printf("ERROR: Mode must be either `test` (default) or `predict`.\n");
	   	exit(EXIT_FAILURE); 
	}
	
	return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




//SSL in __operational__ mode. 
//Produces output (predicted labels) on an output file 
int SSL_predict(cmd_args args){

	printf("SSL in operational mode.\n");
			
	//Parse graph
	printf("Reading edgelist file...\n");
	sz_long edge_count;
	sz_long** edge_list;
	edge_list =  give_edge_list( args.graph_filename, &edge_count );
	
	//Parse labels
	printf("Reading seed file...\n");
	abstract_labels label_in;
	label_in.is_multilabel=args.is_multilabel;
	sz_long* seeds;
	sz_short num_class;
	seeds = read_seed_file( args.label_filename, &args.num_seeds, &num_class, &label_in );		
	printf("Number of seed nodes:%"PRIu32" \n", (uint32_t) args.num_seeds);
	
	//Call method		
	sz_long graph_size;
	abstract_label_output label_out;	
	label_out.is_multilabel=args.is_multilabel;	
	if(args.method_index==0){
		printf("Executing Tunded_RwR...\n");
		graph_size=Tuned_RwR( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds, 
				     label_in , args);		
	}else if(args.method_index==1){
                printf("Executing AdaDIF...\n");
		graph_size=AdaDIF( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds,
				   label_in , args);	        		
	}else if(args.method_index==2){
                printf("AdaDIF_LOO not ready yet\n");
	}else{
                printf("Executing PPR...\n");
		graph_size=my_PPR( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds, 
				   label_in , args);			
	}			
	
	//write predictions to output file
	printf("Writing predictions...\n");
	save_predictions(args.outfile, label_out, graph_size, num_class);
	
	//free buffers
	for(sz_long i=0;i<edge_count;i++) free(edge_list[i]);
	free(edge_list);
	free(seeds);
        if(args.is_multilabel){
        	destroy_one_hot(label_in.mlabel);
        	free(label_out.mlabel);
        }else{
		free(label_in.mclass);
		free(label_out.mclass);        	
        }
	
	return 0;
}




//SSL in __test__ mode.
//Takes as input all the groundtruth labels.
//Randomly samples seeds and evaluates predictions.
int SSL_test(cmd_args args)
{
	double average_micro_f1=0.0, average_macro_f1=0.0; 
	sz_med iter=0;
	
	//Parse graph
	sz_long edge_count;
	sz_long** edge_list;
	edge_list =  give_edge_list(args.graph_filename,&edge_count);

	//Parse labels	
	abstract_labels label_in;
	abstract_labels all_labels;
	abstract_label_output label_out;
	
	label_in.is_multilabel=args.is_multilabel;
	label_out.is_multilabel=args.is_multilabel;
	all_labels.is_multilabel=args.is_multilabel;
	
	sz_long label_count;
	sz_short* num_labels_per_node;
	
	if(args.is_multilabel){
		all_labels.mlabel = read_one_hot_mat(args.label_filename, &label_count); // All true labels in one-hot-matrix form 
		label_in.mlabel = init_one_hot( all_labels.mlabel.num_class , (sz_long) args.num_seeds); 
		num_labels_per_node = return_num_labels_per_node( all_labels.mlabel );
	}else{
		all_labels.mclass = read_labels(args.label_filename, &label_count);  // All true labels in list form 
                label_in.mclass = (class_t*) malloc(args.num_seeds*sizeof(class_t));	
	}
	

	//RUN EXPERIMENTS
	sz_long* seeds=malloc(args.num_seeds*sizeof(sz_long));
	sz_long default_ind[label_count];
	for(sz_long i=0;i<label_count;i++) default_ind[i]=i;

 	srand(time(NULL)); //seed the random number generator

	for(iter=0;iter<args.num_iters;iter++){

		random_sample(seeds, label_in, all_labels, args.num_seeds, label_count);			
			
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		sz_long graph_size;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// THIS IS WHERE I CALL THE METHOD THAT I WANT TO TEST
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(args.method_index==0){
			printf("Execution %"PRIu32" of Tunded_RwR...", (uint32_t) iter);
			graph_size=Tuned_RwR( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds, 
					     label_in , args);		
		}else if(args.method_index==1){
                        printf("Execution %"PRIu32" of AdaDIF...", (uint32_t) iter);
			graph_size=AdaDIF( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds,
					   label_in , args);	        		
		}else if(args.method_index==2){
                        printf("AdaDIF_LOO not ready yet\n");
		}else{
                        printf("Execution %"PRIu32" of PPR...", (uint32_t) iter);
			graph_size=my_PPR( &label_out, (const sz_long**)edge_list, edge_count, (const sz_long*) seeds, 
					   label_in , args);			
		}		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(graph_size!=label_count){printf("ERROR: graph size not matching label size\n");}

	        // Evaluate predictions
   		sz_long num_unlabeled = label_count - args.num_seeds;           
   		for(sz_med i=0;i<args.num_seeds;i++) seeds[i]-=1;     
                sz_long* unlabeled = remove_from_list( (const sz_long*) default_ind ,
                				        (const sz_long*) seeds ,label_count , (sz_long) args.num_seeds );

		one_hot_mat true_one_hot,pred_one_hot;

		if(args.is_multilabel){	
			pred_one_hot = top_k_mlabel( label_out.mlabel , num_labels_per_node, graph_size, label_in.mlabel.num_class);
			true_one_hot = all_labels.mlabel;
		}else{
			class_t class[args.num_seeds];
			sz_short num_class = find_unique( class, (const class_t*) label_in.mclass, args.num_seeds );			
			pred_one_hot = list_to_one_hot( default_ind , label_out.mclass , num_class, class, label_count , label_count); 		
			true_one_hot = list_to_one_hot( default_ind , all_labels.mclass, num_class, class, label_count , label_count); 		
 		}
	
		f1_scores scores = get_averaged_f1_scores(true_one_hot, pred_one_hot, unlabeled, num_unlabeled );

		average_micro_f1 += scores.micro;				
		average_macro_f1 += scores.macro;		

		//free temporary arrays
		destroy_one_hot(pred_one_hot);
		if(args.is_multilabel==0) destroy_one_hot(true_one_hot);
		free(unlabeled);
	}

	//Print results
	average_micro_f1/=(double)args.num_iters;
	average_macro_f1/=(double)args.num_iters;

	printf(" Mean F1 micro: %lf\n Mean F1 macro: %lf\n ", average_micro_f1, average_macro_f1 );

	//free buffers
	for(sz_long i=0;i<edge_count;i++) free(edge_list[i]);
	free(edge_list);
	free(seeds);
        if(args.is_multilabel){
        	destroy_one_hot(label_in.mlabel);
        	destroy_one_hot(all_labels.mlabel);
        	free(label_out.mlabel);
        	free(num_labels_per_node);
        }else{
		free(label_in.mclass);
		free(all_labels.mclass);
		free(label_out.mclass);        	
        }


	return 0;
}












