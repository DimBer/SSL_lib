////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains high-level routines for parsing the command line for arguments,
 and handling multiclass or multilabel input.
 

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

///////////////////////////////////////////////////////////////////////////////////////////////


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "my_IO.h"
#include "my_defs.h"
#include "my_utils.h"


//List of possible methods

const char *method_list[NUM_METHODS]={"Tuned_RwR", "AdaDIF", "AdaDIF_LOO","PPR"};

//Parsing command line arguments with getopt_long_only
void parse_commandline_args(int argc,char** argv , cmd_args* args){

	//set default arguments
	
	(*args) = (cmd_args) {.lambda_trwr = DEFAULT_L_TRWR,
	                      .lambda_addf = DEFAULT_L_ADDF,
			      .walk_length = DEFAULT_NUM_WALK,
			      .num_iters   = DEFAULT_ITERS,
			      .num_seeds   = DEFAULT_NUM_SEEDS,
			      .tel_prob    = DEFAULT_TEL_PROB,	
			      .is_multilabel = DEFAULT_MULTILABEL,	 
			      .no_constr = DEFAULT_UNCONSTRAINED,
			      .graph_filename = DEFAULT_GRAPH,
			      .label_filename = DEFAULT_LABEL,
			      .method = DEFAULT_METHOD,
			      .method_index = DEFAULT_METHOD_IND,
			      .single_thread = DEFAULT_SINGLE_THREAD,
			      .mode = DEFAULT_MODE,
			      .outfile = DEFAULT_OUTFILE };


	int opt= 0;
	//Specifying the expected options
	//The two options l and b expect numbers as argument
	static struct option long_options[] = {
		{"graph_file",    required_argument, 0,  'a' },
		{"label_file",   required_argument, 0,  'b' },
		{"num_iters",   required_argument, 0,  'c' },
		{"num_seeds",   required_argument, 0,  'd' },        
		{"tel_prob",   required_argument, 0,  'e' },        
		{"lambda_trwr",   required_argument, 0,  'f' }, 
		{"method",   required_argument, 0,  'g' }, 		
		{"lambda_addf",   required_argument, 0,  'h' }, 
		{"walk_length",   required_argument, 0,  'i' }, 		
		{"unconstrained",   no_argument, 0,  'j' },
		{"multiclass",   no_argument, 0,  'k' },
		{"single_thread",   no_argument, 0,  'l' }, 		
		{"mode", required_argument, 0 , 'm' }, 
		{"outfile", required_argument, 0 , 'n' }, 		
		{0,           0,                 0,  0   }	
	};

	int long_index =0;
	
	
	while ((opt = getopt_long_only(argc, argv,"", 
					long_options, &long_index )) != -1) {
		switch (opt) {
			case 'a' : args->graph_filename = optarg;
				   if(file_isreg(args->graph_filename)!=1){ 
				   	printf("ERROR: %s does not exist\n",args->graph_filename);
				   	exit(EXIT_FAILURE); 
				   	}
				   break;
			case 'b' : args->label_filename = optarg;
				   if(file_isreg(args->label_filename)!=1){ 
				   	printf("ERROR: %s does not exist\n",args->label_filename);
				   	exit(EXIT_FAILURE); 
				   	}
				   break;
			case 'c' : args->num_iters = atoi(optarg); 
				   if(args->num_iters<1){ 
				   	printf("ERROR: Number of experiments must be positive\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;
			case 'd' : args->num_seeds = atoi(optarg);
				   if(args->num_seeds<1){ 
				   	printf("ERROR: Number of seedss must be positive\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;
			case 'e' : args->tel_prob = atof(optarg);
				   if((args->tel_prob<0.01f)||(args->tel_prob>0.99f)){ 
				   	printf("ERROR: Teleporation probability must be in the range [0.01 - 0.99]\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;
			case 'f' : args->lambda_trwr = atof(optarg);
				   if(args->lambda_trwr<0.0f){ 
				   	printf("ERROR: Regularization parameter lambda_trwr cannot be negative\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;	
			case 'g' : args->method = optarg;
				   int i,method_found=0; 
                                   for(i=0;i<NUM_METHODS;i++){
                                   	if(strcmp(method_list[i],args->method)==0)
                                   	       args->method_index=i;
                                   		method_found=1;
                                   }
				   if(method_found==0){ 
				   	printf("ERROR: Method %s does not exist \n",args->method);
				   	exit(EXIT_FAILURE); 
				   	}                                   
				   break;
			case 'h' : args->lambda_addf = atof(optarg);
				   if(args->lambda_addf<0.0f){ 
				   	printf("ERROR: Regularization parameter lambda_addf cannot be negative\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;				
			case 'i' : args->walk_length = atoi(optarg);
				   if(args->walk_length<1){ 
				   	printf("ERROR: Length of walks must be >=1\n");
				   	exit(EXIT_FAILURE); 
				   	}
				   break;				   	   
			case 'j' : args->no_constr = true;
				   break;		
			case 'k' : args->is_multilabel = false;
				   break;	
   			case 'l' : args->single_thread = true;
				   break;			
			case 'm' : args->mode = optarg;
				   break;
			case 'n' : args->outfile = optarg;
				   break;
				   
				   exit(EXIT_FAILURE);
		}
	}

}


// Take array of labels and return number of classes, list of classes and indicator vectors
sz_short handle_labels(const class_t* labels, sz_med L, class_t* class,
		      sz_short* class_ind , sz_med* num_per_class ){
		      	   
	sz_short count;

	count= find_unique(class, labels, L);

	for(sz_med i=0;i<count;i++){
		num_per_class[i]=0;
		for(sz_med j=0;j<L;j++){
			if(labels[j]==class[i]){
				class_ind[i*L +j] = 1; 
				num_per_class[i]+=1;
			}else{class_ind[i*L +j] = 0;}
		}
	}

	return count;
}

//Take array of labels OR a one-hot-matrix and return number of classes, list of classes and indicator vectors
sz_short abstract_handle_labels( sz_med** num_per_class, sz_short** class_ind, class_t** class,
			        abstract_labels labels, sz_med num_seeds){

        sz_short num_class;

	if(labels.is_multilabel){
		if(num_seeds!=labels.mlabel.length){ 
			printf("Something wrong...size of input one_hot_mat != number of seeds\n");
			exit(EXIT_FAILURE); 
	 	}

		num_class = labels.mlabel.num_class;
		*num_per_class =  malloc(num_class*sizeof(sz_med));
		*class =  malloc(num_class*sizeof(class_t));
  		for(sz_short i=0;i<num_class;i++) *(*class+i)=i; 
		*class_ind =  malloc(num_class*num_seeds*sizeof(sz_short));
		
		for(sz_short i=0;i<num_class;i++){
			*(*num_per_class + i)=0;
			for(sz_med j=0;j<num_seeds;j++){ 
				*(*num_per_class + i) += labels.mlabel.bin[i][j];
				*(*class_ind + (i*num_seeds + j)) = labels.mlabel.bin[i][j];
				}
		}
	}else{
		*num_per_class =  malloc(num_seeds*sizeof(sz_med));
		*class_ind =  malloc(num_seeds*num_seeds*sizeof(sz_short));
		*class =  malloc(num_seeds*sizeof(class_t));
	
		num_class = handle_labels(labels.mclass, num_seeds, *class, *class_ind, *num_per_class);

		*class = realloc(*class,num_class*sizeof(class_t));
		*class_ind = realloc(*class_ind,num_class*num_seeds*sizeof(sz_short));
		*num_per_class =realloc(*num_per_class, num_class*sizeof(sz_med));		
	}

	return num_class;
}
















