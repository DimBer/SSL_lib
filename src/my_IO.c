#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/stat.h>

#include "my_IO.h"

#include "my_defs.h"

#include "my_utils.h"


//these are the possible methods

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
			      .multi_label = DEFAULT_MULTILABEL,	 
			      .no_constr = DEFAULT_UNCONSTRAINED,
			      .graph_filename = DEFAULT_GRAPH,
			      .label_filename = DEFAULT_LABEL,
			      .method = DEFAULT_METHOD,
			      .method_index = DEFAULT_METHOD_IND,
			      .single_thread = DEFAULT_SINGLE_THREAD };


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
		{"multilabel",   no_argument, 0,  'k' },
		{"single_thread",   no_argument, 0,  'l' }, 		 
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
			case 'j' : args->no_constr = 1;
				   break;		
			case 'k' : args->multi_label = 1;
				   break;	
   			case 'l' : args->single_thread = 1;
				   break;			
				   exit(EXIT_FAILURE);
		}
	}

}






//Print the two collumn array of edges
void print_edge_list(const uint64_t** edge_list, uint64_t len){
	uint64_t i;
	printf("EDGE LIST: \n");
	for(i=0;i<len;i++){
		printf("%"PRIu64"  %"PRIu64"\n",edge_list[i][0],edge_list[i][1]);
	}
}




//Print array (double)
void print_predictions(int8_t* arr, uint64_t N){
	uint64_t i;
	printf("Predicted labels: \n");
	for(i=0;i<N;i++){printf("%"PRId8"\n",arr[i]);}

}




// Take array of labels and return number of classes, list of classes and indicator vectors
uint8_t handle_labels(const int8_t* labels, uint16_t L, int8_t* class,
		      uint8_t* class_ind , uint16_t* num_per_class ){
	uint16_t i,j;
	uint8_t count;

	count= find_unique(class, labels, L);

	for(i=0;i<count;i++){
		num_per_class[i]=0;
		for(j=0;j<L;j++){
			if(labels[j]==class[i]){
				class_ind[i*L +j] = 1; 
				num_per_class[i]+=1;
			}else{class_ind[i*L +j] = 0;}
		}
	}

	return count;
}

//Take array of labels OR a one-hot-matrix and return number of classes, list of classes and indicator vectors
uint8_t abstract_handle_labels( uint16_t** num_per_class, uint8_t** class_ind, int8_t** class,
			        abstract_labels labels, uint16_t num_seeds){

        uint8_t num_class;

	if(labels.multi_label){
		if(num_seeds!=labels.mlabel.length){ 
			printf("Something wrong...size of input one_hot_mat != number of seeds\n");
			exit(EXIT_FAILURE); 
	 	}

		num_class = labels.mlabel.num_class;
		*num_per_class =  malloc(num_class*sizeof(uint16_t));
		*class =  malloc(num_class*sizeof(int8_t));
  		for(uint8_t i=0;i<num_class;i++) *(*class+i)=i; 
		*class_ind =  malloc(num_class*num_seeds*sizeof(uint8_t));
		
		for(uint8_t i=0;i<num_class;i++){
			*(*num_per_class + i)=0;
			for(uint16_t j=0;j<num_seeds;j++){ 
				*(*num_per_class + i) += labels.mlabel.bin[i][j];
				*(*class_ind + (i*num_seeds + j)) = labels.mlabel.bin[i][j];
				}
		}
		
 
	}else{
		*num_per_class =  malloc(num_seeds*sizeof(uint16_t));
		*class_ind =  malloc(num_seeds*num_seeds*sizeof(uint8_t));
		*class =  malloc(num_seeds*sizeof(int8_t));
	
		num_class = handle_labels(labels.mclass, num_seeds, *class, *class_ind, *num_per_class);

		*class = realloc(*class,num_class*sizeof(int8_t));
		*class_ind = realloc(*class_ind,num_class*num_seeds*sizeof(uint8_t));
		*num_per_class =realloc(*num_per_class, num_class*sizeof(uint16_t));		
	}

	return num_class;
}


//Pradict labels from largest soft label
void predict_labels( int8_t* label_out, double* soft_labels, int8_t* class,
	             uint64_t graph_size, uint8_t num_class ){
	uint64_t i;
	uint8_t j,max_ind;
	double max_val;

	for(i=0;i<graph_size;i++){
		max_val=-100.0f;
		for(j=0;j<num_class;j++){  
			if(max_val<soft_labels[i*num_class + j]){
				max_val=soft_labels[i*num_class + j];
				max_ind=j;}
		}
		label_out[i]=class[max_ind];
	}
}

//Pradict labels from largest soft label (Soft labels are transposed array)
void predict_labels_type2( int8_t* label_out, double* soft_labels, int8_t* class,
			   uint64_t graph_size, uint8_t num_class ){
	uint64_t j;
	uint8_t i,max_ind;
	double max_val;

	for(j=0;j<graph_size;j++){
		max_val=-100.0f;
		for(i=0;i<num_class;i++){  
			if(max_val<soft_labels[i*graph_size + j]){
				max_val=soft_labels[i*graph_size + j];
				max_ind=i;}	
		}
		label_out[j]=class[max_ind];		
	}
}















