#ifndef MY_DEFS_H_   
#define MY_DEFS_H_

#include <stdbool.h>

#define DEBUG false
#define PRINT_THETAS false

//Input buffer sizes

#define EDGE_BUFF_SIZE 5000000
#define CLASS_BUFF_SIZE 5000000

//Default command line arguments

#define NUM_METHODS 4

#define DEFAULT_NUM_WALK 20

#define DEFAULT_TEL_PROB 0.05

#define DEFAULT_L_TRWR 1.0

#define DEFAULT_L_ADDF 15.0

#define DEFAULT_ITERS 1

#define DEFAULT_NUM_SEEDS 100

#define DEFAULT_UNCONSTRAINED false

#define DEFAULT_MULTILABEL false

#define DEFAULT_GRAPH "graphs/pubmed_adj.txt"

#define DEFAULT_LABEL "graphs/pubmed_label.txt"

#define DEFAULT_OUTFILE "out/label_predictions.txt"

#define DEFAULT_METHOD "AdaDIF"

#define DEFAULT_METHOD_IND 1

#define DEFAULT_SINGLE_THREAD false

#define DEFAULT_MODE "test"

//Default TunedRwR values

#define MAXIT 100
#define TOL 1.0e-8

//Default optimization parameters

#define GD_TOL 1.0e-3

#define GD_TOL_2 1.0e-3

#define STEPSIZE 0.1

#define STEPSIZE_2 0.95

#define MAXIT_GD 1000

#define PROJ_TOL 1.0e-4

#define LU_TOL 1.0e-6

#define L2_REG_LAMBDA 0.05


//INTEGER TYPES

typedef uint64_t sz_long; //Long range unsinged integer. Used for node and edge indexing.

typedef uint16_t sz_med; //Medium range unsinged integer. Used for random walk length, 
                         //iteration indexing and seed indexing

typedef uint8_t sz_short; //Short range unsigned integer. Used for class and thread indexing.

typedef int8_t class_t; // Short integer for actual label values.



//DATA STRUCTURES

//structs to store classification and detection performance statistics

typedef struct{
	double micro_precision;
	double micro_recall;
	double macro_precision;
	double macro_recall;
} classifier_stats;


typedef struct{
	double true_pos;
	double true_neg;
	double false_pos;
	double false_neg;	
} detector_stats;


// classification performance metrics

typedef struct{
	double micro;
	double macro;	
} f1_scores; 


//struct for for one-hot-type matrix

typedef struct{
	sz_short** bin;
	sz_short num_class;
	sz_long length;	
} one_hot_mat; 

// Abstract label formats (list or one-hot-matrix)

typedef struct{	
	bool is_multilabel; 
	class_t* mclass;	
	one_hot_mat mlabel;
} abstract_labels;


typedef struct{	
	bool is_multilabel;
	class_t* mclass;	
	double* mlabel;		
} abstract_label_output;

//Double and index struct for sorting and keeping indexes

typedef struct{
	double val;
	int ind;
} val_and_ind;

//struct forcommand line arguments

typedef struct{
	double lambda_trwr;
	double lambda_addf;
	double tel_prob;
	sz_med num_seeds;
	sz_med num_iters;
	char* graph_filename;
	char* label_filename;
	char* method;
	char* mode;
	char* outfile;
	sz_med walk_length;
	sz_short method_index;
	bool no_constr;
	bool is_multilabel;
	bool single_thread;	
} cmd_args; 



//Csr graph struct

typedef struct{
	double* csr_value;
	sz_long* csr_column;
	sz_long* csr_row_pointer;
	sz_long  num_nodes;
	sz_long  nnz;
	sz_long* degrees;
} csr_graph;



//Define structs that passes pointers and parameters into each thread
//Type 1 is for splitting the matrix power method in Tuned_RwR
//Type 2 is for splitting along different classes in AdaDIF

typedef struct {
	double* G_s; 
	double* G_s_next; 
	csr_graph graph;
	sz_long* seeds;	
	sz_med M;
	double tel_prob;
	sz_med* iter;
	sz_med from;
	sz_med to;
} pass_to_thread_type_1;

typedef struct {
	double* soft_labels;
	sz_short* class_ind;
	sz_med num_seeds;	 
	sz_med* num_per_class;
	sz_short from;
	sz_short to;
	sz_short num_local_classes;
	csr_graph graph;
	const sz_long* seeds;	
	sz_med walk_length;
	double lambda;
	bool no_constr;
}pass_to_thread_type_2;



#endif
