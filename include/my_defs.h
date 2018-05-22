#ifndef MY_DEFS_H_   
#define MY_DEFS_H_


#define EDGE_BUFF_SIZE 10000000
#define CLASS_BUFF_SIZE 10000000



#define MAXIT 100
#define TOL 1.0e-8


#define NUM_METHODS 4

#define DEFAULT_NUM_WALK 15

#define DEFAULT_TEL_PROB 0.05

#define DEFAULT_L_TRWR 1.0

#define DEFAULT_L_ADDF 10.0

#define DEFAULT_ITERS 1

#define DEFAULT_NUM_SEEDS 100

#define DEFAULT_UNCONSTRAINED 0

#define DEFAULT_MULTILABEL 0



#define GD_TOL 1.0e-3

#define GD_TOL_2 1.0e-3

#define STEPSIZE 0.1

#define STEPSIZE_2 0.95

#define MAXIT_GD 1000

#define PROJ_TOL 1.0e-4

#define LU_TOL 1.0e-6

#define L2_REG_LAMBDA 0.05


#define DEBUG 0



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
	uint8_t** bin;
	uint8_t num_class;
	uint64_t length;	
} one_hot_mat; 

// Abstract label formats (list or one-hot-matrix)

typedef struct{	
	uint8_t multi_label;
	int8_t* mclass;	
	one_hot_mat mlabel;
} abstract_labels;


typedef struct{	
	uint8_t multi_label;
	int8_t* mclass;	
	double* mlabel;		
} abstract_label_output;




//struct forcommand line arguments

typedef struct{
	double lambda_trwr;
	double lambda_addf;
	double tel_prob;
	uint16_t num_seeds;
	uint16_t num_iters;
	char* graph_filename;
	char* label_filename;
	char* method;
	uint16_t walk_length;
	uint8_t method_index;
	uint8_t no_constr;
	uint8_t multi_label;	
} cmd_args; 



//Csr graph struct

typedef struct{
	double* csr_value;
	uint64_t* csr_column;
	uint64_t* csr_row_pointer;
	uint64_t  num_nodes;
	uint64_t  nnz;
	uint64_t* degrees;
} csr_graph;



//Define structs that passes pointers and parameters into each thread
//Type 1 is for splitting the matrix power method in Tuned_RwR
//Type 2 is for splitting along different classes in AdaDIF

typedef struct {
	double* G_s; 
	double* G_s_next; 
	csr_graph graph;
	uint64_t* seeds;	
	uint16_t M;
	double tel_prob;
	uint16_t* iter;
	uint16_t from;
	uint16_t to;
} pass_to_thread_type_1;

typedef struct {
	double* soft_labels;
	uint8_t* class_ind;
	uint16_t num_seeds;	 
	uint16_t* num_per_class;
	uint8_t from;
	uint8_t to;
	uint8_t num_local_classes;
	csr_graph graph;
	const uint64_t* seeds;	
	uint16_t walk_length;
	double lambda;
	int8_t no_constr;
}pass_to_thread_type_2;



#endif
