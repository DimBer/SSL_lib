#ifndef MY_UTILS_H_   
#define MY_UTILS_H_

#include <stdbool.h>

#include "my_defs.h"

//various usdeful functions and routines


uint64_t rand_lim(uint64_t);

void random_sample( uint64_t* , abstract_labels , abstract_labels , uint16_t , uint64_t );

int compare ( const void* , const void* );

int compare2 ( const void* , const void* );

double frob_norm(double*,uint16_t);

double mean(double* , int );

void LUPSolve(double** , int* , double* , int , double*);

int LUPDecompose(double **, int , double , int *);

void matvec(double* , double* , double* , uint16_t , uint16_t  );

void matvec_trans(double* , double* , double* , uint16_t , uint16_t  );

void matvec_trans_long( double* , double* , double* , uint64_t , uint16_t );

void project_to_simplex( double* , uint16_t  );

double max_diff(double* , double* , uint16_t );

uint8_t find_unique(int8_t* , const int8_t* ,uint16_t );

uint64_t* find_unique_from_sorted( uint64_t* , uint64_t , uint16_t* );

void my_array_sub(double*, double* , double*, uint64_t );

void matrix_matrix_product(double*, double* , double* ,  uint64_t , uint16_t  , uint8_t );

void my_relative_sorting( uint64_t* , int8_t* , uint64_t );

uint64_t** zip_arrays(uint64_t* , uint64_t*, uint64_t );

void unzip_array(uint64_t** , uint64_t* , uint64_t* , uint64_t );

void rearange(uint64_t* , void* , char* , uint64_t );

uint64_t* remove_from_list(const uint64_t*, const uint64_t*,uint64_t,uint64_t);

void assert_all_nodes_present(csr_graph , const uint64_t* , uint16_t );

//IO tools

uint64_t**  give_edge_list( char* , uint64_t* );

uint64_t read_adjacency_to_buffer(uint64_t**,FILE*);

int8_t* read_labels(char* , uint64_t*);

uint64_t* read_seed_file( char*, uint16_t*,uint8_t* , abstract_labels* );		

void save_predictions(char* , abstract_label_output , uint64_t , uint8_t );

void predict_labels( int8_t* , double* , int8_t* ,uint64_t , uint8_t );

void predict_labels_type2( int8_t* , double* , int8_t* , uint64_t, uint8_t);

void print_array_1D(double* , uint64_t , uint64_t );

void print_edge_list(const uint64_t** , uint64_t );

void print_predictions(int8_t* , uint64_t );

int file_isreg( char* ); 

one_hot_mat read_one_hot_mat(char* , uint64_t* );


// classification and detection performance routines 

double accuracy(int8_t* , int8_t* , uint64_t* , uint64_t );

f1_scores get_averaged_f1_scores(one_hot_mat , one_hot_mat, uint64_t* , uint64_t);

void get_per_class_f1_scores(double* , detector_stats* , uint8_t );

one_hot_mat list_to_one_hot( uint64_t* , int8_t* , uint8_t , int8_t* ,  uint64_t   ,uint64_t );

one_hot_mat init_one_hot(uint8_t ,uint64_t );

void destroy_one_hot(one_hot_mat);

uint8_t* return_num_labels_per_node( one_hot_mat );

one_hot_mat top_k_mlabel( double* , uint8_t* , uint64_t, uint8_t );

detector_stats get_detector_stats( uint8_t* , uint8_t* , uint64_t, int* );

double harmonic_mean( double , double  );

classifier_stats  get_class_stats(detector_stats* , uint8_t );


#endif
