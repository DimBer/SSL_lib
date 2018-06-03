#ifndef MY_UTILS_H_   
#define MY_UTILS_H_

#include <stdbool.h>

#include "my_defs.h"

//various usdeful functions and routines


sz_long rand_lim(sz_long);

void random_sample( sz_long* , abstract_labels , abstract_labels , sz_med , sz_long );

int compare ( const void* , const void* );

int compare2 ( const void* , const void* );

double frob_norm(double*,sz_med);

double mean(double* , int );

void LUPSolve(double** , int* , double* , int , double*);

int LUPDecompose(double **, int , double , int *);

void matvec(double* , double* , double* , sz_med , sz_med  );

void matvec_trans(double* , double* , double* , sz_med , sz_med  );

void matvec_trans_long( double* , double* , double* , sz_long , sz_med );

void project_to_simplex( double* , sz_med  );

double max_diff(double* , double* , sz_med );

sz_short find_unique(class_t* , const class_t* ,sz_med );

sz_long* find_unique_from_sorted( sz_long* , sz_long , sz_med* );

void my_array_sub(double*, double* , double*, sz_long );

void matrix_matrix_product(double*, double* , double* ,  sz_long , sz_med  , sz_short );

void my_relative_sorting( sz_long* , class_t* , sz_long );

sz_long** zip_arrays(sz_long* , sz_long*, sz_long );

void unzip_array(sz_long** , sz_long* , sz_long* , sz_long );

void rearange(sz_long* , void* , char* , sz_long );

sz_long* remove_from_list(const sz_long*, const sz_long*,sz_long,sz_long);

void assert_all_nodes_present(csr_graph , const sz_long* , sz_med );

//IO tools

sz_long**  give_edge_list( char* , sz_long* );

sz_long read_adjacency_to_buffer(sz_long**,FILE*);

class_t* read_labels(char* , sz_long*);

sz_long* read_seed_file( char*, sz_med*,sz_short* , abstract_labels* );		

void save_predictions(char* , abstract_label_output , sz_long , sz_short );

void predict_labels( class_t* , double* , class_t* ,sz_long , sz_short );

void predict_labels_type2( class_t* , double* , class_t* , sz_long, sz_short);

void print_array_1D(double* , sz_long , sz_long );

void print_edge_list(const sz_long** , sz_long );

void print_predictions(class_t* , sz_long );

int file_isreg( char* ); 

one_hot_mat read_one_hot_mat(char* , sz_long* );


// classification and detection performance routines 

double accuracy(class_t* , class_t* , sz_long* , sz_long );

f1_scores get_averaged_f1_scores(one_hot_mat , one_hot_mat, sz_long* , sz_long);

void get_per_class_f1_scores(double* , detector_stats* , sz_short );

one_hot_mat list_to_one_hot( sz_long* , class_t* , sz_short , class_t* ,  sz_long   ,sz_long );

one_hot_mat init_one_hot(sz_short ,sz_long );

void destroy_one_hot(one_hot_mat);

sz_short* return_num_labels_per_node( one_hot_mat );

one_hot_mat top_k_mlabel( double* , sz_short* , sz_long, sz_short );

detector_stats get_detector_stats( sz_short* , sz_short* , sz_long, int* );

double harmonic_mean( double , double  );

classifier_stats  get_class_stats(detector_stats* , sz_short );


#endif
