#ifndef G_SLICE_EXTRACTION_H_   
#define G_SLICE_EXTRACTION_H_

#include "my_defs.h"
#include <stdbool.h>

double max_seed_val_difference(double*, double*, sz_long* , sz_med, sz_med , sz_med );  

sz_med get_slice_of_G( double* , sz_long* , sz_med, double , csr_graph, bool);

void extract_G_ll(double* , double* , sz_long* , sz_med  );  

void* my_power_iter(void* );

void AdaDIF_core_multi_thread( double* , csr_graph , sz_med, const sz_long* , sz_short , sz_short* , sz_med* , sz_med , double, bool , bool );

void* AdaDIF_squezze_to_one_thread( void* );

void AdaDIF_core( double* , csr_graph , sz_med, const sz_long* , sz_short , sz_short*, sz_med*   , sz_med , double, bool );

void my_PPR_single_thread( double* , csr_graph , sz_med , const sz_long* , sz_short , sz_short* , sz_med*, sz_med , double );

void perform_random_walk(double* , double* , csr_graph  , sz_med , sz_long*  , sz_med );

double* get_AdaDIF_parameters(sz_long, sz_long* , double* , double* ,const sz_long*, sz_long* , sz_short* , sz_med , sz_med , sz_med ,double,bool);

#endif
