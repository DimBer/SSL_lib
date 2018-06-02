#ifndef G_SLICE_EXTRACTION_H_   
#define G_SLICE_EXTRACTION_H_

#include "my_defs.h"
#include <stdbool.h>

double max_seed_val_difference(double*, double*, uint64_t* , uint16_t, uint16_t , uint16_t );  

uint16_t get_slice_of_G( double* , uint64_t* , uint16_t, double , csr_graph, bool);

void extract_G_ll(double* , double* , uint64_t* , uint16_t  );  

void* my_power_iter(void* );

void AdaDIF_core_multi_thread( double* , csr_graph , uint16_t, const uint64_t* , uint8_t , uint8_t* , uint16_t* , uint16_t , double, bool , bool );

void* AdaDIF_squezze_to_one_thread( void* );

void AdaDIF_core( double* , csr_graph , uint16_t, const uint64_t* , uint8_t , uint8_t*, uint16_t*   , uint16_t , double, bool );

void my_PPR_single_thread( double* , csr_graph , uint16_t , const uint64_t* , uint8_t , uint8_t* , uint16_t*, uint16_t , double );

void perform_random_walk(double* , double* , csr_graph  , uint16_t , uint64_t*  , uint16_t );

double* get_AdaDIF_parameters(uint64_t, uint64_t* , double* , double* ,const uint64_t*, uint64_t* , uint8_t* , uint16_t , uint16_t , uint16_t ,double,bool);

#endif
