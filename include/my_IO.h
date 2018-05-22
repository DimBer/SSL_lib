#ifndef my_IO_H_   
#define my_IO_H_

#include "my_defs.h"

void print_edge_list(const uint64_t** , uint64_t );

void print_predictions(int8_t* , uint64_t );

uint8_t handle_labels(const int8_t* , uint16_t , int8_t* , uint8_t* , uint16_t* );

uint8_t abstract_handle_labels( uint16_t** , uint8_t** , int8_t** , abstract_labels , uint16_t);

void parse_commandline_args(int ,char**  , cmd_args* );

void predict_labels( int8_t* , double* , int8_t* , uint64_t , uint8_t );

void predict_labels_type2( int8_t* , double* , int8_t* , uint64_t , uint8_t );

#endif
