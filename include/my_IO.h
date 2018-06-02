#ifndef my_IO_H_   
#define my_IO_H_

#include <stdbool.h>

#include "my_defs.h"


uint8_t handle_labels(const int8_t* , uint16_t , int8_t* , uint8_t* , uint16_t* );

uint8_t abstract_handle_labels( uint16_t** , uint8_t** , int8_t** , abstract_labels , uint16_t);

void parse_commandline_args(int ,char**  , cmd_args* );

#endif
