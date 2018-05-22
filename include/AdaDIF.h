#ifndef ADADIF_H_   
#define ADADIF_H_

#include"my_defs.h"


uint64_t AdaDIF(abstract_label_output* , const uint64_t** , uint64_t , const uint64_t* , abstract_labels, uint16_t , uint16_t , double, int8_t );

uint64_t my_PPR( abstract_label_output* , const uint64_t** , uint64_t , const uint64_t* , abstract_labels  , uint16_t , uint16_t , double);

#endif
