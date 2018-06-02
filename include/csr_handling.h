#ifndef CSR_HANDLING_H_   
#define CSR_HANDLING_H_

#include "my_defs.h"

uint64_t edge_list_to_csr(const uint64_t** , double* , uint64_t* , uint64_t* , uint64_t , uint64_t*, uint64_t* );

void make_CSR_col_stoch(csr_graph*);

csr_graph csr_create( const uint64_t** , uint64_t);

csr_graph csr_deep_copy_and_scale(csr_graph, double );

csr_graph* csr_mult_deep_copy( csr_graph, uint8_t );

csr_graph csr_deep_copy(csr_graph);

void csr_destroy( csr_graph );

void csr_array_destroy( csr_graph* , uint8_t);

void my_CSR_matmat( double* Y ,double* X  , csr_graph , uint16_t , uint16_t , uint16_t); 

void my_CSR_matvec( double* ,double* ,csr_graph);

#endif
