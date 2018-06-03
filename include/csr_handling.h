#ifndef CSR_HANDLING_H_   
#define CSR_HANDLING_H_

#include "my_defs.h"

sz_long edge_list_to_csr(const sz_long** , double* , sz_long* , sz_long* , sz_long , sz_long*, sz_long* );

void make_CSR_col_stoch(csr_graph*);

csr_graph csr_create( const sz_long** , sz_long);

csr_graph csr_deep_copy_and_scale(csr_graph, double );

csr_graph* csr_mult_deep_copy( csr_graph, sz_short );

csr_graph csr_deep_copy(csr_graph);

void csr_destroy( csr_graph );

void csr_array_destroy( csr_graph* , sz_short);

void my_CSR_matmat( double* Y ,double* X  , csr_graph , sz_med , sz_med , sz_med); 

void my_CSR_matvec( double* ,double* ,csr_graph);

#endif
