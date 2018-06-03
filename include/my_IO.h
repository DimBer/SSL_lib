#ifndef my_IO_H_   
#define my_IO_H_

#include <stdbool.h>

#include "my_defs.h"


sz_short handle_labels(const class_t* , sz_med , class_t* , sz_short* , sz_med* );

sz_short abstract_handle_labels( sz_med** , sz_short** , class_t** , abstract_labels , sz_med);

void parse_commandline_args(int ,char**  , cmd_args* );

#endif
