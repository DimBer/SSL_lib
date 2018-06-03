#ifndef PARAMETER_OPT_H_   
#define PARAMETER_OPT_H_

#include "my_defs.h"

void tune_all_parameters(double* theta,double* , sz_short* , sz_med , sz_short  ,double , sz_med* );

void simplex_constr_QP_with_PG(double* , double* , double* , sz_med );

void hyperplane_constr_QP(double* , double* , double* , sz_med );

#endif
