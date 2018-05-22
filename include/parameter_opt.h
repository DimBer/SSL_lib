#ifndef PARAMETER_OPT_H_   
#define PARAMETER_OPT_H_



void tune_all_parameters(double* theta,double* , uint8_t* , uint16_t , uint8_t  ,double , uint16_t* );

void simplex_constr_QP_with_PG(double* , double* , double* , uint16_t );

void hyperplane_constr_QP(double* , double* , double* , uint16_t );

#endif
