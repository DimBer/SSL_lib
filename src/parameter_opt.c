//////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routines for constrained quadratic optimization used by AdaDIF and TunedRwR.
 

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

#include "parameter_opt.h"
#include "my_defs.h"
#include "my_utils.h"

// DECLARE STATIC FUNCTIONS

//static double cost_fun ( double* , double* , double , double , int* , int, double);

static sz_med tune_parameters( double* , double* , sz_short* , sz_med , double , sz_med , double );
static double max_abs_dif(double*, double*, double*, sz_long );
static double cost_func(double*,double*,double*,sz_med);

//////////////////////////////////////////////////////////////////////////////////////////////////////////


//Evaluates quadtratic with Hessian A and linear part b at x

static inline double cost_func(double* A, double* b, double* x, sz_med len){
	
	double quad =0.0f, lin = 0.0f;
	
	for(sz_med i=0;i<len;i++){
		for(sz_med j=0;j<len;j++){
			quad+= A[i*len + j]*x[i]*x[j];
		}
		lin+=b[i]*x[i];
	}
	return quad + lin;
} 


//Infinity norm

static double max_abs_dif(double* a, double* b, double* Hess, sz_long len ){ 
	double dif,max=0.0;
	
	for(sz_long i=0;i<len;i++){
		dif = fabs(a[i]-b[i])/sqrt(Hess[i*len + i]);
		max = (dif>max) ? dif : max ; 
	}
	
	return max;
}



//Solving quadratic minimization over the probability simplex via projected gradient
//Used by AdaDIF and AdaDIF_LOO
// The following function returns x =arg min {x^T*A*x +x^T*B} s.t. x in Prob. Simplex
void simplex_constr_QP_with_PG(double* x, double* A, double* b, sz_med K){
	double  inf_norm, step_size;
	double* x_temp=malloc(K*sizeof(double));
	double* x_prev=malloc(K*sizeof(double));		
			
//	step_size = STEPSIZE_2;
//	step_size = STEPSIZE_2/pow(frob_norm(A,K), 2.0f);
	step_size = STEPSIZE_2/frob_norm(A,K);

	//Initialize to uniform
	for(sz_med i=0;i<K;i++) x[i]=1.0f/(double)K;
	
	//Initialize to e_k
	for(sz_med i=0;i<K;i++) x[i] = (i==K-1) ? 1.0f : 0.0f;
	
	sz_med iter=0;
	memcpy(x_prev,x,K*sizeof(double));
	do{
		iter++;
		//Take gradient step
		matvec(x_temp, A , x, K, K );
		
		for(sz_med j=0;j<K;j++) 
			x[j]-= step_size*( 2.0f*x_temp[j] +b[j] ); 
		
		project_to_simplex( x , K );
		
                #if DEBUG
		printf("\n COST: ");
		printf(" %lf ",cost_func(A,b,x,K));
		#endif	
				
		inf_norm = max_abs_dif(x_prev,x, A , (sz_long)K );

		memcpy(x_prev,x,K*sizeof(double));
	
	}while( iter<MAXIT_GD &&  inf_norm>GD_TOL_2 );	
		
	printf("\n Optimization finished after: %"PRIu32" iterations\n", (uint32_t) iter);
	
	free(x_temp);
	free(x_prev);
}

//Solving quadratic minimization with the constrain that x^T*1=1
//Used by AdaDIF and AdaDIF_LOO in unconstrained mode
//The following function returns x =arg min {x^T*A*x +x^T*B} s.t. x^T*1=1
//Solution is given in closed-form using Lagrange multipliers by solving linear systems via LU decomposition   
void hyperplane_constr_QP(double* x, double* A, double* b, sz_med K){ 

	//Change matrix to poitnter to pointer format required by LU routine
	//Note that aliasing happens here and the LU decomposition happens ``in-place'' 
	double** A_temp=(double**)malloc(K*sizeof(double*));
	int* Perm = (int*)malloc((K+1)*sizeof(int));	
	
	for(sz_med i=0;i<K;i++) *(A_temp+i)=&A[i*K];
				

        for(sz_med i=0;i<K;i++) A_temp[i][i]+= L2_REG_LAMBDA;  //L2 regularization is needed due to bad conditioning of A
	
	
	
       #if DEBUG 
	printf("A:");
	for(sz_med i=0;i<K;i++){
		printf("\n");
		for(sz_med j=0;j<K;j++) printf(" %lf ",A_temp[i][j]);
	}			
	printf("\n B: ");
	for(sz_med i=0;i<K;i++) printf(" %lf ",b[i]);

	for(sz_med i=0;i<K;i++){
		b[i]=1.0;
		for(sz_med j=0;j<K;j++) A_temp[i][j]=0.0;
		A_temp[i][i]=2.0 + L2_REG_LAMBDA;
	}			
        #endif

	//First the LU decoposition of A
	LUPDecompose(A_temp, (int)K , LU_TOL , Perm);

	//The unconstrained solution 
	double* x_unc =(double*)malloc(K*sizeof(double));

	LUPSolve(A_temp, Perm, b, (int)K, x_unc);

	//The Lagrance multiplier
	double sum_num = 0.0f, sum_den =0.0f ,lambda_star =0.0f;
	for(sz_med i=0;i<K;i++){
		sum_num += x_unc[i];
		sum_den += b[i]*x_unc[i];	
	}	
	lambda_star = (sum_num - 1.0)/sum_den;
	
		
	//The solution to lambda time all ones
	double* x_all_ones =(double*)malloc(K*sizeof(double));		
	double* all_ones_lambda_star =(double*)malloc(K*sizeof(double));
	for(sz_med i=0;i<K;i++){
		x_all_ones[i]=0.0f;
		all_ones_lambda_star[i] = lambda_star;
	}	
	
	LUPSolve(A_temp, Perm, all_ones_lambda_star, (int)K, x_all_ones);
	 	 	
	//The final solution
	for(sz_med i=0;i<K;i++) x[i] = x_unc[i] - x_all_ones[i];
		
 
        //free
	free(all_ones_lambda_star);
	free(x_all_ones);
	free(x_unc);
	free(Perm);
	free(A_temp);	
}



//Tune parameter vector via simplex-constrained reg LS
// Used by Tuned_RwR
//This function solves the quadratic minimization over double simplex constrains that is required for Tuned RwR

static sz_med tune_parameters( double* x, double* G_ll,  sz_short* ind , sz_med L,
				 double lambda, sz_med num_pos , double step_size){
	sz_med point_pos,point_neg;
	double ratio,max_grad,max_grad_prev=0.0f;
	double one_over_pos;
	double one_over_neg;
	double* res=malloc(L*sizeof(double));
	double* x_temp=malloc(L*sizeof(double));
	double* x_t=malloc(L*sizeof(double));
	double* G_ll_signed=malloc(L*L*sizeof(double));
	double* x_pos=malloc(num_pos*sizeof(double));
	double* x_neg=malloc((L-num_pos)*sizeof(double));


	one_over_pos=1.0f/(double)(num_pos);
	one_over_neg=1.0f/(double)(L- num_pos);

	for(sz_med i=0;i<L;i++){ 
		x[i]= (ind[i] ==1 )?  one_over_pos : one_over_neg; 
		x_t[i]= (ind[i] ==1 )?  one_over_pos : one_over_neg;
	}  //initialize

	//Copy G_ll with signs according to indicator
	for(sz_med i=0;i<L;i++){ 
		if(ind[i]==1){
			for(sz_med j=0;j<L;j++) G_ll_signed[j*L +i] = G_ll[j*L +i] ;
		}else{
			for(sz_med j=0;j<L;j++) G_ll_signed[j*L +i] = - G_ll[j*L +i] ;
		}
	}

	
	sz_med iter=0;
	do{

		//Take gradient step
		matvec(res, G_ll_signed, x, L, L );

		for(sz_med j=0;j<L;j++) res[j] = (ind[j]==1)? res[j]-1 : res[j]+1 ; 

		matvec_trans(x_temp, G_ll_signed, res, L, L );

		for(sz_med j=0;j<L;j++){
			if(ind[j]==1){
				x_t[j]= x[j] - step_size*( x_temp[j] + lambda*(x[j] - one_over_pos) );
			}else{
				x_t[j]= x[j] - step_size*( x_temp[j] + lambda*( x[j] - one_over_neg ) ); 
			}
		}

		max_grad=max_diff(x_t,x,L);
		ratio= max_grad/max_grad_prev;
		max_grad_prev=max_grad;

		//Project positives and negatives onto simplex
		point_pos=0;
		point_neg=0;
		for(sz_med k=0;k<L;k++){
			if(ind[k]==1){
				x_pos[point_pos++]=x_t[k];
			}else{
				x_neg[point_neg++]=x_t[k];
			}
		}

		project_to_simplex( x_pos, num_pos );
		project_to_simplex( x_neg, L - num_pos );

		point_pos=0;
		point_neg=0;
		for(sz_med k=0;k<L;k++){
			if(ind[k]==1){
				x[k]=x_pos[point_pos++];
			}else{
				x[k]=x_neg[point_neg++];
			}
		}

		iter++;
	}while( iter<MAXIT_GD &&  fabs(1.0f-ratio)>GD_TOL ); 
	//Terminate when ratio between l-inf norm of two consequtive gradients is less tha threshold

	for(sz_med j=0;j<L;j++) x[j]=(ind[j]==1)? x[j] : -x[j] ;  //Give back the signs to theta

        //free
	free(res);
	free(x_t);
	free(x_temp);
	free(G_ll_signed);
	free(x_pos);
	free(x_neg);

	return iter;
}



//Tune parameter vectors for each class
void tune_all_parameters(double* theta,double* G_ll, sz_short* class_ind, sz_med num_seeds,
			 sz_short num_class ,double lambda_unscaled, sz_med* num_per_class ){
	sz_med iters;
	double lambda,step_size,G_fro=0.0f;

	lambda=((double)num_seeds) * lambda_unscaled; //scale lambda to dimension of cost

	for(sz_med i=0;i<num_seeds;i++){
		for(sz_med j=0;j<num_seeds;j++) 
			G_fro += pow(G_ll[i*num_seeds + j],2.0f);
	}

	step_size=STEPSIZE/(G_fro + lambda ); // Approximae step_size using Frobenious norm of matrix

	printf(" STEP SIZE: %lf \n ",step_size);

	for(sz_med i=0;i<num_class;i++){
		double* this_theta=malloc(num_seeds*sizeof(double));
		sz_short* this_class_ind=malloc(num_seeds*sizeof(sz_short));

		for(sz_med j=0;j<num_seeds;j++) this_class_ind[j]=class_ind[i*num_seeds + j];

		iters = tune_parameters(this_theta, G_ll, this_class_ind ,num_seeds,lambda,
				        num_per_class[i], step_size);

		printf(" PGD ITERS: %"PRIu32" \n ", (uint32_t) iters);

		for(sz_med j=0;j<num_seeds;j++) theta[j*num_class + i]=this_theta[j];

		free(this_theta);
		free(this_class_ind);
	}

}







