///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains implementations of AdaDIF, TunedRwR, and PPR.
 

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <inttypes.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <stdbool.h>

#include "comp_engine.h"
#include "csr_handling.h"
#include "my_defs.h"
#include "parameter_opt.h"
#include "my_utils.h"


static sz_short get_threads_and_width(sz_short* , sz_short ); 
static double* get_coef_A(sz_long , sz_med , sz_med , sz_long* , double* ,double* , const sz_long* , double );
static double* get_coef_b(sz_long , sz_med , sz_med ,sz_med , sz_long* , double* , sz_long* );


//Multi-threaded AdaDIF method (output is soft labels)
void AdaDIF_core_multi_thread( double* soft_labels, csr_graph graph, sz_med num_seeds, 
			       const sz_long* seed_indices, sz_short num_class, sz_short* class_ind,
			       sz_med* num_per_class, sz_med walk_length, double lambda, bool no_constr, bool single_thread){

	sz_short NUM_THREADS, width;

	NUM_THREADS = get_threads_and_width(&width, num_class);
	
	if(single_thread) NUM_THREADS = 1;
	
	printf("NUMBER OF THREADS: %"PRIu16" \n", (uint16_t) NUM_THREADS);	

	#if DEBUG
	printf("WIDTH= %"PRIu16"\n", (uint16_t) width);
	#endif	

	clock_t begin = clock();

	//The following three blocks create copies of the CSR matrix 
	//Each copy will be used by a thread 

	csr_graph* graph_copies = csr_mult_deep_copy( graph, NUM_THREADS );	

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;               

	printf("Time spent copying csr_matrix: %lf\n",time_spent);

	//MAIN LOOP
	//Prepare data to be passed to each thread

	pthread_t tid[NUM_THREADS];			
	pass_to_thread_type_2* data= (pass_to_thread_type_2*)malloc(NUM_THREADS*sizeof(pass_to_thread_type_2));	
	for(sz_short i=0;i<NUM_THREADS;i++){		
		(*(data+i))= (pass_to_thread_type_2) {.soft_labels=soft_labels, 
			     			      .num_seeds=num_seeds,
						      .num_per_class=num_per_class,
						      .class_ind=class_ind,
						      .graph = graph_copies[i],
						      .seeds=seed_indices,
						      .walk_length=walk_length,
						      .lambda=lambda,
						      .from=i*width, 
						      .no_constr = no_constr };
			     
		if(i==NUM_THREADS-1){		
			(data+i)->to=NUM_THREADS;
			(data+i)->num_local_classes= num_class -i*width;			
		}else{
			(data+i)->to=(i+1)*width;	
			(data+i)->num_local_classes=width;	
		}
	}

	//Spawn threads and start running
	for(sz_short i=0;i<NUM_THREADS;i++){
		pthread_create(&tid[i],NULL,AdaDIF_squezze_to_one_thread,(void*)(data+i));		
	}

	//Wait for all threads to finish before continuing	
	for(sz_short i=0;i<NUM_THREADS;i++){pthread_join(tid[i], NULL);}

	//Free copies and temporary arrays

	csr_array_destroy(graph_copies,(sz_short)NUM_THREADS);
	free(data);           

}



//Here I slice the output(soft labels) and input (class_ind)
//Using aliasing on the shifted pointers such that single threaded AdaDIF_core is compeltely "blind" to the slicing process
void* AdaDIF_squezze_to_one_thread( void* param){
	pass_to_thread_type_2* data = param;

	double* soft_labels = data->soft_labels + (data->graph.num_nodes * data->from);
	sz_short* class_ind = data->class_ind + data->num_seeds*data->from;
	sz_med* num_per_class = data->num_per_class + data->from;

	clock_t begin = clock(); 	

	AdaDIF_core( soft_labels, data->graph,  data->num_seeds, data->seeds, data->num_local_classes, class_ind, num_per_class,
		     data->walk_length, data->lambda, data->no_constr);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;  		

	printf("Thread from classes with index %"PRIu16" to %"PRIu16" finished in %lf sec \n",(uint16_t) data->from, (uint16_t) data->to, time_spent);

	pthread_exit(0);

}

//Core of AdaDIF method that runs on single thread (output is soft labels)
void AdaDIF_core( double* soft_labels, csr_graph graph, sz_med num_seeds,
		  const sz_long* seed_indices, sz_short num_class, sz_short* class_ind,
		  sz_med* num_per_class, sz_med walk_length, double lambda, bool no_constr){       
	
	for(sz_short i=0;i<num_class;i++){		
	
		double* land_prob = (double*) malloc(walk_length*graph.num_nodes*sizeof(double));
		double* dif_land_prob = (double*) malloc(walk_length*graph.num_nodes*sizeof(double));
		sz_long* local_seeds =(sz_long*)malloc(num_per_class[i] *sizeof(sz_long));    		

		sz_med k=0;
		for(sz_med j=0;j<num_seeds;j++){ 
			if( class_ind[i*num_seeds + j] ==1 )			
				local_seeds[k++] = seed_indices[j];
		}

		perform_random_walk(land_prob, dif_land_prob, graph , walk_length , local_seeds , num_per_class[i] );

				
		double* theta = get_AdaDIF_parameters(graph.num_nodes,graph.degrees,land_prob,dif_land_prob,seed_indices,local_seeds,
		                                      class_ind+i*num_seeds,walk_length,num_seeds,num_per_class[i],lambda,no_constr );
		#if PRINT_THETAS
		printf("THETA: ");
		for(int n=0;n<walk_length;n++) printf(" %.3lf",theta[n]);
		printf("\n");		
		#endif
	 
		matvec_trans_long( soft_labels+i*graph.num_nodes , land_prob, theta, graph.num_nodes, walk_length );

		//free landing probabilities
		free(local_seeds);
		free(land_prob);
		free(dif_land_prob);	
		free(theta);
	} 
}


//Extract landing and dif probabilities by performing K steps of simple random walk on graph 

void perform_random_walk(double* land_prob, double* dif_land_prob, csr_graph graph,
			 sz_med walk_length , sz_long* seeds , sz_med num_seeds ){

	double* seed_vector = (double*) malloc(graph.num_nodes* sizeof(double));
	double one_over_num_seeds = 1.0f /(double) num_seeds ;

	//prepare seed vector
	for(sz_long i=0;i<graph.num_nodes;i++) seed_vector[i] = 0.0f ;
	
	for(sz_med j=0;j<num_seeds;j++) seed_vector[seeds[j]] = one_over_num_seeds ;

	//do the random walk
	my_CSR_matvec( land_prob, seed_vector , graph);  

	for(sz_med j=1;j<walk_length;j++){
		my_CSR_matvec( land_prob+j*graph.num_nodes, land_prob+(j-1)*graph.num_nodes , graph);	
		my_array_sub( dif_land_prob+(j-1)*graph.num_nodes, land_prob+(j-1)*graph.num_nodes,
			      land_prob+j*graph.num_nodes, graph.num_nodes);
	}

	#if DEBUG
	printf("LAST LAND PROBs: \n");
	for(sz_long i=0;i<=100;i++) printf(" %lf ",land_prob[(walk_length-1)*graph.num_nodes + i ]) ;
	printf("...................... \n");
	#endif		

	//do one final step to obtain the last differential

	double* extra_step = (double*) malloc(graph.num_nodes* sizeof(double));
	my_CSR_matvec( extra_step, land_prob+ (walk_length -1)*graph.num_nodes , graph);	
	my_array_sub( dif_land_prob+(walk_length-1)*graph.num_nodes, land_prob+(walk_length-1)*graph.num_nodes,
		      extra_step, graph.num_nodes);	

        //free
	free(seed_vector);
	free(extra_step);
}

//Extract the AdaDIF diffusion coefficients 

double* get_AdaDIF_parameters( sz_long N, sz_long* degrees, double* land_prob, 
			       double* dif_land_prob, const sz_long* seed_indices,
			       sz_long* local_seeds, sz_short* class_ind, sz_med walk_length,
			        sz_med num_seeds, sz_med num_pos,double lambda, bool no_constr){
	
	double* theta = (double*) malloc(walk_length*sizeof(double));

        //A and b are the Hessian and linear component of the quadratic cost
        	
	double* A = get_coef_A(N,walk_length,num_seeds,degrees, land_prob, dif_land_prob,seed_indices,lambda);
	
	double* b = get_coef_b(N,walk_length,num_seeds,num_pos,degrees, land_prob, local_seeds);	
	
	if(!no_constr){
		simplex_constr_QP_with_PG(theta,A,b,walk_length);
	}else{
		hyperplane_constr_QP(theta,A,b,walk_length);
	}

        //free
	free(b);
	free(A);
		
	return theta;
}







// This function computes how many threads will be used and how many classes will be allocated per thread
static sz_short get_threads_and_width(sz_short* width ,sz_short num_class){
	sz_short num_procs=get_nprocs();
	sz_short num_threads;

	if(num_class<=num_procs){
		num_threads = num_class;
		*width=1;
	}else{
		num_threads = num_procs;
		*width = (sz_short)ceil((double)(num_class/(double)num_threads));
	}

	return num_threads;
}


//Obtain slice of G  as stationary distributions. (Unweighted) Seed set must be defined 
//Returns numbr of itrations untill convergence
sz_med get_slice_of_G( double* G_s, sz_long* seeds, sz_med num_seeds, double tel_prob, 
			 csr_graph graph, bool single_thread ){

	//Do it SMART: Use a temporary G_s_next where you store the left hand side of iteration
	//Then at eah iteration just flip pointers between G_s and G_s_next
	//Multiple threads compute different slices of G_s
	sz_short NUM_THREADS = get_nprocs();
	printf("NUMBER OF THREADS: %"PRIu16" \n",(uint16_t) NUM_THREADS);
	
	if(single_thread) NUM_THREADS = 1;

	sz_med iter[NUM_THREADS];

	csr_graph graph_scaled = csr_deep_copy_and_scale(graph,1.0f-tel_prob);

	double* G_s_next=malloc(graph.num_nodes*num_seeds*sizeof(double));

	//Initialization
	for(sz_long i=0;i<graph.num_nodes*num_seeds;i++) G_s[i]=0.0f;

	for(sz_med j=0;j<num_seeds;j++) G_s[ num_seeds*seeds[j] +j]=1.0f; 

	clock_t begin = clock();

	//The following three blocks create copies of the CSR matrix 
	//Each copy will be used by a thread 

	csr_graph* graph_copies = csr_mult_deep_copy( graph_scaled, (sz_short) NUM_THREADS );	

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;               

	printf("Time spent copying csr_matrix: %lf\n",time_spent);

	//MAIN LOOP
	//Prepare data to be passed to each thread
	sz_med width;	
	width=(sz_med)floor((double)(num_seeds/(double)NUM_THREADS));
	pthread_t tid[NUM_THREADS];			
	pass_to_thread_type_1* data= (pass_to_thread_type_1*)malloc(NUM_THREADS*sizeof(pass_to_thread_type_1));	
	for(sz_short i=0;i<NUM_THREADS;i++){		

	        (*(data+i))= (pass_to_thread_type_1) {.G_s=G_s,
			 			      .G_s_next=G_s_next,
						      .graph = graph_copies[i],
						      .seeds=seeds,
						      .M=num_seeds,
						      .tel_prob=tel_prob,
						      .from=i*width };

		if(i==NUM_THREADS-1){		
			(data+i)->to=num_seeds;
		}else{
			(data+i)->to=(i+1)*width;		
		}
		(data+i)->iter=&iter[i];
	}

	//Spawn threads and start running
	for(sz_short i=0;i<NUM_THREADS;i++){
		pthread_create(&tid[i],NULL,my_power_iter,(void*)(data+i));		
	}

	//Wait for all threads to finish before continuing	
	for(sz_short i=0;i<NUM_THREADS;i++){pthread_join(tid[i], NULL);}

	//Free copies and temporary arrays
	csr_destroy(graph_scaled);
	csr_array_destroy(graph_copies,(sz_short)NUM_THREADS);
	free(G_s_next);
	free(data);

	return iter[0];
}


//Power iteration that computes a subset (from,to) of the collumns of G_s
//Data are passed via struct that can be readily used by thread
void* my_power_iter(void* param){

	pass_to_thread_type_1* data = param;


	sz_long* seeds=data->seeds;
	double* G_s=data->G_s;
	double* G_s_next= data->G_s_next;
	csr_graph graph=data->graph;

	//Iterate using this "back and forth method" to minimize memory access
	clock_t begin = clock();

	sz_med iter=0;
	sz_short flag=0;	
	
	do{
		iter++;	
		flag=(flag==1) ? 0 : 1 ;
		if(flag==1){
			my_CSR_matmat( G_s_next , G_s  , graph , data->M , data->from, data->to);
			for(sz_med j=data->from;j<data->to;j++) G_s_next[data->M*seeds[j] +j]+=data->tel_prob; 
		}
		else{
			my_CSR_matmat( G_s, G_s_next  , graph , data->M, data->from, data->to);
			for(sz_med j=data->from;j<data->to;j++) G_s[data->M*seeds[j] +j]+= data->tel_prob; 
		}
	}while(iter<MAXIT && max_seed_val_difference( G_s, G_s_next , seeds ,data->M, data->from, data->to) >TOL);

	if(flag==1){
		for(sz_long i=0;i<graph.num_nodes;i++){
			for(sz_med j=data->from;j<data->to;j++)
				G_s[i*data->M + j]=G_s_next[i*data->M + j];
		}
	}

	*(data->iter)=iter;

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;        

	printf("Thread from column %"PRIu32" to %"PRIu32" finished in %lf sec \n", (uint32_t) data->from, (uint32_t) data->to, time_spent);

	pthread_exit(0);
}


//Personalized-Pagerank single thread
void my_PPR_single_thread( double* soft_labels, csr_graph graph, sz_med num_seeds,
			   const sz_long* seed_indices, sz_short num_class, sz_short* class_ind,
			   sz_med* num_per_class, sz_med walk_length, double tel_prob){

	csr_graph graph_scaled = csr_deep_copy_and_scale(graph,1.0f-tel_prob);
	 
	for(sz_short i=0;i<num_class;i++){		
		sz_long* local_seeds =(sz_long*)malloc(num_per_class[i] *sizeof(sz_long));    		

		sz_med k = 0;
		for(sz_med j=0;j<num_seeds;j++){ 
			if( class_ind[i*num_seeds + j] ==1 )			
				local_seeds[k++] = seed_indices[j];
		}

		//prepare seed vector
		double* soft = soft_labels + i*graph.num_nodes;
		double one_over_num_seeds = 1.0f /(double) num_per_class[i] ;
		for(sz_long n=0;n<graph.num_nodes;n++)  soft[n] = 0.0f ;
		for(sz_med j=0;j<num_per_class[i];j++) soft[local_seeds[j]] = one_over_num_seeds ;

		//perform random walk with restart
		sz_short flag=0;
		double* soft_next=malloc(graph.num_nodes*sizeof(double));
		for(sz_med j=0;j<walk_length;j++){
			flag=(flag==1) ? 0 : 1 ;
			if(flag==1){
				my_CSR_matvec(soft_next, soft, graph_scaled );
				for(sz_med k=0;k<num_per_class[i] ;k++)
					soft_next[local_seeds[k]] += tel_prob*one_over_num_seeds; 
			}
			else{
				my_CSR_matvec(soft, soft_next, graph_scaled );
				for(sz_med k=0;k<num_per_class[i] ;k++)
					soft[local_seeds[k]] += tel_prob*one_over_num_seeds; 
			}

		}

		if(flag==1){
			memcpy(soft, soft_next, graph.num_nodes*sizeof(double));
		}		

		
		free(soft_next);
		free(local_seeds);

	} 
	csr_destroy(graph_scaled);
}



//find l_max norm between vecors of known length
double max_seed_val_difference(double* A, double* B, sz_long* points , sz_med num_points, sz_med from, sz_med to){ 
	double dif;
	double max_dif=0.0f;
	for(sz_med i=from;i<to;i++){
		dif=fabs( A[ num_points*points[i] + i] - B[ num_points*points[i] + i] );
		max_dif = ( dif > max_dif ) ? dif : max_dif ;
	}
	return max_dif;
}


//Extract square submatrix of slice G_l that corresponds to labeled only
void extract_G_ll(double* G_ll, double* G_s, sz_long* seeds, sz_med num_seeds){ 

	for(sz_long i=0;i<num_seeds;i++){
		for(sz_med j=0;j<num_seeds;j++)
			G_ll[i*num_seeds + j]=G_s[seeds[i]*num_seeds + j];
	}
}



//The following two functions generate the coefficients required by the AdaDIF QP
static double* get_coef_A(sz_long N, sz_med K, sz_med L , sz_long* d , 
			  double* P ,double* P_dif , const sz_long* L_ind , double lambda ){ 
	
	double* A = (double*) malloc(K*K*sizeof(double));
	
	double* d_inv_times_lambda = (double*)malloc(N*sizeof(double));
	
	for(sz_long i=0;i<N;i++) d_inv_times_lambda[i] =lambda/(double)d[i];  
	
	
	double* P_temp =(double*) malloc(K*N*sizeof(double));

	for(sz_long i=0;i<N;i++){ 
		for(sz_med j=0;j<K;j++)
			P_temp[j*N+i]=d_inv_times_lambda[i]*P_dif[j*N+i];			
	}
	
	
	for(sz_med i=0;i<L;i++){ 
		for(sz_med j=0;j<K;j++)
			P_temp[j*N + L_ind[i] ]+= P[j*N+L_ind[i]]/(double)d[L_ind[i]];				
	}	
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)K, (int)K, (int)N, 1.0f, P, (int)N, P_temp, (int)N, 0.0f, A, (int)K);
		
	free(d_inv_times_lambda);
	free(P_temp);
	return A;
}

static double* get_coef_b(sz_long N, sz_med K, sz_med L ,sz_med num_pos , sz_long* d , double* P , sz_long* L_c ){

	double two_over_L = -2.0f/(double) L;
	
        double*	b = (double*) malloc(K*sizeof(double));

	for(sz_med i=0;i<K;i++) b[i]=0.0f;
	
	for(sz_med i=0;i<K;i++){
		for(sz_med j=0;j<num_pos;j++)
			b[i]+=P[i*N + L_c[j]]/(double)d[L_c[j]];
		b[i]*= two_over_L;
	}
	
	
	return b;
}
























