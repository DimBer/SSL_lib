/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains collection of low-level routines for handling lists, 
 one-hot matrices, evaluation metrics (f1-scores), and other.

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <cblas.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "my_utils.h"
#include "my_defs.h"



void assert_all_nodes_present(csr_graph graph, const sz_long* seed_indices, sz_med num_seeds){
	
	for(sz_med i=0;i<num_seeds;i++){
		if(seed_indices[i]>graph.num_nodes){
			printf("ERROR: Seed node index does not appear in edgelist (probably an isolated node)\n");
		        exit(EXIT_FAILURE); 		
		}			
	}
	
}


sz_long rand_lim(sz_long limit) {
	/* return a random number between 0 and limit inclusive.
	 */
	int divisor = RAND_MAX/((int)limit+1);
	int retval;
	do { 
		retval = rand() / divisor;
	} while (retval > limit);
	return retval;
}


//Draw random samples with replacement from 0 to N-1
void random_sample( sz_long* seeds, abstract_labels labels, abstract_labels all_labels, sz_med num_seeds, sz_long N){

	sz_long temp;
	sz_short flag;
	
	//Draw seed indexes
	seeds[0]=rand_lim(N-1);
	for(sz_med i=1;i<num_seeds;i++){
		do{
			temp=rand_lim(N-1);
			flag=0;
			for(sz_med j=0;j<i;j++){
				if(temp==seeds[j])
					flag=1;
			}
		}while(flag==1);
		seeds[i]=temp;
	}

	//Draw corresponding labels
	if(all_labels.is_multilabel){
		for(sz_med i=0;i<num_seeds;i++){
			for(sz_short j=0; j<all_labels.mlabel.num_class;j++) labels.mlabel.bin[j][i] = all_labels.mlabel.bin[j][seeds[i]];
		}		
	}else{
		
		for(sz_med i=0;i<num_seeds;i++){ 
			labels.mclass[i]=all_labels.mclass[seeds[i]];
			//printf("%d\n",seeds[i]);
		}
	}

	// +1 seed indexes !
	for(sz_med i=0;i<num_seeds;i++){seeds[i]+=1;}	

}



//My comparator for two collumn array. Sorts second col according to first
int compare ( const void *pa, const void *pb ) 
{
	const sz_long *a = *(const sz_long **)pa;
	const sz_long *b = *(const sz_long **)pb;
	if(a[0] == b[0])
		return a[1] - b[1];
	else
		return a[0] - b[0];
}  

//My comparator for a struct with double and index
int compare2 ( const void *pa, const void *pb ) 
{
  val_and_ind *a1 = (val_and_ind*)pa;
  val_and_ind *a2 = (val_and_ind*)pb;
  if((*a1).val>(*a2).val)return -1;
  else if((*a1).val<(*a2).val)return 1;
  else return 0;
}  

//frobenious norm of double-valued square matrix
double frob_norm(double* A, sz_med dim){
	double norm=0.0f;

	for(sz_med i=0;i<dim*dim;i++){
		norm+=pow(A[i],2.0f);
	}
		
	return sqrt(norm);	
}

//mean of double array
double mean(double* a, int len){
	double sum=0;
	for(int i=0;i<len;i++) sum+=a[i];		
	return sum/(double)len;	
}

//Simple linear system solver using the LU decomposition
/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **A, int *P, double *b, int N, double *x){

    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++)
            x[i] -= A[i][k] * x[k];

        x[i] = x[i] / A[i][i];
    }
}



//Simple LU decomposition routine
/* INPUT: A - array of pointers to rows of a square matrix having dimension N
 *        Tol - small tolerance number to detect failure when the matrix is near degenerate
 * OUTPUT: Matrix A is changed, it contains both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
 *        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1 
 *        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N, 
 *        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S    
 */
int LUPDecompose(double **A, int N, double Tol, int *P) {

    int i, j, k, imax; 
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) { 
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; //failure, matrix is degenerate

        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            //pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            //counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1;  //decomposition done 
}



//Interface for CBLAS matrix vector product
void matvec(double*y, double* A, double* x, sz_med M, sz_med N ){
	 	
	for(int i=0;i<M;i++){y[i]=0.0f;}

	cblas_dgemv( CblasRowMajor , CblasNoTrans , (int)M , (int)N, 1.0f, A, (int)M, x, 1, 0.0f, y, 1);


}




//Interface for CBLAS matrix vector product
void matvec_trans(double*y, double* A, double* x, sz_med M, sz_med N ){
	 
	for(sz_med i=0;i<M;i++){y[i]=0.0f;}

	cblas_dgemv( CblasRowMajor , CblasTrans , (int)M , (int)N, 1.0f, A, (int)M, x, 1, 0.0f, y, 1);


}

//Interface for CBLAS trnaspose-matrix vector product
void matvec_trans_long( double* y , double* A, double* x, sz_long N, sz_med p ){		

	cblas_dgemv (CblasRowMajor, CblasTrans, (int) p, (int)N , 1.0f, A,  (int) N , x, 1 , 0.0f, y, 1);



}

//Interface for CBLAS mutrix matrix product
void matrix_matrix_product(double*C, double* A, double* B, sz_long m, sz_med k , sz_short n){

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0f, A, (int)k, B, (int)n, 0.0f, C, (int)n);

}


//Project vector onto simplex by alternating projections onto line and positive quadrant
//Operation happens in place
void project_to_simplex( double* x, sz_med N ){
	double sum,a; 
	sz_short flag;

	do{
		flag=0;
		sum=0.0f;
		
		for(sz_med i=0; i<N; i++) sum+=x[i];

		a=(sum - 1.0f)/(double)N;

		for(sz_med i=0; i<N; i++){
			x[i]-=a;
			if(x[i]<= - PROJ_TOL){
				x[i]=0.0f;
				flag=1;}
		}

	}while(flag==1);
}


//find l_max norm between vecors of known length
double max_diff(double* x, double* y , sz_med L){ 
	double dif;
	double max_dif=0.0f;
	for(sz_med i=0;i<L;i++){
		dif=fabs( x[i] - y[i] );
		max_dif = ( dif > max_dif ) ? dif : max_dif ;
	}
	return max_dif;
}

//Print array (double)
void print_array_1D(double* arr, sz_long N, sz_long M){
	sz_long i,j;
	printf("Array: \n");
	for(i=0;i<N;i++){
		printf("\n");
		for(j=0;j<M;j++) printf("%lf  ",arr[i*M + j]);
	}
	printf("\n");
}

//Pradict labels from largest soft label
void predict_labels( class_t* label_out, double* soft_labels, class_t* class,
	             sz_long graph_size, sz_short num_class ){
	sz_short max_ind;
	double max_val;

	for(sz_long i=0;i<graph_size;i++){
		max_val=-100.0f;
		for(sz_short j=0;j<num_class;j++){  
			if(max_val<soft_labels[i*num_class + j]){
				max_val=soft_labels[i*num_class + j];
				max_ind=j;}
		}
		label_out[i]=class[max_ind];
	}
}

//Pradict labels from largest soft label (Soft labels are transposed array)
void predict_labels_type2( class_t* label_out, double* soft_labels, class_t* class,
			   sz_long graph_size, sz_short num_class ){
			   				   
	sz_short max_ind;
	double max_val;

	for(sz_long j=0;j<graph_size;j++){
		max_val=-100.0f;
		for(sz_short i=0;i<num_class;i++){  
			if(max_val<soft_labels[i*graph_size + j]){
				max_val=soft_labels[i*graph_size + j];
				max_ind=i;}	
		}
		label_out[j]=class[max_ind];		
	}
}

//Print the two collumn array of edges
void print_edge_list(const sz_long** edge_list, sz_long len){
	printf("EDGE LIST: \n");
	for(sz_long i=0;i<len;i++)
		printf("%"PRIu64"  %"PRIu64"\n", (uint64_t) edge_list[i][0], (uint64_t) edge_list[i][1]);
}




//Print array (double)
void print_predictions(class_t* arr, sz_long N){ 
	printf("Predicted labels: \n");
	for(sz_long i=0;i<N;i++) printf("%"PRIi16"\n", (int16_t) arr[i]);
}

//Check if file is valid
int file_isreg(char *path) {
    struct stat st;

    if (stat(path, &st) < 0)
        return -1;

    return S_ISREG(st.st_mode);
}



//Elementwise subtract b from a and store in c 
void my_array_sub(double* c, double* a, double* b, sz_long N ){	
	for(sz_long i=0;i<N;i++) c[i] = a[i] -b[i]; 
}

//Max of array 

sz_short max_u8( sz_short* a, sz_long N ){
	sz_short max= 0;
	for(sz_long i=0; i<N ; i++) max = ( a[i]>max ) ? a[i] : max ; 
	return max;
}

sz_long max_u64( sz_long* a, sz_long N ){
	sz_long max= 0;
	for(sz_long i=0; i<N ; i++) max = ( a[i]>max ) ? a[i] : max ; 
	return max;
}



//Return edge list and count to main 

sz_long** give_edge_list( char* file_name, sz_long* count ){ 

	sz_long** buffer= (sz_long **)malloc(EDGE_BUFF_SIZE * sizeof(sz_long *));
 
	for(sz_long i=0;i<EDGE_BUFF_SIZE;i++)
		buffer[i]= (sz_long*) malloc(2*sizeof(sz_long));

	FILE* file= fopen(file_name, "r");

	if(!file) printf("ERROR: Cannot open graph file");

	// Read adjacency into buffer into buffer and return length count=edges
	*count= read_adjacency_to_buffer(buffer,file);
	printf("Number of edges: %"PRIu64"\n", (uint64_t) *count);

	//print_edge_list( buffer, *count);

	//Free excess memory
	for(sz_long i=*count+1;i<EDGE_BUFF_SIZE;i++)
	{free(buffer[i]);}
	buffer=realloc(buffer,(*count)*sizeof(sz_long*));

	return buffer;
}


//Read .txt file into buffer
sz_long read_adjacency_to_buffer(sz_long** buffer, FILE* file){
        sz_long count = 0;
	for (; count < EDGE_BUFF_SIZE; ++count)
	{
		int got = fscanf(file, "%"SCNu64"\t%"SCNu64"\n", &buffer[count][0] , &buffer[count][1]);
		if ((got != 2)||( (buffer[count][0]==0) && (buffer[count][1]==0))) break; 
		// Stop scanning if wrong number of tokens (maybe end of file) or zero input
	}
	fclose(file);
	return count;
}


//Read class.txt file into buffer (ignore first collumn here)
class_t* read_labels(char* filename, sz_long* label_count){
	sz_long count = 0;
	sz_long* indexes = (sz_long*) malloc(sizeof(sz_long)*CLASS_BUFF_SIZE);
	class_t* buffer = (class_t*) malloc(sizeof(class_t)*CLASS_BUFF_SIZE);
	FILE* file= fopen(filename, "r");

	if(!file) printf("ERROR: Cannot open label file");

	for (; count < CLASS_BUFF_SIZE; ++count)
	{
		int got = fscanf(file, "%"SCNu64"%"SCNu8"", &indexes[count] , &buffer[count]);
		if (got != 2) break; // wrong number of tokens - maybe end of file
	}
	fclose(file);

	*label_count = count;
	buffer = realloc(buffer,sizeof(class_t)*count);
	indexes = realloc(indexes,sizeof(sz_long)*count);

	
	my_relative_sorting( indexes, buffer, count );
      		
        		
	free(indexes);
	return buffer;
}

//Read seed and label file when in operation mode
sz_long* read_seed_file( char* filename, sz_med* num_seeds, sz_short* num_class, abstract_labels* label_in ){

        //First read file into buffers
	sz_long count = 0;
	sz_long* index_buffer = (sz_long*) malloc(sizeof(sz_long)*CLASS_BUFF_SIZE);
	class_t* label_buffer = (class_t*) malloc(sizeof(class_t)*CLASS_BUFF_SIZE);
	FILE* file= fopen(filename, "r");

	if(!file) printf("ERROR: Cannot open seed file");

	for (; count < CLASS_BUFF_SIZE; ++count)
	{
		int got = fscanf(file, "%"SCNu64"%"SCNd8"", &index_buffer[count] , &label_buffer[count]);
		if (got != 2) break; // wrong number of tokens - maybe end of file
	}
	fclose(file);
	label_buffer = realloc(label_buffer,sizeof(class_t)*count);
	index_buffer = realloc(index_buffer,sizeof(sz_long)*count);
	
	//prepare input labels and seeds for is_multilabel or multi_class
	
	sz_long* seed_indices;
	
	if(label_in->is_multilabel){		
		*num_class = max_u8( (sz_short*) label_buffer, count);		
		my_relative_sorting( index_buffer, label_buffer, count );
		seed_indices = find_unique_from_sorted( index_buffer, count , num_seeds );

		label_in->mlabel = init_one_hot(*num_class , *num_seeds);
		sz_long j=0;
		for(sz_long i=0;i<count;i++){
			if(i>0) j = (index_buffer[i] == index_buffer[i-1] ) ? j : j+1;
			label_in->mlabel.bin[label_buffer[i]-1][j]=1;
		}
		
	        free(index_buffer);
		free(label_buffer);		
	}else{
		seed_indices = index_buffer;
		label_in->mclass = label_buffer;
		*num_seeds = (sz_med) count;
	}
	
	return seed_indices;
}	


//write predicted labels (or ranking in multilabel case) to output file
void save_predictions(char* filename, abstract_label_output label_out, sz_long len, sz_short num_class){
	
	FILE* file = fopen(filename, "w");	
	
	if(!file) printf("ERROR: Cannot open outfile");
	
	if(label_out.is_multilabel){ 
		val_and_ind line_of_out[num_class];
		for(sz_long i=0; i<len; i++){
			for(sz_short j=0; j<num_class; j++) line_of_out[j] = (val_and_ind) {.val = label_out.mlabel[j*len + i], .ind=(int)j}; 
				
			qsort( line_of_out, num_class, sizeof(line_of_out[0]), compare2);
			
			fprintf(file, "%"SCNu64":\t", (uint64_t) i+1 );
			
			for(sz_short j=0; j<num_class; j++) fprintf(file, "%"PRIu16" ", (uint16_t) line_of_out[j].ind +1 );
			
			fprintf(file, "\n");
		}
	}else{
		for(sz_long i=0; i<len; i++) fprintf(file, "%"SCNu64"\t%"SCNd16"\n", (uint64_t) i+1 , (int16_t) label_out.mclass[i]);	
	}
	
	fclose(file);
}

//Read class.txt file into one_hot_matrix (ignore first collumn here)
one_hot_mat read_one_hot_mat(char* filename, sz_long* label_count){
	sz_long count = 0;
	sz_long* indexes = (sz_long*) malloc(sizeof(sz_long)*CLASS_BUFF_SIZE);
	sz_short* buffer = (sz_short*) malloc(sizeof(sz_short)*CLASS_BUFF_SIZE);
	FILE* file = fopen(filename, "r");

	if(!file) printf("ERROR: Cannot open label file");

	for (; count < CLASS_BUFF_SIZE; ++count)
	{
		int got = fscanf(file, "%"SCNu64"\t%"SCNu8"", &indexes[count] , &buffer[count]);
		if (got != 2) break; // wrong number of tokens - maybe end of file
	}
	fclose(file);

	buffer = realloc(buffer,sizeof(sz_short)*count);
	indexes = realloc(indexes,sizeof(sz_long)*count);
	
	sz_short num_class = max_u8( buffer, count);

	sz_long length = max_u64( indexes, count);
	
//	printf("LENGTH %"PRIu64"\n",length);

	one_hot_mat all_labels = init_one_hot(num_class , length);
	
	for(sz_long i=0;i<count;i++) all_labels.bin[buffer[i]-1][indexes[i]-1]=1; 

	*label_count = length;
      		        		
	free(indexes);
	free(buffer);
	return all_labels;
}

// Sort A and ind with respect to indexes in ind
void my_relative_sorting( sz_long* ind, class_t* A, sz_long len ){ 
	sz_long* temp = (sz_long*)malloc(len*sizeof(sz_long));
	sz_long* sorted_inds = (sz_long*)malloc(len*sizeof(sz_long));


	for(sz_long i=0;i<len;i++){ temp[i] = i ;}


	sz_long** zipped_arrays = zip_arrays(ind,temp,len); 
 

        qsort( zipped_arrays , len, sizeof(zipped_arrays[0]), compare); 

	
        unzip_array(zipped_arrays, temp, sorted_inds, len );

        rearange(sorted_inds, (void*) A, "class_t" , len );
	
	free(temp);
	free(sorted_inds);

}


//Zip two arrays into a list of length-2 lists
sz_long** zip_arrays(sz_long* A, sz_long* B, sz_long len){
	sz_long** zipped = (sz_long**)malloc(len*sizeof(sz_long*));
	
	for(sz_long i=0;i<len;i++){ 
		zipped[i] = (sz_long*)malloc(2*sizeof(sz_long)); 
                ** (zipped +i) = A[i]; 
                *(* (zipped +i)+1) = B[i]; 
	}
	return zipped;	
}


//Unzip two arrays into input pointers and destroy zipped array
void unzip_array(sz_long** zipped, sz_long* unzip_1, sz_long* unzip_2, sz_long len ){
		
	for(sz_long i=0;i<len;i++){
		unzip_1[i] = zipped[i][0];
		unzip_2[i] = zipped[i][1];
		free(*(zipped+i));
	}	
	
	free(zipped);
}

//Rearange elements of array A accortding to given indexes
//Works for any type of array as long as the type is provided
void rearange(sz_long* ind, void* A, char* type , sz_long len ){

	if(strcmp(type,"double")==0){ 
		double A_temp[len];
		memcpy(A_temp, A, len*sizeof(double));  
		for(sz_long i=0;i<len;i++){ *((double*)A + i) = A_temp[ind[i]];}
	}else if(strcmp(type,"sz_long")==0){
		sz_long A_temp[len];
		memcpy(A_temp, A, len*sizeof(sz_long));  
		for(sz_long i=0;i<len;i++){ *((sz_long*)A + i) = A_temp[ind[i]];}		
	}else if(strcmp(type,"uint32_t")==0){
		uint32_t A_temp[len];
		memcpy(A_temp, A, len*sizeof(uint32_t));  
		for(sz_long i=0;i<len;i++){ *((uint32_t*)A + i) = A_temp[ind[i]];}
	}else if(strcmp(type,"sz_med")==0){
		sz_med A_temp[len];
		memcpy(A_temp, A, len*sizeof(sz_med));  
		for(sz_long i=0;i<len;i++){ *((sz_med*)A + i) = A_temp[ind[i]];}		
	}else if(strcmp(type,"sz_short")==0){
		sz_short A_temp[len];
		memcpy(A_temp, A, len*sizeof(sz_short));  
		for(sz_long i=0;i<len;i++){ *((sz_short*)A + i) = A_temp[ind[i]];}		
	}else if(strcmp(type,"int32_t")==0){
		int32_t A_temp[len];
		memcpy(A_temp, A, len*sizeof(int32_t));  
		for(sz_long i=0;i<len;i++){ *((int32_t*)A + i) = A_temp[ind[i]];}		
	}else if(strcmp(type,"int16_t")==0){
		int16_t A_temp[len];
		memcpy(A_temp, A, len*sizeof(int16_t));  
		for(sz_long i=0;i<len;i++){ *((int16_t*)A + i) = A_temp[ind[i]];}
	}else if(strcmp(type,"class_t")==0){
		class_t A_temp[len];
		memcpy(A_temp, A, len*sizeof(class_t));  
		for(sz_long i=0;i<len;i++){ *((class_t*)A + i) = A_temp[ind[i]];}
	}else if(strcmp(type,"long double")==0){
		long double A_temp[len];
		memcpy(A_temp, A, len*sizeof(long double));  
		for(sz_long i=0;i<len;i++){ *((long double*)A + i) = A_temp[ind[i]];}  		
	}else if(strcmp(type,"int")==0){
		int A_temp[len];
		memcpy(A_temp, A, len*sizeof(int));  
		for(sz_long i=0;i<len;i++){ *((int*)A + i) = A_temp[ind[i]];}
	}else{
		printf("Unknown rearange type\n");
		exit(EXIT_FAILURE);
	}					
	
}



//Remove items of given indexes from array (list)
// Return result in NEW  list 
sz_long* remove_from_list(const sz_long* list, const sz_long* indexes_to_be_removed, 
			   sz_long len, sz_long num_removed ){

	sz_long* new_list = (sz_long*) malloc((len-num_removed)*sizeof(sz_long));

	int mask[len];
	
	memset(mask, 0, len*sizeof(int));
	
	for(sz_long i =0; i<num_removed; i++){ mask[indexes_to_be_removed[i]] =1 ;}

	
	sz_long k=0;
	for(sz_long i =0; i<len; i++){
		if(mask[i]==0){
		    new_list[k++] = list[i];			
		}
	}
	
	return new_list;
}



//Find unique elements among known number of entries in buffer
//Also eturn number of unique entries
sz_short find_unique(class_t* unique_elements, const class_t* buffer,sz_med N){
	 
	sz_short new,end=1;

	unique_elements[0]=buffer[0];
	for(sz_med i=0;i<N;i++){
		new=1;
		for(sz_short j=0;j<end;j++)
		{if(unique_elements[j]==buffer[i])
			new=0;}

		if(new)
			unique_elements[end++]=buffer[i];

	}
	return end;
}

//Find unique entries from list of sorted (unisgned indexes) 
sz_long* find_unique_from_sorted( sz_long* sorted_buffer, sz_long len , sz_med* num_unique ){
	sz_long* unique = (sz_long*) malloc(len*sizeof(sz_long));
	*num_unique = 1;
	
	unique[0] = sorted_buffer[0]; 
	for(sz_long i=1;i<len;i++){
		if(!(sorted_buffer[i] == sorted_buffer[i-1] )){
			*num_unique += 1;
			unique[*num_unique-1] = sorted_buffer[i];	
		}		
	} 	
	unique = realloc(unique,*num_unique*sizeof(sz_long));
	return unique;
}

// Converts list to one hot binary matrix of one_hot_mat type
// Label count is number of indexes with known labels
// label_count<=length
//If label_count<=length, then rows of the one_hot matrix without a coresponding labeled index will be [0...0]
one_hot_mat list_to_one_hot( sz_long* ind , class_t* labels, sz_short num_class ,
			     class_t* class ,  sz_long label_count ,sz_long length){
	
	one_hot_mat one_hot = init_one_hot( num_class , length);
	
	for(sz_long i=0;i<label_count;i++){ 
		for(sz_short j=0;j<num_class;j++){ 
			if( labels[i] == class[j] ){
				one_hot.bin[j][ind[i]]=1;
				break;
			}
		}
	}		
		
	return one_hot;
}


//Allocate one_hot_mat
one_hot_mat init_one_hot(sz_short num_class ,sz_long length){	
	one_hot_mat one_hot;
	one_hot.bin = (sz_short**) malloc(num_class*sizeof(sz_short*));

	for(sz_short i=0;i<num_class;i++){ 
		one_hot.bin[i] = (sz_short*) malloc(length*sizeof(sz_short)); 
		for(sz_long j=0;j<length;j++){	        
			one_hot.bin[i][j] =0 ;
		}
	}		
	one_hot.num_class = num_class;
	one_hot.length = length;	
	return one_hot;
}


// Destroy (free) one_hot matrix
void destroy_one_hot(one_hot_mat one_hot){
	for(sz_short i=0;i<one_hot.num_class;i++) free(one_hot.bin[i]); 
	free(one_hot.bin);
}


// Return array with number of non-zero entries in each row of one-hot-matrix
sz_short* return_num_labels_per_node( one_hot_mat one_hot ){
	sz_short* num_labels = (sz_short*)malloc(one_hot.length*sizeof(sz_short));
	for(sz_long i=0;i<one_hot.length;i++){
		num_labels[i]=0;
		for(sz_short j=0;j<one_hot.num_class;j++) num_labels[i]+= one_hot.bin[j][i];		
	}
	return num_labels;
}


// Return a one_hot_mat with non-zeros for each row given to the k-largest (specified by num_lpn) corresponding soft labels
// Does not do sorting and has O(kC) complexity per node instead of O(C logC). Will be slow if C and k large
one_hot_mat top_k_mlabel( double* soft_labels , sz_short* num_lpn, sz_long length, sz_short num_class ){
	double max_max,max,val;
	sz_short max_ind;
	one_hot_mat label_out = init_one_hot(num_class ,length);
		
	for(sz_long i=0; i<length;i++ ){
		max_max =1.1f;
		for(sz_short k = 0; k<num_lpn[i];k++){
			max = 0.0f;
			for(sz_short j=0; j<num_class ;j++ ){
				val = soft_labels[j*length +i]; 
				if( ( val < max_max) && (val > max) ){
					max = val;
					max_ind = j;
				}			
			}		
			max_max = max;
			label_out.bin[max_ind][i]=1; 		
		}
	}
	
	return label_out;
}


// Computes rate of true positive, False positive and false negative
// A binary mask determines which values I am interested in (usually unlabeled entries..)
detector_stats get_detector_stats( sz_short* true_val, sz_short* pred_val, sz_long len, int* mask ){
	
	detector_stats stats = {.true_pos = 0.0,
			        .true_neg =0.0,	
			        .false_pos = 0.0,
			        .false_neg = 0.0};
	
	sz_long k=0;
	for(sz_long i=0;i<len;i++){
	    if(mask[i]!=0){		
	    	k++;
		if(true_val[i]==1){
			if(pred_val[i]==1){
				stats.true_pos+=1.0;
			}else{
				stats.false_neg+=1.0;
			}				
		}else{
			if(pred_val[i]==1){
				stats.false_pos+=1.0;
			}else{
				stats.true_neg+=1.0;
			}
		}
	    }	
	}       
	
	stats.true_pos/=(double) k;
	stats.true_neg/=(double) k;
	stats.false_pos/=(double) k;
	stats.false_neg/=(double) k;
	
	return stats;
}


// Harmonic mean of 2 real numbers
double harmonic_mean( double x, double y ){	
	if(x+y == 0.0f){
		printf("ERROR: Harmonic mean evaluated at 0\n ");
		exit(EXIT_FAILURE);
	}
	return (2.0f*(x*y))/(x+y);	
}


// Produce micro and macro averaged precision and recall
// Input is the more simple detector_stats for different classes 
classifier_stats  get_class_stats(detector_stats* all_stats , sz_short num_class ){

	classifier_stats stats = {.micro_precision=0.0f,
				  .macro_precision=0.0f,
				  .micro_recall=0.0f,
				  .macro_recall=0.0f }; 
	
	double sum_of_true_pos = 0.0f, micro_prec_denom = 0.0f, micro_recall_denom = 0.0f ;

	for(sz_short i=0; i<num_class;i++){
		sum_of_true_pos += all_stats[i].true_pos;
		micro_prec_denom += all_stats[i].true_pos + all_stats[i].false_pos;		
		micro_recall_denom += all_stats[i].true_pos + all_stats[i].false_neg;
		
		stats.macro_precision += all_stats[i].true_pos/( all_stats[i].true_pos + all_stats[i].false_pos );
		stats.macro_recall += all_stats[i].true_pos/( all_stats[i].true_pos + all_stats[i].false_neg );		
	}
	
	stats.micro_precision = sum_of_true_pos/micro_prec_denom;
	stats.micro_recall = sum_of_true_pos/micro_recall_denom;

	stats.macro_precision /= (double) num_class;
	stats.macro_recall /= (double) num_class;
        
	
	return stats;
}


// Takes array of detector statistics as input and returns array with f-1 scores for every class

void get_per_class_f1_scores(double* f1_per_class, detector_stats* all_stats, sz_short num_class ){
	
	for(sz_short i=0;i<num_class;i++){ 
		double TP = all_stats[i].true_pos;
		double FN = all_stats[i].false_neg;
		double FP = all_stats[i].false_pos;
		f1_per_class[i] = 2.0*TP / ( 2.0*TP + FN + FP );
	}
}


 
// Computes Micro and Macro f1-score
// Inputs must be one_hot structs
// f1_scores are only evaluated on the indexes denoted by unlabeled

f1_scores get_averaged_f1_scores(one_hot_mat true_labels, one_hot_mat pred_labels, sz_long* unlabeled , sz_long num_unlabeled ){
	
	if( (true_labels.num_class != pred_labels.num_class) || (true_labels.length != pred_labels.length) ){
		printf("Classifier stats ERROR: One-hot matrixes dimensions dont match\n");
		exit(EXIT_FAILURE);
	}

	int mask[true_labels.length];
	memset(mask, 0 , true_labels.length*sizeof(int));
	for(sz_long i=0;i< num_unlabeled; i++ ){ mask[unlabeled[i]]=1; }

        // Start by gathering detection statistics per each class 

	detector_stats all_stats[true_labels.num_class];

	for(sz_short i=0;i<true_labels.num_class;i++){
		all_stats[i] = get_detector_stats( true_labels.bin[i] , pred_labels.bin[i] , true_labels.length , mask );
	}

	// Use detection stats to get Micro and Macro averaged F1 scores
	
	f1_scores scores = {.micro = 0.0f,
			    .macro = 0.0f };
	
	classifier_stats class_stats = get_class_stats(all_stats , true_labels.num_class );
	
	double f1_per_class[true_labels.num_class];
	
	get_per_class_f1_scores(f1_per_class, all_stats, true_labels.num_class);
	
	scores.micro = harmonic_mean(class_stats.micro_precision, class_stats.micro_recall );
        scores.macro = mean( f1_per_class, (int)true_labels.num_class );
/*
        This definition of Macro F1 (using macro averaged precision and recall) does not seem to work well
	scores.macro = harmonic_mean(class_stats.macro_precision, class_stats.macro_recall ); //
*/	
	return scores;	
}




//Compute error rate

double accuracy(class_t* true_label, class_t* pred,  sz_long* unlabeled, sz_long num_unlabeled){
	double sum=0.0;
	for(sz_long i=0;i<num_unlabeled;i++){
			if(true_label[unlabeled[i]]==pred[unlabeled[i]])
				sum+=1.0;
		}		     

	return sum/(double)num_unlabeled;
}




