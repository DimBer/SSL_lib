///////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routines for handling compressed-sparse-row (CSR) graphs
 ( allocating , copying, normalizing, scaling, mat-vec, mat-mat, freeing )

 Dimitris Berberidis 
 University of Minnesota 2017-2018
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

#include "csr_handling.h"
#include "my_defs.h"
#include "my_utils.h"



// Make a copy of graph with edges multiplied by some scalar
csr_graph csr_deep_copy_and_scale(csr_graph graph, double scale ){
         
	csr_graph graph_temp;
	
	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted. 
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value=(double*)malloc(graph.nnz*sizeof(double));

	graph_temp.csr_column=(uint64_t*)malloc(graph.nnz*sizeof(uint64_t));
	
	graph_temp.csr_row_pointer=(uint64_t*)malloc((graph.num_nodes+1)*sizeof(uint64_t));

	graph_temp.degrees=(uint64_t*)malloc(graph.num_nodes*sizeof(uint64_t));
	
	graph_temp.num_nodes=graph.num_nodes;
        
        graph_temp.nnz=graph.nnz;

	//copy data
        
        memcpy(graph_temp.csr_row_pointer,graph.csr_row_pointer, (graph.num_nodes+1)*sizeof(uint64_t));
        
        memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes*sizeof(uint64_t));        

        memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz*sizeof(uint64_t));
        
        uint64_t i;
        for(i=0;i<graph.nnz;i++){
        	graph_temp.csr_value[i]=scale*graph.csr_value[i];
        }
		
	return graph_temp;
}

// Make a copy of graph with edges multiplied by some scalar
csr_graph csr_deep_copy(csr_graph graph){
         
	csr_graph graph_temp;
	
	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted. 
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value=(double*)malloc(graph.nnz*sizeof(double));

	graph_temp.csr_column=(uint64_t*)malloc(graph.nnz*sizeof(uint64_t));
	
	graph_temp.csr_row_pointer=(uint64_t*)malloc((graph.num_nodes+1)*sizeof(uint64_t));

	graph_temp.degrees=(uint64_t*)malloc(graph.num_nodes*sizeof(uint64_t));
	
	graph_temp.num_nodes=graph.num_nodes;
        
        graph_temp.nnz=graph.nnz;

	//copy data
        
        memcpy(graph_temp.csr_row_pointer,graph.csr_row_pointer, (graph.num_nodes+1)*sizeof(uint64_t));
        
        memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes*sizeof(uint64_t));        

        memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz*sizeof(uint64_t));

        memcpy(graph_temp.csr_value, graph.csr_value, graph.nnz*sizeof(double));
        	
	return graph_temp;
}

//Return an array with multiple copies of the input graph
csr_graph* csr_mult_deep_copy( csr_graph graph, uint8_t num_copies ){
	uint8_t i;
	csr_graph* graph_array=(csr_graph*)malloc(num_copies*sizeof(csr_graph));	
	for(i=0;i<num_copies;i++){
		graph_array[i]=csr_deep_copy(graph);
	}	
	return graph_array;
}


//Allocate memory and create csr_graph from edgelist input
csr_graph csr_create( const uint64_t** edgelist, uint64_t num_edges ){
	
	csr_graph graph;
	
	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted. 
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph.csr_value=(double*)malloc(2*num_edges*sizeof(double));

	graph.csr_column=(uint64_t*)malloc(2*num_edges*sizeof(uint64_t));
	
	graph.csr_row_pointer=(uint64_t*)malloc(2*num_edges*sizeof(uint64_t));

	graph.degrees=(uint64_t*)malloc(2*num_edges*sizeof(uint64_t));

	//Convert undirected edge list to CSR format and return graph size
	graph.num_nodes = edge_list_to_csr(edgelist, graph.csr_value, graph.csr_column, graph.csr_row_pointer,
					   num_edges, &graph.nnz, graph.degrees); 

	printf("nnz %"PRIu64"\n",graph.nnz);
	printf("nume_edges %"PRIu64"\n",num_edges);
	

	graph.csr_row_pointer = realloc(graph.csr_row_pointer, (graph.num_nodes+1)*sizeof(uint64_t));
	graph.csr_value = realloc(graph.csr_value, graph.nnz*sizeof(double));
	graph.csr_column = realloc(graph.csr_column, graph.nnz*sizeof(uint64_t));
	graph.degrees = realloc(graph.degrees, graph.num_nodes*sizeof(uint64_t));
	
		
	return graph;
	
}

// Free memory allocated to csr_graph
void csr_destroy( csr_graph graph ){
	free(graph.csr_value);
	free(graph.csr_column);
	free(graph.csr_row_pointer);
	free(graph.degrees);
}

// Free memory allocated to array of csr_graphs
void csr_array_destroy(csr_graph* graph_array, uint8_t num_copies){
	uint8_t i;
	for(i=0;i<num_copies;i++){ csr_destroy(graph_array[i]); }
	free(graph_array);
}


//Subroutine: modify csr_value to be column stochastic
//First find degrees by summing element of each row
//Then go through values and divide by corresponding degree (only works for undirected graph)
void make_CSR_col_stoch(csr_graph* graph){
	uint64_t i;
	
	for(i=0;i<graph->nnz;i++){
		graph->csr_value[i]=graph->csr_value[i]/(double)graph->degrees[graph->csr_column[i]];
	}
	
}



//Convert directed edgelist into undirected csr_matrix
uint64_t edge_list_to_csr(const uint64_t** edge, double* csr_value, uint64_t* csr_column,
			  uint64_t* csr_row_pointer, uint64_t len, uint64_t* nnz, uint64_t* degrees){
	//Start bu making a 2D array twice the size where (i,j) exists for every (j,i)
	uint64_t i,count_nnz;
	uint64_t** edge_temp=(uint64_t **)malloc(2*len * sizeof(uint64_t *));
	for(i=0;i<2*len;i++)
		edge_temp[i]=(uint64_t*)malloc(2*sizeof(uint64_t));

	for(i=0;i<len;i++){
		edge_temp[i][0]= edge[i][0];
		edge_temp[i][1]= edge[i][1];
		edge_temp[i+len][1]= edge[i][0];
		edge_temp[i+len][0]= edge[i][1];
	}
	//QuickSort buffer_temp with respect to first column (Study and use COMPARATOR function for this)
	qsort(edge_temp, 2*len, sizeof(edge_temp[0]), compare); 
	//for(i=0;i<20;i++)
	//printf("%d %d\n",buffer_temp[i][0],buffer_temp[i][1]);

	//The first collumn of sorted array readily gives csr_row_pointer (just loop through and look for j s.t. x[j]!=x[j-1])
	//Not sure yet but i probably need to define small dynamic subarray with elements of second collumn and
	// sort it before stacking it it into csr_column (A: I dont need to)
	//This can all be done together in one loop over sorted buffer_temp
	csr_row_pointer[0]=0;
	csr_value[0]=1.0;
	csr_column[0]=edge_temp[0][1]-1;
	uint64_t j=1;
	count_nnz=1;
	for(i=1;i<2*len;i++){
		if(!(edge_temp[i-1][0]==edge_temp[i][0] && edge_temp[i-1][1]==edge_temp[i][1])){
//			printf("%"PRIu64" %"PRIu64"\n",edge_temp[i][0],edge_temp[i][1]);
			csr_value[count_nnz]=1.0;
			csr_column[count_nnz]=edge_temp[i][1]-1;
			if(edge_temp[i][0]!=edge_temp[i-1][0]){
				csr_row_pointer[j]=count_nnz;
				j++;}
			count_nnz++;
		}
	}
	csr_row_pointer[j]=count_nnz;
	*nnz=count_nnz;
	
	for(i=0;i<j;i++){degrees[i]=csr_row_pointer[i+1]-csr_row_pointer[i];}
	
	//Free temporary list
	for(i=0;i<2*len;i++)
	{free(edge_temp[i]);}
	free(edge_temp);
	return j;
}




//Subroutine: take x, multiply with csr matrix from right and store result in y
void my_CSR_matvec( double* y ,double* x  , csr_graph graph){
	uint64_t i,j;

	for(i=0;i<graph.num_nodes;i++)
		y[i]=0.0;

	
	for(i=0;i<graph.num_nodes;i++)
	{
		for(j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++)
		{y[i]+=x[graph.csr_column[j]]*graph.csr_value[j];}
	}
}


//Subroutine: take X, multiply with csr matrix from right and store result in Y
void my_CSR_matmat( double* Y ,double* X  , csr_graph graph, uint16_t M, uint16_t from, uint16_t to){
	uint64_t i,j;
	uint16_t k;
	for(i=0;i<graph.num_nodes;i++){for(j=from;j<to;j++){Y[i*M+j]=0.0f;}}

	for(i=0;i<graph.num_nodes;i++){	
		for(j=graph.csr_row_pointer[i];j<graph.csr_row_pointer[i+1];j++){
			for(k=from;k<to;k++){ Y[i*M + k] +=  X[ M*graph.csr_column[j] + k]*graph.csr_value[j];}
		}
	}
}







