#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"

#define weights(i,j) weights[(i)*current_layer_size + (j)] 
#define next_layer_delta_trans(i,j) next_layer_delta[(i)*next_layer_size + (j)]
#define current_layer_delta_trans(i,j) current_layer_delta[(i)*current_layer_size + (j)]
#define weights_sub(i,j) weights_sub[(i)*(blockDim.x+1) + (j)] 
#define next_layer_delta_sub(i,j) next_layer_delta_sub[(i)*(blockDim.x+1) + (j)] 
#define current_layer_delta_sub(i,j) current_layer_delta_sub[(i)*(blockDim.x+1) + (j)] 

/* Calculate delta values of inner layer 
 * ! next_layer_delta and layer_outputs are transposed in memory */ 
__global__ void deltal(float *weights, float *next_layer_delta, float *layer_outputs,
		float *current_layer_delta, int current_layer_size , int next_layer_size, int patterns) {
  int row, column; 
  
  /* Result */
  float value = 0.f;

  /* Shared memory submatrices */
  extern __shared__ float sharedMem[];
  __shared__ float *weights_sub;
  __shared__ float *next_layer_delta_sub;
	__shared__ float *current_layer_delta_sub;

  /* Setup shared memory pointers */
  weights_sub = sharedMem;
  next_layer_delta_sub = weights_sub + blockDim.y*(blockDim.x+1);
	current_layer_delta_sub = next_layer_delta_sub + blockDim.y*(blockDim.x+1);

  /* Iterate over submatrices needed to compute output */
  int k;
  for (k=0; k < (int)ceilf((float)next_layer_size/blockDim.x); k++) {

		/* Copy transposed matrix patches into shared memory */
		row = k*blockDim.x + threadIdx.y;
		column = blockIdx.y*blockDim.y + threadIdx.x;
		if (row < next_layer_size && column < current_layer_size) {
			weights_sub(threadIdx.x,threadIdx.y) = weights(row,column);
		} else {
			weights_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

		row = blockIdx.x*blockDim.x + threadIdx.y;
		column = k*blockDim.y + threadIdx.x;
		if (row < patterns && column < next_layer_size) {
			next_layer_delta_sub(threadIdx.x,threadIdx.y) = next_layer_delta_trans(row,column);
		} else {
			next_layer_delta_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

    __syncthreads();
    
    /* Subvector product */
    int j;
    for (j=0; j<blockDim.x; j++) {
			value += weights_sub(threadIdx.y, j) * next_layer_delta_sub(j, threadIdx.x);
    }
   
    __syncthreads();
  }

  /* Copy result to shared memory */
	current_layer_delta_sub(threadIdx.y,threadIdx.x) = value;
	__syncthreads();
  
  /* Write transposed result to global memory
	 * if in limits of matrix 
	 * Switch row and column indexes to make
	 * global memory access coalesced */
	row = blockIdx.x * blockDim.x + threadIdx.y;
	column = blockIdx.y * blockDim.y + threadIdx.x;

	/* Index of layer_output element */
  int index = row*current_layer_size+column;

	/* Compute and copy result to global memory */
	if (row < patterns && column < current_layer_size) {
		current_layer_delta_trans(row,column) = current_layer_delta_sub(threadIdx.x, threadIdx.y)
			*dkernel(layer_outputs[index]);
	}
}

