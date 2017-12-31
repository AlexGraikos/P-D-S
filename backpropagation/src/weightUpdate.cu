#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"

#define deltas_trans(i,j) deltas[(i)*current_layer_size + (j)]
#define layer_outputs_trans(i,j) layer_outputs[(i)*previous_layer_size + (j)]
#define weights(i,j) weights[(i)*previous_layer_size + (j)] 
#define deltas_sub(i,j) deltas_sub[(i)*(blockDim.x+1) + (j)] 
#define layer_outputs_sub(i,j) layer_outputs_sub[(i)*blockDim.x + (j)] 

/* Upadtes weight matrix of layer 
 * deltas and layer_outputs are transposed in memory */   
__global__ void weightUpdate(float *deltas, float *layer_outputs, float *weights,
		int current_layer_size , int previous_layer_size, int patterns, float mu, int isInput) {
  /* Element we are currently computing */
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  /* Result */
  float value = 0.f;

  /* Shared memory submatrices */
  extern __shared__ float sharedMem[];
  __shared__ float *deltas_sub;
  __shared__ float *layer_outputs_sub;

  /* Setup shared memory pointers */
  deltas_sub = sharedMem;
  layer_outputs_sub = deltas_sub + blockDim.y*(blockDim.x+1);

  /* Iterate over submatrices needed to compute output */
  int k;
  for (k=0; k < (int)ceilf((float)patterns/blockDim.x); k++) {

		/* Copy transposed matrix patch into shared memory */
		int rowT = k*blockDim.x + threadIdx.y;
		int colT = blockIdx.y*blockDim.y + threadIdx.x;
		if (rowT < patterns && colT < current_layer_size) {
			deltas_sub(threadIdx.x,threadIdx.y) = deltas_trans(rowT,colT);
		} else {
			deltas_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

		/* Copy patch into shared memory */
		rowT = k*blockDim.y + threadIdx.y;
	  if (rowT < patterns && column < previous_layer_size) {
			layer_outputs_sub(threadIdx.y,threadIdx.x) = layer_outputs_trans(rowT,column);
		} else {
			layer_outputs_sub(threadIdx.y,threadIdx.x) = 0.f;
		}

    __syncthreads();
    
    /* Subvector product */
    int j;
    for (j=0; j<blockDim.x; j++) {
			if (isInput) {
				value += deltas_sub(threadIdx.y, j) * layer_outputs_sub(j, threadIdx.x);
			} else {
				value += deltas_sub(threadIdx.y, j) * kernel(layer_outputs_sub(j, threadIdx.x));
			}
    }

    __syncthreads();
  }
  
	/* Update weights' matrix element */
	if (row < current_layer_size && column < previous_layer_size) {
		weights(row,column) = weights(row,column) - mu*value/patterns;
	}
}

