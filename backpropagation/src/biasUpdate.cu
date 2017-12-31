#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"

#define deltas_trans(i,j) deltas[(i)*layer_size + (j)]
#define deltas_sub(i,j) deltas_sub[(i)*(blockDim.x+1) + (j)] 
#define bias_sub(i,j) bias_sub[(i)*(blockDim.x+1) + (j)]

/* Update biases from batch deltas calculated 
 * ! deltas is transposed in memory */
__global__ void biasUpdate(float *deltas, float *biases, int layer_size, 
		int patterns, float mu) {
  /* Result */
  float value = 0.f;

  /* Shared memory submatrices */
  extern __shared__ float sharedMem[];
  __shared__ float *deltas_sub;
	__shared__ float *bias_sub;

  /* Setup shared memory pointers */
  deltas_sub = sharedMem;
	bias_sub = deltas_sub + blockDim.y*(blockDim.x+1);

  /* Iterate over submatrices needed to compute output */
  int k;
  for (k=0; k < (int)ceilf((float)patterns/blockDim.x); k++) {

		/* Copy transposed matrix patch into shared memory*/
		int rowT = k*blockDim.x + threadIdx.y;
		int colT = blockIdx.y*blockDim.y + threadIdx.x;
		if (rowT < patterns && colT < layer_size) {
			deltas_sub(threadIdx.x,threadIdx.y) = deltas_trans(rowT,colT);
		} else {
			deltas_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

    __syncthreads();
    
    /* Row sum */
    int j;
    for (j=0; j<blockDim.x; j++) {
			value += deltas_sub(threadIdx.y, j);
    }
   
    __syncthreads();
  }

	/* Copy result to shared memory */
	bias_sub(threadIdx.y,threadIdx.x) = value;
  __syncthreads();

	/* Use the starting threads of the block
	 * to access global memory in a coalesced way */
	int index = threadIdx.y*blockDim.y + threadIdx.x;
	if (index < blockDim.y && index+blockIdx.y*blockDim.y < layer_size) {
		index += blockIdx.y*blockDim.y;
	  biases[index] = biases[index] - mu*bias_sub(threadIdx.x,threadIdx.y)/patterns;
	}
}
