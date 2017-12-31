#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"

#define network_outputs_trans(i,j) network_outputs[(i)*output_size + (j)]
#define training_outputs_trans(i,j) training_outputs[(i)*output_size + (j)]
#define square_errors_sub(i,j) square_errors_sub[(i)*(blockDim.x+1) + (j)] 
#define mse_sub(i,j) mse_sub[(i)*(blockDim.x+1) + (j)] 

/* Computes mean square error vector over all patterns
 * network_outputs and training_outputs are transposed in memory */
__global__ void mse(float *network_outputs, float *training_outputs, float *mse,
		int output_size, int patterns) {
	int row, column;
  /* Result */
  float value = 0.f;

  /* Shared memory submatrices */
  extern __shared__ float sharedMem[];
  __shared__ float *square_errors_sub;
	__shared__ float *mse_sub;

  /* Setup shared memory pointers */
  square_errors_sub = sharedMem;
	mse_sub = square_errors_sub + blockDim.y*(blockDim.x+1);

  /* Iterate over submatrices needed to compute output */
  int k;
  for (k=0; k < (int)ceilf((float)patterns/blockDim.x); k++) {

		/* Copy transposed matrix patch into shared memory */
		row = k*blockDim.x + threadIdx.y;
		column = blockIdx.y*blockDim.y + threadIdx.x;
		if (row < patterns && column < output_size) {
			square_errors_sub(threadIdx.x,threadIdx.y) = 
				0.5*powf((kernel(network_outputs_trans(row,column)) - training_outputs_trans(row,column)),2);
		} else {
		  square_errors_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

    __syncthreads();
    
    /* Row sum */
    int j;
    for (j=0; j<blockDim.x; j++) {
			value += square_errors_sub(threadIdx.y, j);
    }
   
    __syncthreads();
  }

	/* Copy result to shared memory */
	mse_sub(threadIdx.x,threadIdx.y) = value;
	__syncthreads();

	/* Use the starting threads of the block
	 * to access global memory in a coalesced way */
	int index = threadIdx.y*blockDim.y + threadIdx.x;
	if (index < blockDim.y && index+blockIdx.y*blockDim.y < output_size) {
    index += blockIdx.y*blockDim.y;
		mse[index] = mse_sub(threadIdx.y,threadIdx.x)/patterns;
	}
}
