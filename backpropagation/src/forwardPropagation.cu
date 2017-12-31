#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"

#define weights(i,j) weights[(i)*weights_cols + (j)] 
#define inputs_trans(i,j) inputs[(i)*weights_cols + (j)]
#define outputs_trans(i,j) outputs[(i)*weights_rows + (j)]
#define weights_sub(i,j) weights_sub[(i)*blockDim.x + (j)] 
#define inputs_sub(i,j) inputs_sub[(i)*(blockDim.x+1) + (j)] 
#define outputs_sub(i,j) outputs_sub[(i)*(blockDim.x+1) + (j)]

/* Layer forward propagation kernel
 * inputs are transposed in memory */
__global__ void forwardPropagation(float *weights, float *biases, float *inputs, float *outputs, 
		int weights_rows, int weights_cols, int inputs_cols, int isInput) {
  /* Element we are currently computing */
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;
	int rowT, colT;
  
  /* Result */
  float value = 0.f;

  /* Shared memory submatrices */
  extern __shared__ float sharedMem[];
  __shared__ float *weights_sub;
  __shared__ float *inputs_sub;
	__shared__ float *outputs_sub;
  __shared__ float *biases_shared; 

  /* Setup shared memory pointers */
  weights_sub = sharedMem;
  inputs_sub = weights_sub + blockDim.y*blockDim.x;
	outputs_sub = inputs_sub + blockDim.y*(blockDim.x+1);
	biases_shared = outputs_sub + blockDim.y*(blockDim.x+1);

	/* Copy biases to shared memory 
	 * If layer size is larger than number of threads
	 * in block, the copy is broken down to subcopies 
	 * biases_shared[row] indexes the neuron we are currently at */
  int k;
  for (k=0; k<(int)ceilf((float)weights_rows/blockDim.x); k++) {
    int index = threadIdx.y*blockDim.x + threadIdx.x + k*blockDim.y*blockDim.x;
		if (index < weights_rows) {
		 	biases_shared[index] = biases[index];
		}
  }
	  
  /* Iterate over submatrices needed to compute output */
  for (k=0; k < (int)ceilf((float)weights_cols/blockDim.y); k++) {

    /* Copy patch to shared memory */
		if (row < weights_rows && threadIdx.x+k*blockDim.x < weights_cols) { 
      weights_sub(threadIdx.y, threadIdx.x) = weights(row, threadIdx.x + k*blockDim.x);
    } else {
			weights_sub(threadIdx.y, threadIdx.x) = 0.f;
		}

		/* Copy transposed matrix patch to shared mem */
		rowT = blockIdx.x*blockDim.x + threadIdx.y;
		colT = k*blockDim.y + threadIdx.x;
		if (rowT < inputs_cols && colT < weights_cols) {
			inputs_sub(threadIdx.x,threadIdx.y) = inputs_trans(rowT,colT);
		} else {
			inputs_sub(threadIdx.x,threadIdx.y) = 0.f;
		}

    __syncthreads();
    
    /* Subvector product */
		int j;
    for (j=0; j<blockDim.y; j++) {
			/* Do not apply kernel function to input data */
			if (isInput) {
			  value += weights_sub(threadIdx.y, j) * inputs_sub(j, threadIdx.x);
			} else {
			  value += weights_sub(threadIdx.y, j) * kernel(inputs_sub(j, threadIdx.x));
			}
    }
    
		__syncthreads();
  }
  
	/* Copy result to shared memory */
	outputs_sub(threadIdx.y,threadIdx.x) = value;
	__syncthreads();

	/* Result is transposed in memory
	 * for reusage by next forward propagation loop
	 * Compute the new row and column indexes
	 * for coalesced access in global memory */
	row = blockIdx.x*blockDim.x + threadIdx.y;
	column = blockIdx.y*blockDim.y + threadIdx.x;

	/* If new indexes in limits of C_trans */
	if (column < weights_rows && row < inputs_cols) {
	  
		/* Write layer ouput to memory */
    outputs_trans(row,column) = outputs_sub(threadIdx.x,threadIdx.y)
			+ biases_shared[column];
	}
} 
