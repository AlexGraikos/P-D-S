#include <stdio.h>
#include "definitions.h"

/* Calculates hadamard product between error and outputs */
__global__ void deltaL(float *network_outputs, float *training_outputs, float *delta,
		 int output_size, int number_of_patterns) {
	/* Element of the matrix we are computing */
	int row  = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	
	/* Unwrap 2d matrix to vector and perform hadamard product */
	int index = row*number_of_patterns + column;
  if (index < number_of_patterns*output_size) {
		float network_output = network_outputs[index];

		delta[index] = (kernel(network_output) - training_outputs[index])
		 *dkernel(network_output);
	}
}

