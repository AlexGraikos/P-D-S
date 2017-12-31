#include <stdio.h>
#include "definitions.h"

#define weights(k,i,j) weights[(k)][(i)*((k)-1>=0? net.layer_size[(k)-1]: net.input_size) + (j)]
#define biases(k,i) biases[(k)][(i)]
#define training_inputs(i,j) training_inputs[(i)*net.input_size + (j)]
#define training_outputs(i,j) training_outputs[(i)*net.output_size + (j)]
#define A(i,j) A[(i)*n + (j)]
#define B(i,j) B[(i)*k + (j)]

/* Prints elements of matrix A (m x n) */
void printMatrix(float *A, int m, int n) {
  int i,j;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      printf("%f ", A(i,j));
    }
    putc('\n', stdout);
  }
}

/* Prints elements of matrix A (m x n) applying the kernel function */
void printMatrixKernel(float *A, int m, int n) {
  int i,j;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      printf("%f ", h_kernel(A(i,j)));
    }
    putc('\n', stdout);
  }
}

/* Shuffles rows of matrices A,B */
void shuffleRows(float *A, float *B, int m, int n, int k) {
	int i,j,random_index;
	float temp;
	for (i=0; i<m; i++) {
		/* Swap current row with random matrix row */
		//random_index = rand() % m; - Limited by RAND_MAX
		random_index = (int)((float)rand() / RAND_MAX * m);

		for (j=0; j<n; j++) {
			temp = A(i,j);
			A(i,j) = A(random_index,j);
			A(random_index,j) = temp;
		}

		for (j=0; j<k; j++) {
			temp = B(i,j);
			B(i,j) = B(random_index,j);
			B(random_index,j) = temp;
		}
	}
}			

/* Loads network properties from file to network_description struct */
network_description load_network_description(char *file, int verbosity) {
  network_description net;
  int i, j, normalize;

  /* Read network properties from file */
  FILE *network_file = fopen(file, "r");
  if (network_file == NULL) {
    fprintf(stderr, "Could not open file %s\n", file);
    exit(0);
  }

  /* Read input size, output size and number of layers */
	fscanf(network_file, "%d %d\n%d\n", &net.input_size, &net.output_size, &net.number_of_layers);
  printf("=== Network Description ===\nTraining input size: %d, Training output size: %d, Network Layers: %d\n",
      net.input_size, net.output_size, net.number_of_layers);

  /* Create layer size array and read layer sizes from file */
  net.layer_size  = (int*) malloc(net.number_of_layers*sizeof(int));
  for (i=0; i<net.number_of_layers; i++) {
    fscanf(network_file, "%d ", net.layer_size+i);

    printf("Layer: %d Size: %d\n", i+1, net.layer_size[i]);
  }

	/* Warn if output layer does not match output size */
	if (net.layer_size[net.number_of_layers-1] != net.output_size) {
		printf("-- Output layer size is not equal to training data output size!\n");
	}

  /* Allocate memory for network matrices */
  net.weights = (float**) malloc(net.number_of_layers*sizeof(float*));
	net.biases = (float**) malloc(net.number_of_layers*sizeof(float*));
  net.layer_outputs = (float**) malloc(net.number_of_layers*sizeof(float*));
  net.deltas = (float**) malloc(net.number_of_layers*sizeof(float*));

  /* Number of patterns to train the network */
  fscanf(network_file, "%d\n", &net.number_of_patterns);

	/* Normalization of data */
	fscanf(network_file, "%d\n", &normalize);

  /* Seed rng */
  srand(time(NULL));

  for (i=0; i<net.number_of_layers; i++) {
    /* Layer k matrix has layer_size[k] x layer_size[k-1] size */
    int previous_layer_size = i-1>=0? net.layer_size[i-1]: net.input_size;

    /* Allocate current layer's matrices */
    net.weights[i] = (float*) malloc(net.layer_size[i]*previous_layer_size*sizeof(float));
		net.biases[i] = (float*) malloc(net.layer_size[i]*sizeof(float));
    net.layer_outputs[i] = (float*) malloc(net.layer_size[i]*net.number_of_patterns*sizeof(float));
    net.deltas[i] = (float*) malloc(net.layer_size[i]*net.number_of_patterns*sizeof(float));


    /* Assign small random values to weights/biases */
    int k,l;
    for (k=0; k<net.layer_size[i]; k++) {
			/* Set biases' values */
			net.biases(i,k) = (float)rand() / RAND_MAX / 10;

      for (l=0; l<(i-1>=0? net.layer_size[i-1]: net.input_size); l++) {
				/* Set weights' values */
        net.weights(i,k,l) = (float)rand() / RAND_MAX / 10;
      }
    }

    /* Print matrices */
		if (verbosity) {
			printf("Layer %d\n",i+1);
    	printf("Weights: %d\n", i+1);
    	printMatrix(net.weights[i], net.layer_size[i], previous_layer_size);
			printf("Biases: %d\n", i+1);
			printMatrix(net.biases[i], net.layer_size[i], 1);
		}
  }

  /* Create training inputs matrix 
   * Matrix created is transposed to fit with the 
	 * forwardPropagation kernel */
  net.training_inputs = (float*) malloc(net.input_size*net.number_of_patterns*sizeof(float));

  /* Create training outputs matrix
   * Matrix created is transposed to fit with deltaL kernel */
  net.training_outputs = (float*) malloc(net.output_size*net.number_of_patterns*sizeof(float));

  /* Create MSE vector to hold batch mse values*/
  net.mse_vector = (float*) malloc(net.output_size*sizeof(float)); 

  /* Read training inputs and outputs matrices from file (transposed) */

  /* Find max values for input normalization */
	float *max = (float*) malloc(net.input_size*sizeof(float));

  for (i=0; i<net.number_of_patterns; i++) {
    for (j=0; j<net.input_size; j++) {
      fscanf(network_file, "%f,", &net.training_inputs(i,j));

			if (max[j] < abs(net.training_inputs(i,j))) {
				max[j] = abs(net.training_inputs(i,j));
			}
    }

    for (j=0; j<net.output_size-1; j++) {
      fscanf(network_file, "%f,", &net.training_outputs(i,j));
    }
    fscanf(network_file, "%f\n", &net.training_outputs(i,j));
  }

  if (verbosity) {
		printf("Training Inputs\n");
		printMatrix(net.training_inputs, net.number_of_patterns, net.input_size);
		printf("Training Outputs\n");
		printMatrix(net.training_outputs, net.number_of_patterns, net.output_size);
	}
	
  /* Normalize training inputs */
	if (normalize) {
  	for (i=0; i<net.number_of_patterns; i++) {
    	for (j=0; j<net.input_size; j++) {
				net.training_inputs(i,j) /= max[j];
    	}
		}
	}
	free(max);

	shuffleRows(net.training_inputs, net.training_outputs, net.number_of_patterns,
			net.input_size, net.output_size);

  return net;
}

/* Initialized device matrices for neural net */
device_network create_device_network(network_description net) {
	device_network d_net;
	int i;

	/* Allocate device memory for each matrix in the network 
	 * To do this allocate an array of pointers to host memory
	 * Each pointer will point to the device memory we require 
	 * Since all threads share the same weights and biases we only allocate them once */
	d_net.d_layer_outputs = (float**) malloc(net.number_of_layers*sizeof(float*));
	d_net.d_deltas = (float**) malloc(net.number_of_layers*sizeof(float*));

	for (i=0; i<net.number_of_layers; i++) {
		cudaMalloc(d_net.d_layer_outputs+i, net.layer_size[i]*net.training_patterns*sizeof(float)); 
		cudaMalloc(d_net.d_deltas+i, net.layer_size[i]*net.training_patterns*sizeof(float));
  }

	/* Allocate memory for training inputs/outputs 
	 * We allocate memory for training patterns since 
	 * validation patterns are less */
	cudaMalloc(&d_net.d_reference_inputs, net.input_size*net.training_patterns*sizeof(float));
	cudaMalloc(&d_net.d_reference_outputs, net.output_size*net.training_patterns*sizeof(float));

  /* Allocate memory for mse calculation */
  cudaMalloc(&d_net.d_mse, net.output_size*sizeof(float)); 

	return d_net;
}

device_network create_device_network_shared_properties(network_description net) {
  device_network d_net;
	int i;

	d_net.d_weights = (float**) malloc(net.number_of_layers*sizeof(float*));
	d_net.d_biases = (float**) malloc(net.number_of_layers*sizeof(float*));

	for (i=0; i<net.number_of_layers; i++) {
	  int previous_layer_size = i-1>=0? net.layer_size[i-1]: net.input_size;
		
		cudaMalloc(d_net.d_weights+i, net.layer_size[i]*previous_layer_size*sizeof(float));
		cudaMalloc(d_net.d_biases+i, net.layer_size[i]*sizeof(float));

			/* Copy weights/biases from host to device */
		cudaMemcpy(d_net.d_weights[i], net.weights[i], net.layer_size[i]*previous_layer_size*sizeof(float),
		 cudaMemcpyHostToDevice);
	  cudaMemcpy(d_net.d_biases[i], net.biases[i], net.layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
	}

	return d_net;
}

/* Frees network allocated memory */
void network_description_free(network_description net) {
	int i;
	for (i=0; i<net.number_of_layers; i++) {
		free(net.weights[i]);
		free(net.biases[i]);
		free(net.layer_outputs[i]);
		free(net.deltas[i]);
	}
	free(net.layer_size);
	free(net.weights);
	free(net.biases);
	free(net.layer_outputs);
	free(net.deltas);
	free(net.training_inputs);
	free(net.training_outputs);
  free(net.mse_vector);
}

void device_network_free(device_network d_net, network_description net) {
	/* Free reserved memory */
  int i;
	for (i=0; i<net.number_of_layers; i++) {
		cudaFree(d_net.d_layer_outputs[i]);
		cudaFree(d_net.d_deltas[i]);
	}
	cudaFree(d_net.d_reference_inputs);
	cudaFree(d_net.d_reference_outputs);
	cudaFree(d_net.d_mse);
  free(d_net.d_layer_outputs);
	free(d_net.d_deltas);
}

void device_network_free_shared_properties(device_network d_net, network_description net) {
	/* Free reserved memory */
  int i;
	for (i=0; i<net.number_of_layers; i++) {
	  cudaFree(d_net.d_weights[i]);
	  cudaFree(d_net.d_biases[i]);
	}
	free(d_net.d_weights);
	free(d_net.d_biases);
}

