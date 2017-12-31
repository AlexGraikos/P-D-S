#include <pthread.h>

/* Struct that holds network properties */
typedef struct {
	int input_size; // Input vector size
	int output_size; // Output vector size
	int number_of_layers; // Number of network layers
	int *layer_size; // Size of network's layers
  
	float **weights; // Layer weights matrices
	float **biases; // Layer node biases
  float **layer_outputs; // Layer output matrices 
	float **deltas; // Layer delta matrices 

  float *training_inputs; // Training inputs fed to network
	float *validation_inputs; // Validation inputs to test network
  float *training_outputs; // Training outputs
	float *validation_outputs; // Validation outputs
	float *mse_vector;
  int number_of_patterns; // Number of patterns to train the network
  int training_patterns;
	int validation_patterns;
} network_description;

/* Network device properties */
typedef struct {
  float **d_weights, **d_biases, **d_layer_outputs,
				**d_deltas, *d_reference_inputs, *d_reference_outputs,
				*d_mse;
} device_network;

/* Thread arguments */
typedef struct {
  network_description net;
  float learning_rate, mse_threshold;
	float **d_weights, **d_biases;
	int id, number_of_threads, epochs;
  int *mse_validation;
	int verbosity;
  pthread_barrier_t *network_update_barrier;
  pthread_mutex_t *network_update_mutex;
} thread_args;

void printMatrix(float*, int, int);
void printMatrixKernel(float*, int, int);
void shuffleRows(float*,float*,int,int,int);

network_description load_network_description(char*,int);
void network_description_free(network_description);

void *thread_train(void*);
void train_net(cudaStream_t, device_network, network_description,
		pthread_mutex_t*, pthread_barrier_t*,int,float,float,int,int,int*,int);

device_network create_device_network(network_description);
device_network create_device_network_shared_properties(network_description);
void device_network_free(device_network, network_description);
void device_network_free_shared_properties(device_network, network_description);

inline __device__ float kernel(float value) {
  return 1.f / (1.f+exp(-value));
}

inline __device__ float dkernel(float value) {
  return (1.f-kernel(value))*kernel(value);
}

inline float h_kernel(float value) {
	return 1.f/(1.f+expf(-value));
}

inline __device__ float d_min(float a, float b) {
  return (a<b)? a: b;
}

__global__ void forwardPropagation(float*,float*,float*,float*,int,int,int,int);
__global__ void deltaL(float*,float*,float*,int,int);
__global__ void deltal(float*,float*,float*,float*,int,int,int);
__global__ void weightUpdate(float*,float*,float*,int,int,int,float,int);
__global__ void biasUpdate(float*,float*,int,int,float);
__global__ void mse(float*,float*,float*,int,int);
