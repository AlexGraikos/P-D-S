#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include "definitions.h"

#define CUDA_CHECK_ERROR printf("Error in line %d of file %s:\t\n%s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()))

int main(int argc, char **argv) {
	/* Check arguments passed (Requires network description file)*/
	if (argc != 7) {
	  fprintf(stderr, "Usage: %s <Neural Network Description File> \
<Epochs> <Learning Rate> <MSE Threshold> <Number of Threads> <Verbosity>\n", argv[0]);
	  exit(0);
	}
 
	int epochs = atoi(argv[2]);
	float learning_rate = atof(argv[3]);
	float mse_threshold = atof(argv[4]);
  int number_of_threads = atoi(argv[5]); 
	int verbosity = atoi(argv[6]);

	/* Load network description details from file
	 * and create properties shared by all threads */
	network_description net = load_network_description(argv[1], verbosity);
  device_network d_net = create_device_network_shared_properties(net);

	/* Initialize threads */
  pthread_t* threads = (pthread_t*) malloc(number_of_threads*sizeof(pthread_t));
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	/* Initialize mutex */
	pthread_mutex_t network_update_mutex;
	pthread_mutex_init(&network_update_mutex, NULL);

	/* Initialize barrier */
	pthread_barrier_t network_update_barrier;
	pthread_barrier_init(&network_update_barrier, NULL, number_of_threads);

	/* Initialize thread arguments */
	thread_args *args = (thread_args*) malloc(number_of_threads*sizeof(thread_args));
	int *mse_validation = (int*) malloc(number_of_threads*sizeof(int));

	/* Start timer */
	struct timeval start_time;
	gettimeofday(&start_time, NULL);

	/* Create threads */
	int i;
	for (i=0; i<number_of_threads; i++) {
		/* Copy arguments to be passed */
	  args[i].net = net;
    args[i].d_weights = d_net.d_weights;
	  args[i].d_biases = d_net.d_biases;
	  args[i].number_of_threads = number_of_threads;
    args[i].mse_validation = mse_validation;
	  args[i].network_update_mutex = &network_update_mutex;
		args[i].network_update_barrier = &network_update_barrier;
		args[i].epochs = epochs;
		args[i].learning_rate = learning_rate;
		args[i].mse_threshold = mse_threshold;
		args[i].id = i;
		args[i].verbosity = verbosity;

    if (pthread_create(&threads[i], &attr, thread_train, (void*)&args[i]) != 0) {
			fprintf(stderr, "Could not create thread %d\n", i);
		}
	}

	/* Wait for threads to finish */
	for (i=0; i<number_of_threads; i++) {
		if (pthread_join(threads[i], NULL) != 0) {
			fprintf(stderr, "Could not join thread %d\n", i);
		}
	}

	/* Calculate and print time elapsed */
	struct timeval end_time;
	gettimeofday(&end_time, NULL);

	printf("Time elapsed: %f\n", (float)(end_time.tv_sec - start_time.tv_sec) +
			(float)(end_time.tv_usec - start_time.tv_usec)/1000000);

	/* Free allocated memory */
	device_network_free_shared_properties(d_net, net);
	network_description_free(net);
	free(mse_validation);
	free(args);
	free(threads);

	return 0;
}

void *thread_train(void *args) {
	/* Unwrap arguments passed */
	thread_args *t_args = (thread_args*) args;
	network_description net = t_args->net;

  int thread_id = t_args->id; 
	int number_of_threads = t_args->number_of_threads;

	/* Set pointers to thread's training data */
  int number_of_patterns = net.number_of_patterns/number_of_threads;
  net.training_inputs += thread_id*number_of_patterns*net.input_size;
	net.training_outputs += thread_id*number_of_patterns*net.output_size;
  
	/* Assign remaining training data to last thread */
  if (thread_id == number_of_threads-1)  {
		number_of_patterns = net.number_of_patterns - number_of_patterns*thread_id;
	}

	/* Split data into test and validation set 
	 * 60% For training and 40% for validation */
	net.training_patterns = (int) (0.6*number_of_patterns);
	net.validation_patterns = number_of_patterns - net.training_patterns;

  /* Set pointers to validation inputs/outputs */
	net.validation_inputs = net.training_inputs + net.training_patterns*net.input_size;
	net.validation_outputs = net.training_outputs + net.training_patterns*net.output_size;

	/* Create thread device matrices 
	 * Copy pointers to shared weights/biases matrices */
	device_network d_net;
	d_net = create_device_network(net);
	d_net.d_weights = t_args->d_weights;
  d_net.d_biases = t_args->d_biases;

	/* Specify device stream that thread will run on */
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	/* Train network */
	train_net(stream, d_net, net, t_args->network_update_mutex, t_args->network_update_barrier, 
			t_args->epochs, t_args->learning_rate, t_args->mse_threshold, thread_id, number_of_threads,
			t_args->mse_validation, t_args->verbosity);
  
  /* Free allocated memory */
	cudaStreamDestroy(stream);
  device_network_free(d_net, net);

	pthread_exit(NULL);
}

void train_net(cudaStream_t stream, device_network d_net, network_description net, pthread_mutex_t *network_update_mutex, 
		pthread_barrier_t *network_update_barrier, int epochs, float mu, float threshold, 
		int thread_id, int number_of_threads, int *mse_validation, int verbosity) {
  int i,e;
	float mse_value = 0.f;

  /* Train for epochs given */ 
  for (e=0; e<epochs; e++) {
 		printf("\n --- Epoch %d ---\n", e);

		/* --- Network Training --- */
 	  /* Shuffle training data and copy to device */
		shuffleRows(net.training_inputs, net.training_outputs, net.training_patterns, net.input_size, net.output_size);
  	cudaMemcpyAsync(d_net.d_reference_inputs, net.training_inputs, net.input_size*net.training_patterns*sizeof(float),
		 cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_net.d_reference_outputs, net.training_outputs, net.output_size*net.training_patterns*sizeof(float),
		 cudaMemcpyHostToDevice, stream);

		/* Forward propagation loop */
		for (i=0; i<net.number_of_layers; i++) {
      dim3 blockDims(32,32);
	  	dim3 gridDims(ceil((float)net.training_patterns/blockDims.x),ceil((float)net.layer_size[i]/blockDims.y));
		
		  /* Shared memory usage 
		   * 3 patches for the submatrices 
		   * The biases vector  */
		  size_t sharedMemSize = (blockDims.y*blockDims.x + 2*blockDims.y*(blockDims.x+1) + 
			 net.layer_size[i])*sizeof(float);
 	  
			/* First layer does not apply network's kernel function to inputs */
		  if (i == 0) {
	      forwardPropagation<<<gridDims, blockDims, sharedMemSize>>>
	       (d_net.d_weights[i], d_net.d_biases[i], d_net.d_reference_inputs,
					d_net.d_layer_outputs[i], net.layer_size[i], net.input_size, net.training_patterns, 1);
		  } else {
		    forwardPropagation<<<gridDims, blockDims, sharedMemSize>>>
		     (d_net.d_weights[i], d_net.d_biases[i], d_net.d_layer_outputs[i-1],
					d_net.d_layer_outputs[i], net.layer_size[i], net.layer_size[i-1],
	        net.training_patterns, 0);
		  }
    }
  
	  /* Backwards propagation of delta */
	  for (i=net.number_of_layers-1; i>=0; i--) {
	    dim3 blockDims(32,32);
	    dim3 gridDims(ceil((float)net.training_patterns/blockDims.x),
	     ceil((float)net.layer_size[i]/blockDims.y));
      /* 3 patches needed for matrix multiplication */
	    size_t shared_mem_size = (3*(blockDims.x+1)*blockDims.y)*sizeof(float);

		  /* Final layer kernel differs from rest */
		  if (i == net.number_of_layers-1) {
	      deltaL<<<gridDims, blockDims>>>
	  	   (d_net.d_layer_outputs[i], d_net.d_reference_outputs, d_net.d_deltas[i],
          net.output_size, net.training_patterns);
		  } else {
 	      deltal<<<gridDims, blockDims, shared_mem_size>>>
		     (d_net.d_weights[i+1], d_net.d_deltas[i+1], d_net.d_layer_outputs[i], d_net.d_deltas[i], net.layer_size[i],
		      net.layer_size[i+1], net.training_patterns);
			}
	  }

	  /* Update weights */
		cudaStreamSynchronize(stream);
    pthread_barrier_wait(network_update_barrier);
		pthread_mutex_lock(network_update_mutex);

	  for (i=0; i<net.number_of_layers; i++) {
	    dim3 blockDims(32,32);
	    dim3 gridDims(ceil((float)(i-1>=0? net.layer_size[i-1]: net.input_size)/blockDims.x),
					ceil((float)net.layer_size[i]/blockDims.y));
      size_t shared_mem_size = (blockDims.y*(blockDims.x+1) +
					blockDims.y*blockDims.x)*sizeof(float);
    
		  if (i == 0) {
        weightUpdate<<<gridDims, blockDims, shared_mem_size>>>
		     (d_net.d_deltas[i], d_net.d_reference_inputs, d_net.d_weights[i], net.layer_size[i],
		      net.input_size, net.training_patterns, mu, 1);  
		  } else {
        weightUpdate<<<gridDims, blockDims, shared_mem_size>>>
		     (d_net.d_deltas[i], d_net.d_layer_outputs[i-1], d_net.d_weights[i], net.layer_size[i],
		      net.layer_size[i-1], net.training_patterns, mu, 0);  
		  }
	  }

	  /* Update biases */
	  for (i=0; i<net.number_of_layers; i++) {
		  dim3 blockDims(32,32);
		  dim3 gridDims(1,ceil((float)net.layer_size[i]/blockDims.y));
		  size_t shared_mem_size = (2*blockDims.y*(blockDims.x+1))*sizeof(float);

		  biasUpdate<<<gridDims, blockDims, shared_mem_size>>>
		   (d_net.d_deltas[i], d_net.d_biases[i], net.layer_size[i], net.training_patterns, mu);
	  }

		pthread_mutex_unlock(network_update_mutex);
		pthread_barrier_wait(network_update_barrier);

    /* --- Network Valiation --- */
 	  /* Shuffle validation data and copy to device */
		shuffleRows(net.validation_inputs, net.validation_outputs, net.validation_patterns, net.input_size, net.output_size);
  	cudaMemcpyAsync(d_net.d_reference_inputs, net.validation_inputs, net.input_size*net.validation_patterns*sizeof(float),
		 cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(d_net.d_reference_outputs, net.validation_outputs, net.output_size*net.validation_patterns*sizeof(float),
		 cudaMemcpyHostToDevice, stream);

		if (verbosity) {
			printf("--- Validation Outputs ---\n");
			printMatrix(net.validation_outputs, net.validation_patterns, net.output_size);
	  }	

		/* Forward propagation */
		for (i=0; i<net.number_of_layers; i++) {
      dim3 blockDims(32,32);
	  	dim3 gridDims(ceil((float)net.validation_patterns/blockDims.x),ceil((float)net.layer_size[i]/blockDims.y));
		
		  /* Shared memory usage 
		   * 3 patches for the submatrices 
		   * The biases vector  */
		  size_t sharedMemSize = (blockDims.y*blockDims.x + 2*blockDims.y*(blockDims.x+1) + 
			 net.layer_size[i])*sizeof(float);
 	  
			/* First layer does not apply network's kernel function to inputs */
		  if (i == 0) {
	      forwardPropagation<<<gridDims, blockDims, sharedMemSize>>>
	       (d_net.d_weights[i], d_net.d_biases[i], d_net.d_reference_inputs, d_net.d_layer_outputs[i], net.layer_size[i], net.input_size,
				  net.validation_patterns, 1);
		  } else {
		    forwardPropagation<<<gridDims, blockDims, sharedMemSize>>>
		     (d_net.d_weights[i], d_net.d_biases[i], d_net.d_layer_outputs[i-1], d_net.d_layer_outputs[i], net.layer_size[i], net.layer_size[i-1],
	        net.validation_patterns, 0);
		  }
    }
   
		if (verbosity) {
    	/* Copy network outputs to host and print */
	  	printf("--- Network Outputs ---\n");
			int i = net.number_of_layers-1;
		  cudaMemcpyAsync(net.layer_outputs[i], d_net.d_layer_outputs[i], 
		  net.layer_size[i]*net.validation_patterns*sizeof(float), cudaMemcpyDeviceToHost, stream);
      printMatrixKernel(net.layer_outputs[i], net.validation_patterns,	net.layer_size[i]);
		}

	  /* Mean Square Error */
    {
      /* Calculate MSE over validation patterns */
      dim3 blockDims(32,32);
      dim3 gridDims(1,ceil((float)net.layer_size[net.number_of_layers-1]/blockDims.y));
      size_t shared_mem_size = (2*blockDims.y*(blockDims.x+1))*sizeof(float);

	    mse<<<gridDims, blockDims, shared_mem_size>>>
		   (d_net.d_layer_outputs[net.number_of_layers-1], d_net.d_reference_outputs, d_net.d_mse,
		    net.output_size, net.validation_patterns);

	    cudaMemcpyAsync(net.mse_vector, d_net.d_mse, net.output_size*sizeof(float), cudaMemcpyDeviceToHost, stream);
  
			/* Calculate total MSE */
	    for (i=0; i<net.output_size; i++) {
        mse_value += net.mse_vector[i];
	    }
			mse_value /= net.output_size;
	   	printf("MSE: %f\n", mse_value);

			/* If MSE is lower than threshold given update validation vector */
	    if (mse_value < threshold) {
				mse_validation[thread_id] = 1;
	    } else {
				mse_validation[thread_id] = 0;
			}

			/* If all threads have reached thershold mse stop training 
			 * MSE will be zeroed for next iteration */
			pthread_barrier_wait(network_update_barrier);
			for (i=0; i<number_of_threads; i++) {
				if (mse_validation[i] == 0) {
					mse_value = 0.f;
				}
			}

			/* If MSE has not been reset we have reached training goal */
			if (mse_value != 0.f) {
				printf("Goal reached, stopping training\n");
		 		break;
			}
		}
  }
}
