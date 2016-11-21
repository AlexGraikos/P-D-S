#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "pthread.h"

#define DIM 3

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);
}

/* Change the quantize function so that it takes an extra argument:
 * the start index of the array to compute the hash codes
*/

/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int start, int N){

  for(int i=start; i<N; i++){
    for(int j=0; j<DIM; j++){
      codes[i*DIM + j] = compute_code(X[i*DIM + j], low[j], step); 
    }
  }

}

/* We define a struct for the arguments we need to be passed to our
 * quantize function that is to be executed in parallel
*/

typedef struct{
  unsigned int *codes;
  float *X;
  float *low;
  float step;
  int start;
  int N;
} quantizeArgs;


/* The parallel implementation of the qunatize function
*/

void *parallelQuantize(void *arguments) {
  quantizeArgs *args = (quantizeArgs*) arguments;

  for(int i=args->start; i<args->N; i++){
    for(int j=0; j<DIM; j++){
      args->codes[i*DIM + j] = compute_code(args->X[i*DIM + j], args->low[j], args->step);
    }
  }
  
  pthread_exit(NULL);
}

float max_range(float *x){

  float max = -FLT_MAX;
  for(int i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N,
                        int nbins, float *min, float *max,
			int nThreads){
  
  float range[DIM];
  float qstep;

  for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
  }

  qstep = max_range(range) / nbins; // The quantization step 
  
  // Initialize the thread ids and items per thread to be assigned
  pthread_t *threads = (pthread_t*) malloc((nThreads-1) * sizeof(pthread_t)); 
  long itemsPerThread = N / nThreads;
  int t;

  // Initialize the attributes of the threads -- All Joinable for this case
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  /* Create an the array of arguments to be passed to each thread
   * Assign to each thread a portion of the dataset to compute the
   * hash codes
  */
  quantizeArgs *arguments = (quantizeArgs*) malloc((nThreads-1)*sizeof(quantizeArgs));

  for(t=0; t<nThreads-1; t++) {

    arguments[t].codes = codes;
    arguments[t].X = X;
    arguments[t].low = min;
    arguments[t].step = qstep;
    arguments[t].start = t*itemsPerThread;
    arguments[t].N = (t+1)*itemsPerThread;

    if(pthread_create(&threads[t], &attr, parallelQuantize, (void*)&arguments[t])) {
      printf("Could not create thread in hash_codes.c with id=%d\n", t);
    }
  }
  
  // The last computations will be executed by the main thread 
  quantize(codes, X, min, qstep, (nThreads-1)*itemsPerThread, N);

  // Join all threads
  for(t=0; t<nThreads-1; t++) {
    if(pthread_join(threads[t], NULL)) {
      printf("Could not join threads in hash_codes.c\n");
    }
  }

  free(threads);
  free(arguments);

}

