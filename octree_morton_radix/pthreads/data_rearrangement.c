#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pthread.h"

#define DIM 3

typedef struct{
  float *Y;
  float *X;
  unsigned int *permutation_vector;
  int start;
  int finish;
} rearrangementArgs;

// Definition for usage with pThreads
void *parallel_data_rearrangement(void* arguments) {
  rearrangementArgs *args = (rearrangementArgs*) arguments;

  for(int i=args->start; i<args->finish; i++) {
    memcpy(&args->Y[i*DIM], &args->X[args->permutation_vector[i]*DIM], DIM*sizeof(float));
  }
  
  pthread_exit(NULL);
}


void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N, int nThreads){

  // Initialize the thread ids
  pthread_t *threads = (pthread_t*) malloc((nThreads-1)*sizeof(pthread_t));
  long itemsPerThread = N / nThreads;
  int t;

  // Initialize the thread attributes we want (Joinable)
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // Create the array of arguments to pass to each thread
  rearrangementArgs *arguments = (rearrangementArgs*) malloc((nThreads-1)*sizeof(rearrangementArgs));

  for(t=0; t<nThreads-1;t++) {

    arguments[t].Y = Y;
    arguments[t].X = X;
    arguments[t].permutation_vector = permutation_vector;
    arguments[t].start = t*itemsPerThread;
    arguments[t].finish = (t+1)*itemsPerThread;
    
    if(pthread_create(&threads[t], &attr, parallel_data_rearrangement, (void*)&arguments[t])) {
      printf("Could not create thread in data_rearrangement.c with id=%d\n", t);
    }

  }
  
  // Main thread does the last part
  for(int i=(nThreads-1)*itemsPerThread; i<N; i++){
    memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
  }
 
  for(t=0; t<nThreads-1; t++) {
    if(pthread_join(threads[t], NULL)) {
      printf("Could not join threads in data_rearrangement.c\n");
    }
  }
  
  free(threads);
  free(arguments);

}
