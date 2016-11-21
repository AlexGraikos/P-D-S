#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"

#define DIM 3

inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

// Struct for argument passing
typedef struct{
  unsigned long *mcodes;
  unsigned int *codes;
  int start;
  int finish;
} encodeArgs;

// Parallel implementation of basic morton_encoding function
void *parallel_encode_bits(void *arguments) {
  encodeArgs *args = (encodeArgs*)arguments;

  for(int i=args->start; i<args->finish; i++) {
    args->mcodes[i] = mortonEncode_magicbits(args->codes[i*DIM], args->codes[i*DIM + 1], args->codes[i*DIM +2]);
  }
  
  pthread_exit(NULL);
}

/* The function that transform the morton codes into hash codes */ 
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level, int nThreads){
  
  // Initialize thread ids and codes to be assigned to each thread
  pthread_t *threads = (pthread_t*) malloc((nThreads-1)*sizeof(pthread_t)); 
  long itemsPerThread = N / nThreads;
  int t;

  // Initialize the thread attributes
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // Create array of arguments
  encodeArgs *arguments = (encodeArgs*) malloc((nThreads-1)*sizeof(encodeArgs));
  
  // Assing a part of the vector to each thread to compute
  for(t=0; t<nThreads-1;t++) {

    // Init arguments
    arguments[t].mcodes = mcodes;
    arguments[t].codes = codes;
    arguments[t].start = t*itemsPerThread;
    arguments[t].finish = (t+1)*itemsPerThread;

    if(pthread_create(&threads[t], &attr, parallel_encode_bits, (void*)&arguments[t])) {
      printf("Could not create thread in morton_encoding.c with id=%d\n", t);
    }
  }

  for(int i=(nThreads-1)*itemsPerThread; i<N; i++){
    // Compute the morton codes from the hash codes using the magicbits mathod
    mcodes[i] = mortonEncode_magicbits(codes[i*DIM], codes[i*DIM + 1], codes[i*DIM + 2]);
  }
  
  // Wait for threads to finish
  for(t=0; t<nThreads-1; t++) {
    if(pthread_join(threads[t], NULL)) {
      printf("Could not join threads in morton_encoding.c\n");
    }
  }

  free(threads);
  free(arguments);
}


