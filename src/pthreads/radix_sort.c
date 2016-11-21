#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "pthread.h"

#define MAXBINS 8

// Mutex to avoid race conditions on thread count
pthread_mutex_t threadCountMutex = PTHREAD_MUTEX_INITIALIZER;

inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

typedef struct{
  unsigned long int **x;
  unsigned long int **y;
} swap_long_args;

// Parallel modification of swap_long so that it fits with pThreads standards
inline void* parallel_swap_long(void* arguments) {
  swap_long_args *args = (swap_long_args*) arguments;

  unsigned long int *tmp;
  tmp = args->x[0];
  args->x[0] = args->y[0];
  args->y[0] = tmp;
}

typedef struct{
  unsigned int **x;
  unsigned int **y;
} swap_args;

// Parallel version of swap
void *parallel_swap(void *arguments) {
  swap_args* args = (swap_args*) arguments;

  unsigned int *tmp;
  tmp = args->x[0];
  args->x[0] = args->y[0];
  args->y[0] = tmp;
}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;
}

// Struct used for passing arguments to radix_sort calls
typedef struct{
  unsigned long int *morton_codes;
  unsigned long int *sorted_morton_codes;
  unsigned int *permutation_vector;
  unsigned int *index;
  unsigned int *level_record;
  int N;
  int population_threshold;
  int sft;
  int lv;
  int nThreads;
  int *threadsInUse;
} radix_sortArgs;


// Modified to use with pThreads
void *truncated_radix_sort(void *arguments){
  radix_sortArgs *args = (radix_sortArgs*)arguments;
  
  // Copy arguments passed to local variables
  unsigned long int *morton_codes = args->morton_codes;
  unsigned long int *sorted_morton_codes = args->sorted_morton_codes;
  unsigned int *permutation_vector = args->permutation_vector;
  unsigned int *index = args->index;
  unsigned int *level_record = args->level_record;
  int N = args->N;
  int population_threshold = args->population_threshold;
  int sft = args->sft;
  int lv = args->lv;
  int nThreads = args->nThreads;
  int *threadsInUse = args->threadsInUse;

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;

  //printf("Current number of running threads %d\n", *threadsInUse);

  if(N<=0){

    return NULL;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 

    return NULL;
  }
  else{


    // Mutex implementation too slow
    level_record[0] = lv;
    // Find which child each point belongs to 
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)  
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }
    
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }

    // Initialize the thread attributes we need (joinable)
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
 
    /* If we have available threads to do the swap in parallel then
     * execute it
     * Did not speed up execution of the program
    */
    
    /*
    pthread_t swapThread;

    swap_args args;
    args.x = &index;
    args.y = &permutation_vector;

    pthread_mutex_lock(&threadCountMutex); 
    if(*threadsInUse < nThreads) {
      
      (*threadsInUse)++; 
      pthread_mutex_unlock(&threadCountMutex);

      if(pthread_create(&swapThread, &attr, parallel_swap, (void*)&args)) {
        printf("Could not create swap thread\n");
      }

      swap_long(&morton_codes, &sorted_morton_codes);
    
      if(!pthread_join(swapThread, NULL)) {
        pthread_mutex_lock(&threadCountMutex);
	(*threadsInUse)--;
	pthread_mutex_unlock(&threadCountMutex);

      } else {
        printf("Could not join swap thread\n");
      }

    } else {
      pthread_mutex_unlock(&threadCountMutex);
      parallel_swap((void*)&args);
      swap_long(&morton_codes, &sorted_morton_codes);
    }
    */ 

    swap(&index, &permutation_vector);
    swap_long(&morton_codes, &sorted_morton_codes);

    /* If we have any free threads assign the work to them
     * else let this thread execute it
    */
     
    pthread_t *threads = (pthread_t*) malloc(nThreads*sizeof(pthread_t));
    int t = 0;

    radix_sortArgs *recursionArgs = (radix_sortArgs*) malloc(MAXBINS*sizeof(radix_sortArgs));

    /* Call the function recursively to split the lower levels */
    for(int i=0; i<MAXBINS; i++){
      int offset = (i>0) ? BinSizes[i-1] : 0;
      int size = BinSizes[i] - offset;
      
      // Arguments to be passed to recursive call
      recursionArgs[i].morton_codes = &morton_codes[offset];
      recursionArgs[i].sorted_morton_codes = &sorted_morton_codes[offset];
      recursionArgs[i].permutation_vector = &permutation_vector[offset];
      recursionArgs[i].index = &index[offset];
      recursionArgs[i].level_record = &level_record[offset];
      recursionArgs[i].N = size;
      recursionArgs[i].population_threshold = population_threshold;
      recursionArgs[i].sft = sft-3;
      recursionArgs[i].lv = lv+1;
      recursionArgs[i].nThreads = nThreads;
      recursionArgs[i].threadsInUse = threadsInUse;
      
      pthread_mutex_lock(&threadCountMutex);
      if(*threadsInUse < nThreads && i < MAXBINS - 1)   {
	(*threadsInUse)++;
	pthread_mutex_unlock(&threadCountMutex);
	
	if(pthread_create(&threads[t], &attr, truncated_radix_sort, (void*)&recursionArgs[i])) {
	  printf("Could not create thread in radix_sort.c\n");
	}
	// Add one to the threads we have created
	t++;
      
      } else {
        pthread_mutex_unlock(&threadCountMutex);
	truncated_radix_sort((void*)&recursionArgs[i]);
      }
    
    }
    
    // Wait for any threads we created to terminate
    for(int j=0; j<t; j++) {
      if(pthread_join(threads[j], NULL)) {
        printf("Could not join threads in radix_sort.c\n");
      }
    }
    
    // Update the count of running threads
    pthread_mutex_lock(&threadCountMutex);
    *threadsInUse = *threadsInUse - t;
    pthread_mutex_unlock(&threadCountMutex);

    // Free the memory and reduce the index of threads we use
    free(threads);
    free(recursionArgs);
  } 
}

