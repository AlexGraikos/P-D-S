#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "omp.h"

#define MAXBINS 8


inline void swap_long(unsigned long int **x, unsigned long int **y){
  
  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void truncated_radix_sort(unsigned long int *morton_codes, 
			  unsigned long int *sorted_morton_codes, 
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N, 
			  int population_threshold,
			  int sft, int lv){

  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};
  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;

  if(N<=0){

    return;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    
    // Doing this in parallel adds a lot of time consumption :/
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 

    return;
  }
  else{

    level_record[0] = lv;
    // Find which child each point belongs to 

    /* Implementation with critical (or atomic)
     * adds a lot of delay -- did not use it
    */
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

    // Can do the swaps in the 2 different matrices in parallel
    #pragma omp parallel sections
    {
      //swap the index pointers  
      #pragma omp section
      swap(&index, &permutation_vector);
      
      //swap the code pointers
      #pragma omp section
      swap_long(&morton_codes, &sorted_morton_codes);
    }

    
    /* Calculate the chunk to assign to each thread, there is
     * a chance that chunk = 0 if threads > MAXBINS so in that 
     * case we assign 1 iteration to each thread
    */

    int i;
    int nThreads = omp_get_max_threads();
    unsigned long chunk = MAXBINS / nThreads;
    if (chunk == 0){
      chunk++;
    }
   
     
    /* Call the function recursively to split the lower levels */
    
    // Parallelize the for loop
    #pragma omp parallel for default(shared) schedule(dynamic, chunk) private(i) 
      for(i=0; i<MAXBINS; i++){
        int offset = (i>0) ? BinSizes[i-1] : 0;
        int size = BinSizes[i] - offset;
      
        truncated_radix_sort(&morton_codes[offset], 
  			     &sorted_morton_codes[offset], 
		       	     &permutation_vector[offset], 
  		             &index[offset], &level_record[offset], 
			     size, 
			     population_threshold,
		             sft-3, lv+1);
      }
    


    /* Tried to parallelize with openMP tasks but
     * runtimes were much slower than with the 
     * parallel for method
    */
     
    /*
    #pragma omp parallel
    #pragma omp single nowait
    {
    for(i=0; i<MAXBINS; i++){
        int offset = (i>0) ? BinSizes[i-1] : 0;
        int size = BinSizes[i] - offset;
        
	int currThreads = omp_get_num_threads();
        printf("Currently using %d threads\n", currThreads);
	
	if(i < MAXBINS -1) {
	  #pragma omp task  
         // printf("i=%d === thread %d executing task\n", i, omp_get_thread_num());
	    truncated_radix_sort(&morton_codes[offset], 
  			     &sorted_morton_codes[offset], 
		       	     &permutation_vector[offset], 
  		             &index[offset], &level_record[offset], 
			     size, 
			     population_threshold,
		             sft-3, lv+1);
       } else {
           truncated_radix_sort(&morton_codes[offset], 
  			     &sorted_morton_codes[offset], 
		       	     &permutation_vector[offset], 
  		             &index[offset], &level_record[offset], 
			     size, 
			     population_threshold,
		             sft-3, lv+1);
      }

    }
    }
    */

  }

}


