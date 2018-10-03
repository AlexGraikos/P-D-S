#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "cilk/cilk.h"

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
    
    // === Each memcpy is called to a different array ===
    // Did not parallelize due to slowdown 
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the permutation vector 
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 
    return;
  }
  else{

    level_record[0] = lv;
    // Find which child each point belongs to 
    
    /* Can't execute in parallel since all loops
     * share the same memory in BinSizes array and
     * can tamper with it at the same moment leading
     * to errors [2 or more threads may want to increment
     * the bin size of ii at the same time]
    */
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)  
    /*
     * Can't be executed in parallel
     * offset is shared memory
    */
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }
    
    // BinCursor array is shared can't execute
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }
    
    //swap the index pointers  
    cilk_spawn swap(&index, &permutation_vector);
    //swap the code pointers 
    swap_long(&morton_codes, &sorted_morton_codes);
    cilk_sync;

    /* Call the function recursively to split the lower levels */
    // === Each level is independent from the other levels ===
    for(int i=0; i<MAXBINS; i++){
      int offset = (i>0) ? BinSizes[i-1] : 0;
      int size = BinSizes[i] - offset;
      
      // If we are at the last call do not call a worker to take on the spawn
      if(i < MAXBINS -1){
        cilk_spawn truncated_radix_sort(&morton_codes[offset], 
 			     &sorted_morton_codes[offset], 
			     &permutation_vector[offset], 
			     &index[offset], &level_record[offset], 
			     size, 
			     population_threshold,
			     sft-3, lv+1);
      }else{
        truncated_radix_sort(&morton_codes[offset], 
 			     &sorted_morton_codes[offset], 
			     &permutation_vector[offset], 
			     &index[offset], &level_record[offset], 
			     size, 
			     population_threshold,
			     sft-3, lv+1);
      }


    }
    cilk_sync;
  } 
}

