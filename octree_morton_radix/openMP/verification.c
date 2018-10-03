#include "stdio.h"
#include "stdlib.h"
#include "utils.h"
#include "math.h"

#define DIM 3

int cmpfunc(const void *a, const void *b){
   return ( *(int *)a - *(int *)b );
}

int check_index(unsigned int *index, int N){

  /* sort the permutation vector */
  qsort(index, N, sizeof(unsigned int), cmpfunc);


  /* Check if all indexes are present in the input vector */
  int count = 0;
  for(int i=0; i<N; i++){
    count += (index[i] == i);
  }

  return(count == N);

}



int check_codes(float *X, unsigned long int *morton_codes, 
		unsigned int *level_record, int N, int maxlev){

  unsigned long int mcode = 0;


  // count the number of leafs
  int counter = 0;
  int clevel = 0; 
  for(int i=0; i<N; i++){
    if(level_record[i]>0){
      mcode = morton_codes[i] >> (3*(maxlev - level_record[i]));
      clevel = level_record[i];
      counter++;
    }
    else{

      counter += (mcode == morton_codes[i] >> (3*(maxlev-clevel))); 

    }
  }

  return(counter == N);

}

