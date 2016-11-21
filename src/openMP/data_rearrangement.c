#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "omp.h"

#define DIM 3


void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){
  int i;
  int nThreads = omp_get_max_threads();
  unsigned long chunk = N / nThreads;

  #pragma omp parallel for default(shared) schedule(dynamic, chunk) private(i)
    for(i=0; i<N; i++){
      memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
    }

}
