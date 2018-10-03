#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "cilk/cilk.h"
#include "cilk/cilk_api.h"

#define DIM 3

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);
}


void callCompute(unsigned int *codes, float *X, float *low, float step, int i){
  int j;
  for(j=0; j<DIM; j++){
    codes[i*DIM + j] = compute_code(X[i*DIM +j], low[j], step);
  }
}


/* Function that does the quantization */

void quantize(unsigned int *codes, float *X, float *low, float step, int N){
  
  // === Parallelize the for loop since no dependencies ===
  
  cilk_for(int i=0; i<N; i++){
    callCompute(codes, X, low, step, i);

    // Errors when using the for loop
    /*
    for(int j=0; j<DIM; j++){
      codes[i*DIM + j] = compute_code(X[i*DIM + j], low[j], step); 
    }
    */
  }
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
			int nbins, float *min, 
			float *max){
  
  float range[DIM];
  float qstep;

  for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add somthing small to avoid having points exactly at the boundaries 
  }

  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization
}



