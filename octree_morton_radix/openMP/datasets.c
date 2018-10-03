/*

This file contains the different distributions used for testing the octree construction

author: Nikos Sismanis
date: Oct 2014

*/

#include "stdio.h" 
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "time.h"

#define DIM 3 // Dimension of the space 
#define PI 3.1415927 

// Write dataset to file dataset.bin
void write_dataset(float *X, int N) {
  FILE *filePtr;
  printf("Writing to file ... ");
  filePtr = fopen("dataset.bin","w");
  
  for (int i = 0; i < N; i++) {
    fprintf(filePtr, "%.8f\t%.8f\t%.8f\n", X[i*DIM], X[i*DIM + 1], X[i*DIM + 2]);
  }

  fclose(filePtr);
  printf("Done!\n");
}

void cube(float *X, int N){

  srand(time(NULL));
  for(int i=0; i<N; i++){
    for(int j=0; j<DIM; j++){
      X[i*DIM + j] = (float)rand() / (float) RAND_MAX;
    }
  }
  
}

void sphere(float *X, int N){

  srand(time(NULL));
  for(int i=0; i<N; i++){
    X[i*DIM] = (float)rand() / (float) RAND_MAX;
    X[i*DIM + 1] = ((float)rand() / (float) RAND_MAX) *
      sqrt(1 - pow(X[i*DIM], 2));
    X[i*DIM + 2] = sqrt(1 - pow(X[i*DIM], 2) - pow(X[i*DIM + 1], 2));
  }

}

/* Function that creates a dataset for testing 0 (uniform cude) 
1 (spherical octant) */
void create_dataset(float *X, int N, int dist){

  switch(dist){
  case 0:
    cube(X, N);
    break;
  case 1:
    sphere(X, N);
    break;
  default:
    sphere(X, N);
    break;
    
  break;
  }

  // output dataset as file dataset.bin
  // to load into MATLAB
  // 
  // write_dataset(X,N);
  
}
  
