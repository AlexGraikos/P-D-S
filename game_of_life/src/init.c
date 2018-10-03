#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include <game-of-life.h>

/* set everthing to zero */

void initialize_board (int *board, int M, int N) {
  int   i, j;

  for (i=0; i<M; i++)
    for (j=0; j<N; j++) 
      Board(i,j) = 0;
}

/* generate random table 
 * rand is NOT thread safe 
 * We are using drand48_r to generate our random numbers
 * in each cell
 * Seed for each task and thread differently to generate
 * different sequences of numbers
*/

void generate_table (int *board, int M, int N, float threshold, int rank) {

  // Buffer needed for seeding drand48_r in each thread
  struct drand48_data drand_buf;
  double random;
  long int seed;
  
  #pragma omp parallel private(drand_buf, random, seed) default(shared)
  {
    // Seed for individual task+thread
    seed = time(NULL) + rank + omp_get_thread_num();
    srand48_r(seed , &drand_buf); 

    int i, j;
    #pragma omp for private(j)
    for (i=0; i<M; i++) {
      for (j=0; j<N; j++) {
        
        // Get random number
        drand48_r(&drand_buf, &random);
	Board(i,j) = random  < threshold; 
      }
    }
  }
  
}

