#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include <game-of-life.h>

void play (int *board, int *newboard, int M, int N, int rank, int num_of_tasks) {
  /*
    (copied this from some web page, hence the English spellings...)

    1.STASIS : If, for a given cell, the number of on neighbours is 
    exactly two, the cell maintains its status quo into the next 
    generation. If the cell is on, it stays on, if it is off, it stays off.

    2.GROWTH : If the number of on neighbours is exactly three, the cell 
    will be on in the next generation. This is regardless of the cell's
    current state.

    3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will 
    be off in the next generation.
  */
  int i, j;

  /* for each cell, apply the rules of Life */
  
  /* We can calculate the next state of each cell
   * independently
   * We break down the matrix in rows since that is how the memory
   * is allocated -> gain in less cache block movements
   *
   * We send the first row to rank-1 task and the last row
   * to rank + 1 task
   * Ask for rows from the tasks
   * Compute every row inbetween
   * Finally wait to receive and compute last 2 rows
   * Sync tasks so that they all end generation
  */
  
  MPI_Request req[4];
  MPI_Status stats[2];
  int *row_above = (int*) malloc(N * sizeof(int));
  int *row_below = (int*) malloc(N * sizeof(int));
  int tag_above = 0;
  int tag_below = 1;

  int prev = rank - 1 < 0 ? num_of_tasks - 1 : rank - 1;
  int next = rank + 1 == num_of_tasks ? 0 : rank + 1;
  
  if (num_of_tasks > 1) {
    // Send first and last rows
    MPI_Isend(&Board(0,0), N, MPI_INT, prev, tag_below, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(&Board(M-1,0), N, MPI_INT, next, tag_above, MPI_COMM_WORLD, &req[1]);

    // Receive row_above and row_below
    MPI_Irecv(row_above, N, MPI_INT, prev, tag_above, MPI_COMM_WORLD, &req[2]);
    MPI_Irecv(row_below, N, MPI_INT, next, tag_below, MPI_COMM_WORLD, &req[3]);
  
    // Calculate next state for the cells independent of other tasks

    #pragma omp parallel for private(j)
    for (i=1; i<M-1; i++)
      for (j=0; j<N; j++) {
        
        int a = adjacent_to (board, i, j, M, N, row_above, row_below);
        if (a == 2) NewBoard(i,j) = Board(i,j);
        if (a == 3) NewBoard(i,j) = 1;
        if (a < 2) NewBoard(i,j) = 0;
        if (a > 3) NewBoard(i,j) = 0;
    }

    // Wait until row_above and row_below are received 
    MPI_Waitall(2, &req[2], stats);
    
    // Compute first and last row
    int k;
    for (k=0; k<2; k++) {
      if (k==0) i=0;
      if (k==1) i=M-1;
      
      #pragma omp parallel for
      for (j=0; j<N; j++) {
      
          int a = adjacent_to (board, i, j, M, N, row_above, row_below);
          if (a == 2) NewBoard(i,j) = Board(i,j);
          if (a == 3) NewBoard(i,j) = 1;
          if (a < 2) NewBoard(i,j) = 0;
          if (a > 3) NewBoard(i,j) = 0;
      }
    }

    // Wait until row0 and rowM-1 are sent 
    // If we modify them in the next generation
    // our neighbors will compute a wrong next state
    MPI_Waitall(2, &req[0], stats);

  } else {
    //printf("Only 1 task will execute\n");

    row_above = &Board(M-1,0);
    row_below = &Board(0,0);

    #pragma omp parallel for private(j)
    for (i=0; i<M; i++)
      for (j=0; j<N; j++) {
        
        int a = adjacent_to (board, i, j, M, N, row_above, row_below);
        if (a == 2) NewBoard(i,j) = Board(i,j);
        if (a == 3) NewBoard(i,j) = 1;
        if (a < 2) NewBoard(i,j) = 0;
        if (a > 3) NewBoard(i,j) = 0;
     } 
  }


}
