/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 * Added M argument from console that specifies number of rows
 * of the genarated matrix
 * Added Thrd argument -> number of threads
 */


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include <game-of-life.h>

int main (int argc, char *argv[]) {
  int   *board, *newboard, i;
  
  int num_of_tasks, rank, len;
  char hostname[MPI_MAX_PROCESSOR_NAME];

  struct timeval start_time, end_time;

  if (argc != 7) { // Check if the command line arguments are correct 
    printf("Usage: %s M N thres disp\n"
	   "where\n"
	   "  M     : number of rows\n"
	   "  N     : number of columns\n"
	   "  thres : probability of alive cell\n"
           "  t     : number of generations\n"
	   "  Thrd  : number of threads\n"
	   "  disp  : {1: display output, 0: hide output}\n"
           , argv[0]);
    return (1);
  }
  
  // Initialize MPI 
  MPI_Init(&argc, &argv);

  // Get number of tasks to execute job
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_tasks);
  
  // Get rank ID
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Get host name and print details
  MPI_Get_processor_name(hostname, &len);

  printf("Total tasks: %d , Task ID: %d , Host Name: %s \n", num_of_tasks,
  	rank, hostname);

  // Input command line arguments
  int M = atoi(argv[1]);        // Array size
  int N = atoi(argv[2]);
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations 
  int nThreads = atoi(argv[5]); // Number of threads
  int disp = atoi(argv[6]);     // Display output?
  printf("Size %dx%d with probability: %0.1f%%, number of threads: %d, size assigned to this task: %dx%d \n", M, N, thres*100, nThreads, M / num_of_tasks, N);
 
  // Setup OpenMP threads for use
  omp_set_num_threads(nThreads);
    
  // Open a file to write output
  /* 
  char name[20];
  sprintf(name, "Task%d", rank);
  FILE *file = fopen(name, "w+");
  fprintf(file, "Task %d starting\n", rank);
  */

  // Synchronize tasks and start clock
  MPI_Barrier(MPI_COMM_WORLD);

  // Start clock
  gettimeofday(&start_time, NULL);
  
  // Split the table by rows for each task
  M = M / num_of_tasks;
 
  board = NULL;
  newboard = NULL;
  
  board = (int *)malloc(M*N*sizeof(int));

  if (board == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }

  /* second pointer for updated result */
  newboard = (int *)malloc(M*N*sizeof(int));

  if (newboard == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }
  
  // Initialization is not needed since generation is enough

  generate_table (board, M, N, thres, rank);
  printf("Board generated\n");

  /* play game of life t times */
  
  for (i=0; i<t; i++) {
    // Ignore file output
    //if (disp) display_table (board, M, N, file);
    
    play (board, newboard, M, N, rank, num_of_tasks);    
    
    // Swap new board and board
    int *temp = newboard;
    newboard = board;
    board = temp;
  
  }
   
  // Sync all tasks
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Close file
  //fclose(file);

  // Finalize MPI
  MPI_Finalize();

  // End clock and calculate time needed
  gettimeofday(&end_time, NULL);

  double time_elapsed = (double) ( (end_time.tv_usec - start_time.tv_usec) / 1000000.0
                                  + (end_time.tv_sec - start_time.tv_sec));

  printf("Game finished in task %d after %d generations. Time needed %lf \n", rank, t, time_elapsed);

  return 0;
}
