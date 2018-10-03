#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* print the life board */

/*
 * Printing in reverse order so that it is printed as
 * a MxN matrix (downloaded version printed a NxM matrix)
 * Changed code to print into files for each individual task
*/

void print (int *board, int M, int N, FILE *file) {
  int   i, j;

  /* for each row */
  for (i=0; i<M; i++) {

    /* print each column position... */
    for (j=0; j<N; j++) {
      //printf ("%c", Board(i,j) ? 'x' : ' ');
      fprintf (file, "%c", Board(i,j) ? 'x' : ' ');
    }

    /* followed by a carriage return */
    fprintf(file, "\n");
  }
}



/* display the table with delay and clear console */

void display_table(int *board, int M, int N, FILE *file) {
  print (board, M, N, file);
  usleep(100000);  
  
  // Print to distinguish generations in files created
  fprintf(file, "\n Next Generation \n");
}
