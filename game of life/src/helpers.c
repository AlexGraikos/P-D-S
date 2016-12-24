#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N) {
  i += a;
  while (i < 0) i += N;
  while (i >= N) i -= N;
  return i;
}

/* add to a height index, wrapping around */

int yadd (int i, int a, int N) {
  i += a;
  while (i < 0) i += N;
  while (i >= N) i -= N;
  return i;
}


/* return the number of on cells adjacent to the i,j cell 
 * If we are trying to wrap above or below the matrix use
 * row_above or row_below for possible neighbors
 */

int adjacent_to (int *board, int i, int j, int M, int N, int *row_above, int *row_below) {
  int   k, l, count;

  count = 0;

  /* go around the cell */

  for (k=-1; k<=1; k++)
    for (l=-1; l<=1; l++)

      /* only count if at least one of k,l isn't zero */
      if (k || l) {
              
        // If we are reffering to the row below use the row_below buffer
        if (i + k == M) {
          if (row_below[yadd(j,l,N)]) count++;
        }
        // If we are reffering to the row above use the row_above buffer
        else if (i + k < 0) {
          if (row_above[yadd(j,l,N)]) count++;
        
	} else {
	  if (Board(xadd(i,k,M),yadd(j,l,N))) count++;
        }
     }

  return count;
}
