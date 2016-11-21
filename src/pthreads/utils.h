
/* datasets.c */
void create_dataset(float *X, int N, int dist);


/* general function */
float find_max(float *max_out, float* X, int N);
float find_min(float *min_out, float *X, int N);


/* hash_codes.c */
void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, 
			float *max, int nThreads);
float max_range(float *x);

/* morton_encoding.c */
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level, int nThreads);
unsigned int compute_code(float x, float low, float step);

/* radix_sort.c */
/*
void truncated_radix_sort(unsigned long int *morton_codes, 
			  unsigned long int *sorted_morton_codes, 
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N, 
			  int population_threshold,
			  int sft, int lv);
*/

void *truncated_radix_sort(void *arguments);

/* data rearrangment */
void data_rearrangement(float *Y, float *X, unsigned int *permutation_vector, int N, int nThreads);


/* verification */
int check_index(unsigned int *index, int N);
int check_codes(float *X, unsigned long int *morton_codes, 
		unsigned int *level_record, int N, int maxlev);
 
