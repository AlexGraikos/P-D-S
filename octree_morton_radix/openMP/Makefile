GCC=gcc -O4 -fopenmp -std=gnu99

VERSION=0.04

all:
	$(GCC) test_octree.c datasets.c general_functions.c hash_codes.c morton_encoding.c radix_sort.c data_rearrangement.c verification.c -lm -o test_octree

clean:
	rm -f *~ *.o *.bin test_octree 

release:
	tar -cvf octree-code-$(VERSION).tar *.c *.h *m *sh Makefile
