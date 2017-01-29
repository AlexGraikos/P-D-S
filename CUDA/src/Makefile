SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC= nvcc -arch=$(CC)
FILE= cuda/non_local_means_kernel
ARGS= -ptx

all: sample

sample:
	$(NVCC) $(ARGS) $(FILE).cu -o $(FILE).ptx

clean:
	rm -rf *~ *.ptx
