#obj = obj/forwardPropagation.o obj/neuralNet.o obj/helpers.o obj/deltaL.o obj/deltal.o obj/weightUpdate.o

src = $(wildcard src/*.cu)
obj = $(patsubst src/%.cu,obj/%.o,$(src))
inc = -I inc/
CC = nvcc $(inc)

all: $(obj) 
	$(CC) $(obj) -o bin/backPropagation -lpthread 

obj/forwardPropagation.o: src/forwardPropagation.cu
	$(CC) -c $< -o $@

obj/neuralNet.o: src/neuralNet.cu
	$(CC) -c $< -o $@ 

obj/helpers.o: src/helpers.cu
	$(CC) -c $< -o $@

obj/deltaL.o: src/deltaL.cu
	$(CC) -c $< -o $@

obj/deltal.o: src/deltal.cu
	$(CC) -c $< -o $@

obj/weightUpdate.o: src/weightUpdate.cu
	$(CC) -c $< -o $@

obj/biasUpdate.o: src/biasUpdate.cu
	$(CC) -c $< -o $@

obj/mse.o: src/mse.cu
	$(CC) -c $< -o $@

clean: 
	rm obj/* bin/backPropagation
