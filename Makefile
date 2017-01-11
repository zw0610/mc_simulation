CUDA_ROOT=/usr/local/cuda
NVCC=$(CUDA_ROOT)/bin/nvcc
NVCC_FLAGS=-ccbin g++ -m64 -std=c++11
NVCC_ARCH_FLAGS=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30
NVCC_IFLAGS=-I$(CUDA_ROOT)/include -I$(CUDA_ROOT)/samples/common/inc
LFLAGS=-lcurand

all: exec

mc_price.o: mc_price.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_IFLAGS) $(NVCC_ARCH_FLAGS) -c $< -o $@

exec: mc_price.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH_FLAGS) $< -o $@ $(LFLAGS)

clean:
	rm -f *.o

clobber: clean
	rm -f exec
