all: cuda cpu

cuda: src/main.cu
	nvcc --std=c++11 src/main.cu -o main_cuda

cpu: src/main.cpp
	g++ --std=c++11 src/main.cpp -o main_cpu