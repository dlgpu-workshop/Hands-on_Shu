all: vector_addition_cpu vector_addition_gpu

vector_addition_cpu: vector_addition_cpu.c
	gcc -o vector_addition_cpu vector_addition_cpu.c

vector_addition_gpu: vector_addition_gpu.cu
	nvcc -arch compute_50 -o vector_addition_gpu vector_addition_gpu.cu

clean:
	rm -f vector_addition_cpu vector_addition_gpu
