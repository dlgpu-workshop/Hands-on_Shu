all: matrix_vector_cpu matrix_vector_gpu

matrix_vector_cpu: matrix_vector_cpu.c
	gcc -o matrix_vector_cpu matrix_vector_cpu.c

matrix_vector_gpu: matrix_vector_gpu.cu
	nvcc -arch compute_50 -o matrix_vector_gpu matrix_vector_gpu.cu

clean:
	rm -f matrix_vector_cpu matrix_vector_gpu
