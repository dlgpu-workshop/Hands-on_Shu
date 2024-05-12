all: matrix_multiplication_cpu matrix_multiplication_gpu matrix_multiplication_gpu-fw

matrix_multiplication_cpu: matrix_multiplication_cpu.cu
	nvcc -arch compute_50 -o matrix_multiplication_cpu matrix_multiplication_cpu.cu

matrix_multiplication_gpu: matrix_multiplication_gpu.cu
	nvcc -arch compute_50 -o matrix_multiplication_gpu matrix_multiplication_gpu.cu

matrix_multiplication_gpu-fw: matrix_multiplication_gpu-fw.cu
	nvcc -arch compute_50 -o matrix_multiplication_gpu-fw matrix_multiplication_gpu-fw.cu

clean:
	rm -f matrix_multiplication_cpu matrix_multiplication_gpu matrix_multiplication_gpu-fw

