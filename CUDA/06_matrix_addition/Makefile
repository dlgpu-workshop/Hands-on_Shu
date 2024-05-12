all: matrix_addition_cpu matrix_addition_cpu_static matrix_addition_gpu matrix_addition_gpu_s2d

matrix_addition_cpu: matrix_addition_cpu.cu
	nvcc -arch compute_50 -o matrix_addition_cpu matrix_addition_cpu.cu

matrix_addition_cpu_static: matrix_addition_cpu_static.cu
	nvcc -arch compute_50 -o matrix_addition_cpu_static matrix_addition_cpu_static.cu

matrix_addition_gpu: matrix_addition_gpu.cu
	nvcc -arch compute_50 -o matrix_addition_gpu matrix_addition_gpu.cu

matrix_addition_gpu_s2d: matrix_addition_gpu_s2d.cu
	nvcc -arch compute_50 -o matrix_addition_gpu_s2d matrix_addition_gpu_s2d.cu

clean:
	rm -f matrix_addition_cpu matrix_addition_cpu_static matrix_addition_gpu matrix_addition_gpu_s2d
