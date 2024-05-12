all: dot_product_cpu dot_product_gpu

dot_product_cpu: dot_product_cpu.c
	gcc -o dot_product_cpu dot_product_cpu.c

dot_product_gpu: dot_product_gpu.cu
	nvcc -arch compute_50 -o dot_product_gpu dot_product_gpu.cu

clean:
	rm -f dot_product_cpu dot_product_gpu
