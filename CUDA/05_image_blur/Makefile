all: image_blur_cpu image_blur_gpu

image_blur_cpu: image_blur_cpu.cu
	nvcc -arch compute_50 -o image_blur_cpu image_blur_cpu.cu

image_blur_gpu: image_blur_gpu.cu
	nvcc -arch compute_50 -o image_blur_gpu image_blur_gpu.cu

clean:
	rm -f image_blur_cpu image_blur_gpu
