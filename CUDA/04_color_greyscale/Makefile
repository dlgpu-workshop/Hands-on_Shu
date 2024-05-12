all: color_greyscale_cpu color_greyscale_gpu

color_greyscale_cpu: color_greyscale_cpu.cu
	nvcc -arch compute_50 -o color_greyscale_cpu color_greyscale_cpu.cu

color_greyscale_gpu: color_greyscale_gpu.cu
	nvcc -arch compute_50 -o color_greyscale_gpu color_greyscale_gpu.cu

clean:
	rm -f color_greyscale_cpu color_greyscale_gpu
