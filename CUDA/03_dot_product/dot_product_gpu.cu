#include<stdio.h>
#include<stdlib.h>

// Each thread performs one pair-wise addition
__global__ void dotProdKernel(float* d_A, float* d_B, float* d_C, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		d_C[i] = d_A[i] * d_B[i];
}

void dotProduct(float* h_A, float* h_B, float* h_C, int n)
{
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;    // pointers to device copies of A, B, C

	// Allocate device memory space for device copies of A, B, C
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);

	// Copy vectors A and B from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Launch the kernel function to have the device to perform the actual vector addition
	int threads_per_block = 32;
	int no_of_blocks = ceil(n / (float) threads_per_block);	
	dotProdKernel<<<no_of_blocks, threads_per_block>>>(d_A, d_B, d_C, n);

	// Copy result vector C from the device memory to host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Free device memory for A, B, C
	cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_C);
	
	for (int i = 1; i < n; i++)
		h_C[0] += h_C[i];
}

int main(int argc, char *argv[]) {
	float *h_A, *h_B, *h_C;
	int n = 64;
	int size = n * sizeof(float);

	// Memory allocation for h_A, h_B and h_C
	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);
	h_C = (float *) malloc(size);

	// Setup input values into each of n elements of h_A and h_B
	printf("A = [");
	for (int i = 0; i < n; i++) {
		h_A[i] = (float) i;
		printf(" %.1f ", h_A[i]);
	}
	printf("]\nB = [");
	for (int i = 0; i < n; i++) {
		h_B[i] = (float) i;
		printf(" %.1f ", h_B[i]);
	}
	printf("]\n");

	dotProduct(h_A, h_B, h_C, n);    // Call the host function for dot product

	// Output the results
	printf("Dot Product = %.1f\n", h_C[0]);

	// Free host memory for A and B
	free(h_A);
	free(h_B);
	return 0;
}
