#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 6

__global__ void blockTransposeKernel(float* A_elements, unsigned int A_width, unsigned int A_height)
{
	__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
	int baseIdx = blockIdx.x * blockDim.x + threadIdx.x;
	baseIdx += (blockIdx.y * blockDim.y + threadIdx.y) * A_width;
	blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
	__syncthreads();
	A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

void blockTranspose(float* h_A, unsigned int A_width, unsigned int A_height)
{
	int size = A_height * A_width * sizeof(float);
	float *d_A;
	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid(ceil(A_width / (float) dimBlock.x), ceil(A_height / (float) dimBlock.y));
	blockTransposeKernel<<<dimGrid, dimBlock>>>(d_A, A_width, A_height);

	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

int main(int argc, char *argv[])
{
	int A_width = 2 * BLOCK_WIDTH;
	int A_height = 3 * BLOCK_WIDTH;
	int size = A_height * A_width * sizeof(float);

	float *h_A = (float *) malloc(size);

	printf("Input Matrix A:\n");
	for(int row = 0; row < A_height; row++) {
		for(int col = 0; col < A_width; col++) {
			int offset = row * A_width + col;
			h_A[offset] = offset;
			printf("%.1f\t", h_A[offset]);
		}
		printf("\n");
	}

	blockTranspose(h_A, A_width, A_height);

	printf("\nOutput Matrix A:\n");
	for(int row = 0; row < A_height; row++) {
		for(int col = 0; col < A_width; col++) {
			printf("%.1f\t", h_A[row * A_width + col]);
		}
		printf("\n");
	}

	free(h_A);
	return 0;
}
