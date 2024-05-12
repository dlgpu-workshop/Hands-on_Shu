#include<stdio.h>
#include<stdlib.h>

#define NUM_COL 10
#define NUM_ROW 17

struct Dim2 {
	unsigned char nc;
	unsigned char nr;
};

// Each thread produces one output matrix element.
__global__ void matAddKernel0(float* d_C, float* d_A, float* d_B, struct Dim2 dim)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < dim.nc && row < dim.nr) {
		int offset = row * dim.nc + col;
		d_C[offset] = d_A[offset] + d_B[offset];
	}
}

// Each thread produces one output matrix row.
__global__ void matAddKernel1(float* d_C, float* d_A, float* d_B, struct Dim2 dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < dim.nr) {
		int offset = row * dim.nc;
		for(int col = 0; col < dim.nc; col++) {
			d_C[offset] = d_A[offset] + d_B[offset];
			offset++;
		}
	}
}

// Each thread produces one output matrix column.
__global__ void matAddKernel2(float* d_C, float* d_A, float* d_B, struct Dim2 dim)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < dim.nc) {
		for (int row = 0; row < dim.nr; row++) {
			int offset = row * dim.nc + col;
			d_C[offset] = d_A[offset] + d_B[offset];
		}
	}
}

/*
 * A host stub function:  
 * allocating memory for the input and output matrices, 
 * transferring input data to device, 
 * launch the kernel, 
 * transferring the output data to host, 
 * and freeing the device memory for the input and output data.
 */
void matAdd(float h_C[NUM_ROW][NUM_COL], float h_A[NUM_ROW][NUM_COL], float h_B[NUM_ROW][NUM_COL], char map)
{
	float *d_A, *d_B, *d_C;    // pointers to device copies of A, B, C
	struct Dim2 dim;
	dim.nc = NUM_COL;
	dim.nr = NUM_ROW;
	int size = dim.nr * dim.nc * sizeof(float);

	// Allocate device memory space for device copies of A, B, C
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);

	// Copy matrices A and B from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Launch the kernel function to have the device to perform the actual matrix addition
	int threads_per_block, no_of_blocks;
	switch(map) {
		case 'C':
			printf("Exercise 1.C.\n");
			threads_per_block = 32;
			no_of_blocks = ceil(dim.nr / (float) threads_per_block);	
			matAddKernel1<<<no_of_blocks, threads_per_block>>>(d_C, d_A, d_B, dim);
			break;
		case 'D':
			printf("Exercise 1.D.\n");
			threads_per_block = 16;
			no_of_blocks = ceil(dim.nc / (float) threads_per_block);	
			matAddKernel2<<<no_of_blocks, threads_per_block>>>(d_C, d_A, d_B, dim);
			break;
		case 'B':
		default:
			printf("Exercise 1.A-B by default.\n");
			dim3 dimGrid(ceil(dim.nc / (float) 8), ceil(dim.nr / (float) 8), 1);
			dim3 dimBlock(8, 8, 1);
			matAddKernel0<<<dimGrid,dimBlock>>>(d_C, d_A, d_B, dim);
	}

	// Copy result matrix C from the device memory to host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	// Free device memory for A, B, C
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main(int argc, char *argv[])
{
	printf("Enter a character to specify the element-to-thread mapping policy (B, C, or D): ");
	char map;
	scanf("%c", &map);

	// Memory allocation for h_A, h_B, and h_C
	float h_A[NUM_ROW][NUM_COL];
	float h_B[NUM_ROW][NUM_COL];
	float h_C[NUM_ROW][NUM_COL];

	// Setup input values into each of n elements of h_A and h_B
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			int offset = row * NUM_COL + col;
			h_A[row][col] = (float) offset;
			h_B[row][col] = (float) offset;
		}
	}

	// Call the host function for matrix addition
	matAdd(h_C, h_A, h_B, map);

	// Output the results
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			printf("%.1f + %.1f = %.1f\n", h_A[row][col] , h_B[row][col], h_C[row][col]);
		}
	}

	return 0;
}
