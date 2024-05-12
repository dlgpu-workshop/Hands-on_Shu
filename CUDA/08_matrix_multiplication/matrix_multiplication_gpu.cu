#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "myHead.h"

// Comment the following statement to hide the printed elements in all matrices
//#define OUTPUT 1
#ifdef OUTPUT
#define FUNC_PRINT(fmt, args...) printf(fmt, ## args)
#else
#define FUNC_PRINT(...) 
#endif

#define TILE_WIDTH 32

struct Dim {
	unsigned int nr;	// the number of rows in M
	unsigned int ne;	// the number of columns in M and the number of rows in N
	unsigned int nc;	// the number of columns in N
};

// Figure 4.3: A simple matrix multiplication kernel using one thread to compute one P element.
// Each thread produces one output matrix element.
__global__ void matMulKernel0(float* d_P, float* d_M, float* d_N, Dim dim)
{
	// Calculate the column index of the element in d_P
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the row index of the element in d_P
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < dim.nr && col < dim.nc) {
		float dotProc = 0.0;
		for (int k = 0; k < dim.ne; k++) {
			dotProc += d_M[row * dim.ne + k] * d_N[k * dim.nc + col];
		}
		d_P[row * dim.nc + col] = dotProc;
	}
}

/*
 * Figures 4.16 and 4.20: A tiled matrix multiplication kernel 
 * with square tiles
 * with boundary condition checks 
 * using shared memory with a fixed size.
 */
__global__ void matMulKernel1(float* d_P, float* d_M, float* d_N, Dim dim)
{
	__shared__ float ds_M[TILE_WIDTH * TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH * TILE_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int col = blockIdx.x * TILE_WIDTH + tx;
	int row = blockIdx.y * TILE_WIDTH + ty;

	float dotProc = 0.0;
	int num_phase = ceil(dim.ne / (float) TILE_WIDTH);

	// Loop over the d_M and d_N tiles required to compute d_P element
	for (int phase = 0; phase < num_phase; phase++) {
		// Collaborative loading of d_M and d_N tiles into shared memory
		if (row < dim.nr && phase * TILE_WIDTH + tx < dim.ne)
			ds_M[ty * TILE_WIDTH + tx] = d_M[row * dim.ne + phase * TILE_WIDTH + tx];
		if (col < dim.nc && phase * TILE_WIDTH + ty < dim.ne)
			ds_N[ty * TILE_WIDTH + tx] = d_N[(phase * TILE_WIDTH + ty) * dim.nc + col];

		__syncthreads();

		int phase_length = TILE_WIDTH;
		if (phase == num_phase - 1)
			phase_length = dim.ne % TILE_WIDTH;
		for (int k = 0; k < phase_length; k++) {
			dotProc += ds_M[ty * TILE_WIDTH + k] * ds_N[k * TILE_WIDTH + tx];
		}

		__syncthreads();
	}
	if (col < dim.nc && row < dim.nr)
		d_P[row * dim.nc + col] = dotProc;
}

/*
 * Figures 4.16 and 4.20: A tiled matrix multiplication kernel 
 * with square tiles
 * with boundary condition checks 
 * using shared memory with an adjustable size.
 */
__global__ void matMulKernel2(float* d_P, float* d_M, float* d_N, Dim dim, unsigned int tile_width)
{
	extern __shared__ float ds[];
	float *ds_M = ds;
	float *ds_N = (float *) &ds_M[tile_width * tile_width];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int col = blockIdx.x * tile_width + tx;
	int row = blockIdx.y * tile_width + ty;

	float dotProc = 0.0;
	int num_phase = ceil(dim.ne / (float) tile_width);

	// Loop over the d_M and d_N tiles required to compute d_P element
	for (int phase = 0; phase < num_phase; phase++) {
		// Collaborative loading of d_M and d_N tiles into shared memory
		if (row < dim.nr && phase * tile_width + tx < dim.ne)
			ds_M[ty * tile_width + tx] = d_M[row * dim.ne + phase * tile_width + tx];
		if (col < dim.nc && phase * tile_width + ty < dim.ne)
			ds_N[ty * tile_width + tx] = d_N[(phase * tile_width + ty) * dim.nc + col];

		__syncthreads();

		int phase_length = tile_width;
		if (phase == num_phase - 1)
			phase_length = dim.ne % tile_width;
		for (int k = 0; k < phase_length; k++) {
			dotProc += ds_M[ty * tile_width + k] * ds_N[k * tile_width + tx];
		}

		__syncthreads();
	}
	if (col < dim.nc && row < dim.nr)
		d_P[row * dim.nc + col] = dotProc;
}

/*
 * Exercise 5.10/Figure 5.17: A tiled matrix multiplication kernel 
 * with rectangular tiles (combining two adjacent horizontal blocks to compute adjacent horizontal tiles)
 * with boundary condition checks 
 * using shared memory with an adjustable size.
 */
__global__ void matMulKernel3(float* d_P, float* d_M, float* d_N, Dim dim, unsigned int tile_width)
{
	extern __shared__ float ds[];
	float *ds_M = ds;
	float *ds_N1 = (float *) &ds_M[tile_width * tile_width];
	float *ds_N2 = (float *) &ds_N1[tile_width * tile_width];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int col = blockIdx.x * tile_width * 2 + tx;
	int row = blockIdx.y * tile_width + ty;

	float dotProc1 = 0.0, dotProc2 = 0.0;
	int num_phase = ceil(dim.ne / (float) tile_width);

	// Loop over the d_M and d_N tiles required to compute d_P element
	for (int phase = 0; phase < num_phase; phase++) {
		// Collaborative loading of d_M and d_N tiles into shared memory
		if (row < dim.nr && phase * tile_width + tx < dim.ne)
			ds_M[ty * tile_width + tx] = d_M[row * dim.ne + phase * tile_width + tx];
		if (col < dim.nc && phase * tile_width + ty < dim.ne)
			ds_N1[ty * tile_width + tx] = d_N[(phase * tile_width + ty) * dim.nc + col];
		if (col + tile_width < dim.nc && phase * tile_width + ty < dim.ne)
			ds_N2[ty * tile_width + tx] = d_N[(phase * tile_width + ty) * dim.nc + col + tile_width];

		__syncthreads();

		int phase_length = tile_width;
		if (phase == num_phase - 1)
			phase_length = dim.ne % tile_width;
		for (int k = 0; k < phase_length; k++) {
			dotProc1 += ds_M[ty * tile_width + k] * ds_N1[k * tile_width + tx];
			dotProc2 += ds_M[ty * tile_width + k] * ds_N2[k * tile_width + tx];
		}

		__syncthreads();
	}
	if (col < dim.nc && row < dim.nr)
		d_P[row * dim.nc + col] = dotProc1;
	if (col + tile_width < dim.nc && row < dim.nr)
		d_P[row * dim.nc + col + tile_width] = dotProc2;
}

/*
 * A host stub function:  
 * allocating memory for the input and output matrices, 
 * transferring input data to device, 
 * launch the kernel, 
 * transferring the output data to host, 
 * and freeing the device memory for the input and output data.
 */
void matMul(float* h_P, float* h_M, float* h_N, Dim dim, int opt)
{
	float *d_M, *d_N, *d_P;    // pointers to device copies of M, N, P
	int size_M = dim.nr * dim.ne * sizeof(float);
	int size_N = dim.ne * dim.nc * sizeof(float);
	int size_P = dim.nr * dim.nc * sizeof(float);

	// Allocate device memory space for device copies of M, N, P
	cudaMalloc((void **) &d_M, size_M);
	cudaMalloc((void **) &d_N, size_N);
	cudaMalloc((void **) &d_P, size_P);

	// Copy matrices M and N from host memory to device memory
	cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);

	// Launch the kernel function to have the device to perform the actual matrix addition
	dim3 dimGrid(ceil(dim.nc / (float) TILE_WIDTH), ceil(dim.nr / (float) TILE_WIDTH), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	cudaDeviceProp prop;
	unsigned int tile_width, sharedMemSize;
	switch(opt) {
		case 0:
			matMulKernel0<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, dim);
			break;
		case 1:
			matMulKernel1<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, dim);
			break;
		case 2:
			cudaGetDeviceProperties(&prop, 0);
			tile_width = TILE_WIDTH;
			sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
			if (sharedMemSize > prop.sharedMemPerBlock) {
				tile_width = floor(sqrt(prop.sharedMemPerBlock / 2.0 / sizeof(float)));
				sharedMemSize = 2 * tile_width * tile_width * sizeof(float);
			}
			dimBlock.x = dimBlock.y = tile_width;
			dimGrid.x = ceil(dim.nc / (float) tile_width);
			dimGrid.y = ceil(dim.nr / (float) tile_width);
			matMulKernel2<<<dimGrid, dimBlock, sharedMemSize>>>(d_P, d_M, d_N, dim, tile_width);
			break;
		case 3:
		default:
			cudaGetDeviceProperties(&prop, 0);
			tile_width = TILE_WIDTH;
			sharedMemSize = 3 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
			if (sharedMemSize > prop.sharedMemPerBlock) {
				tile_width = floor(sqrt(prop.sharedMemPerBlock / 3.0 / sizeof(float)));
				sharedMemSize = 3 * tile_width * tile_width * sizeof(float);
			}
			dimBlock.x = dimBlock.y = tile_width;
			dimGrid.x = ceil(dim.nc / (2.0 * tile_width));
			dimGrid.y = ceil(dim.nr / (float) tile_width);
			matMulKernel3<<<dimGrid, dimBlock, sharedMemSize>>>(d_P, d_M, d_N, dim, tile_width);
	}

	// Copy result matrix P from the device memory to host memory
	cudaMemcpy(h_P, d_P, size_P, cudaMemcpyDeviceToHost);

	// Free device memory for M, N, P
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

int main(int argc, char *argv[])
{
	int opt = 0;
	if (argc > 1) {
		opt = atoi(argv[1]);
		printf("Optimization level = %d\n", opt);
	}

	struct Dim dim;
	printf("Enter the number of rows in M: ");
	scanf("%d", &dim.nr);
	printf("Enter the number of columns in M and the number of rows in N: ");
	scanf("%d", &dim.ne);
	printf("Enter the number of columns in N: ");
	scanf("%d", &dim.nc);

	int size_M = dim.nr * dim.ne * sizeof(float);
	int size_N = dim.ne * dim.nc * sizeof(float);
	int size_P = dim.nr * dim.nc * sizeof(float);

	// Memory allocation for h_M, h_N, and h_P
	float *h_M = (float *) malloc(size_M);
	float *h_N = (float *) malloc(size_N);
	float *h_P = (float *) malloc(size_P);

	// Setup input values into each of n elements of h_M and h_N
	FUNC_PRINT("Matrix M:\n");
	for(int row = 0; row < dim.nr; row++) {
		for(int col = 0; col < dim.ne; col++) {
			int offset = row * dim.ne + col;
			h_M[offset] = (float) offset;
			FUNC_PRINT("%.1f\t", h_M[offset]);
		}
		FUNC_PRINT("\n");
	}
	FUNC_PRINT("*\nMatrix N:\n");
	for(int row = 0; row < dim.ne; row++) {
		for(int col = 0; col < dim.nc; col++) {
			int offset = row * dim.nc + col;
			h_N[offset] = (float) offset;
			FUNC_PRINT("%.1f\t", h_N[offset]);
		}
		FUNC_PRINT("\n");
	}

	// Call the host function for matrix multiplication
	struct timespec startTime;
	clock_gettime(CLOCK_REALTIME, &startTime);
	matMul(h_P, h_M, h_N, dim, opt);
	struct timespec endTime;
	clock_gettime(CLOCK_REALTIME, &endTime);

	// Output the results
	FUNC_PRINT("=\nMatrix P:\n");
	for(int row = 0; row < dim.nr; row++) {
		for(int col = 0; col < dim.nc; col++)
			FUNC_PRINT("%.1f\t", h_P[row * dim.nc + col]);
		FUNC_PRINT("\n");
	}

	struct timespec diffTime = getDiffTime(&startTime, &endTime);
	printf("Execution time: %ld s and %ld us.\n", diffTime.tv_sec, (long) round(diffTime.tv_nsec / 1000.0));

	// Free host memory for M, N, P
	free(h_M);
	free(h_N);
	free(h_P);
	return 0;
}
