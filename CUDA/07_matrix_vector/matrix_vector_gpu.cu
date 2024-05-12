#include<stdio.h>
#include<stdlib.h>

#define NUM_COL 2
#define NUM_ROW 3

struct Dim2 {
	unsigned char nc;
	unsigned char nr;
};

__global__ void matVecMulKernel(float* d_A, float* d_B, float* d_C, struct Dim2 dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < dim.nr) {
		int sum = 0.0;
		for(int col = 0; col < dim.nc; col++) {
			sum += d_B[row * dim.nc + col] * d_C[col];
		}
		d_A[row] = sum;
	}
}

void matVecMul(float* h_A, float* h_B, float* h_C, struct Dim2 dim)
{
	float *d_A, *d_B, *d_C;

	int size_A = NUM_ROW * sizeof(float);
	int size_B = NUM_ROW * NUM_COL * sizeof(float);
	int size_C = NUM_COL * sizeof(float);

	cudaMalloc((void **) &d_A, size_A);
	cudaMalloc((void **) &d_B, size_B);
	cudaMalloc((void **) &d_C, size_C);

	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

	int threads_per_block = 32;
	int no_of_blocks = ceil(dim.nr / (float) threads_per_block);	
	matVecMulKernel<<<no_of_blocks, threads_per_block>>>(d_A, d_B, d_C, dim);

	cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main(int argc, char *argv[])
{
	float *h_A = (float *) malloc(NUM_ROW * sizeof(float));
	float *h_B = (float *) malloc(NUM_ROW * NUM_COL * sizeof(float));
	float *h_C = (float *) malloc(NUM_COL * sizeof(float));

	printf("B = \n");
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			int offset = row * NUM_COL + col;
			h_B[offset] = (float) (rand() % 10);
			printf("%.1f\t", h_B[offset]);
		}
		printf("\n");
	}
	printf("\nC = \n");
	for(int row = 0; row < NUM_COL; row++) {
		h_C[row] = (float) (rand() % 10);
		printf("%.1f\n", h_C[row]);
	}

	struct Dim2 dim;
	dim.nc = NUM_COL;
	dim.nr = NUM_ROW;
	matVecMul(h_A, h_B, h_C, dim);

	printf("\nA = \n");
	for(int row = 0; row < NUM_ROW; row++) {
		printf("%.1f\n", h_A[row]);
	}

	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}
