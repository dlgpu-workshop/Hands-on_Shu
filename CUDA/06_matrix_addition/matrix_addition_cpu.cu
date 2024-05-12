#include<stdio.h>
#include<stdlib.h>

#define NUM_COL 10
#define NUM_ROW 17

struct Dim2 {
	unsigned char nc;
	unsigned char nr;
};

// Compute matrix sum h_C = h_A + h_B with the host function
void matAdd(float* h_C, float* h_A, float* h_B, struct Dim2 dim)
{
	for(int row = 0; row < dim.nr; row++) {
		int offset = row * dim.nc;
		for(int col = 0; col < dim.nc; col++) {
			h_C[offset] = h_A[offset] + h_B[offset];
			offset++;
		}
	}
}

int main(int argc, char *argv[])
{
	int size = NUM_ROW * NUM_COL * sizeof(float);

	// Memory allocation for h_A, h_B, and h_C
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);

	// Setup input values into each of n elements of h_A and h_B
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			int offset = row * NUM_COL + col;
			h_A[offset] = (float) offset;
			h_B[offset] = (float) offset;
		}
	}

	// Call the host function for matrix addition
	struct Dim2 dim;
	dim.nc = NUM_COL;
	dim.nr = NUM_ROW;
	matAdd(h_C, h_A, h_B, dim);

	// Output the results
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			int offset = row * NUM_COL + col;
			printf("%.1f + %.1f = %.1f\n", h_A[offset] , h_B[offset], h_C[offset]);
		}
	}

	// Free host memory for A, B, C
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}
