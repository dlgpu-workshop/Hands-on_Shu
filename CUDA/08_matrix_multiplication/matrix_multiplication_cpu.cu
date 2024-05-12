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

struct Dim {
	unsigned int nr;	// the number of rows in M
	unsigned int ne;	// the number of columns in M and the number of rows in N
	unsigned int nc;	// the number of columns in N
};

// Compute matrix product h_P = h_M * h_N with the host function
void matMul(float* h_P, float* h_M, float* h_N, Dim dim)
{
	for(int row = 0; row < dim.nr; row++) {
		for(int col = 0; col < dim.nc; col++) {
			int offset_P = row * dim.nc + col;
			h_P[offset_P] = 0;
			for(int k = 0; k < dim.ne; k++) {
				int offset_M = row * dim.ne + k;
				int offset_N = k * dim.nc + col;
				h_P[offset_P] += h_M[offset_M] * h_N[offset_N];
			}
		}
	}
}

int main(int argc, char *argv[])
{
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
	matMul(h_P, h_M, h_N, dim);
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
