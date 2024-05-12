#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define NUM_COL 2
#define NUM_ROW 3

struct Dim2 {
	unsigned char nc;
	unsigned char nr;
};

void matVecMul(float* h_A, float* h_B, float* h_C, struct Dim2 dim)
{
	for(int row = 0; row < dim.nr; row++) {
		h_A[row] = 0.0;
		for(int col = 0; col < dim.nc; col++) {
			h_A[row] += h_B[row * dim.nc + col] * h_C[col];
		}
	}
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
