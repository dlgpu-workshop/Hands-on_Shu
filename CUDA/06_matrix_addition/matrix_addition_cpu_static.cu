#include<stdio.h>
#include<stdlib.h>

#define NUM_COL 10
#define NUM_ROW 17

// Compute matrix sum h_C = h_A + h_B with the host function
void matAdd(float h_C[NUM_ROW][NUM_COL], float h_A[NUM_ROW][NUM_COL], float h_B[NUM_ROW][NUM_COL])
{
	printf("in matAdd, h_A = %u, h_B = %u, h_C = %u\n", h_A, h_B, h_C);
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			h_C[row][col] = h_A[row][col] + h_B[row][col];
		}
	}
}

int main(int argc, char *argv[])
{
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
	printf("In main, h_A = %u, h_B = %u, h_C = %u\n", h_A, h_B, h_C);
	matAdd(h_C, h_A, h_B);

	// Output the results
	for(int row = 0; row < NUM_ROW; row++) {
		for(int col = 0; col < NUM_COL; col++) {
			printf("%.1f + %.1f = %.1f\n", h_A[row][col] , h_B[row][col], h_C[row][col]);
		}
	}

	return 0;
}
