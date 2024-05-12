#include<stdio.h>
#include<stdlib.h>

// Compute vector sum h_C = h_A + h_B with the host function
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
	for (int i = 0; i < n; i++)
		h_C[i] = h_A[i] + h_B[i];
}

int main(int argc, char *argv[])
{
	float *h_A, *h_B, *h_C;
	int n = 64;
	int size = n * sizeof(float);

	// Memory allocation for h_A, h_B, and h_C
	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);
	h_C = (float *) malloc(size);

	// Setup input values into each of n elements of h_A and h_B
	for (int i = 0; i < n; i++)
		h_A[i] = (float) i;
	for (int i = 0; i < n; i++)
		h_B[i] = (float) i;

	vecAdd(h_A, h_B, h_C, n);    // Call the host function for vector addition

	// Output the results
	for(int i = 0; i < n; i++)
		printf("%.1f + %.1f = %.1f\n", h_A[i] , h_B[i], h_C[i]);

	// Free host memory for A, B, C
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}
