#include<stdio.h>
#include<stdlib.h>

void dotProduct(float* h_A, float* h_B, float* h_C, int n)
{
	*h_C = 0.0;
	for (int i = 0; i < n; i++)
		*h_C += h_A[i] * h_B[i];
}

int main(int argc, char *argv[]) {
	float *h_A, *h_B, h_C;
	int n = 64;
	int size = n * sizeof(float);

	// Memory allocation for h_A and h_B
	h_A = (float *) malloc(size);
	h_B = (float *) malloc(size);

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

	dotProduct(h_A, h_B, &h_C, n);    // Call the host function for dot product

	// Output the results
	printf("Dot Product = %.1f\n", h_C);

	// Free host memory for A and B
	free(h_A);
	free(h_B);
	return 0;
}
