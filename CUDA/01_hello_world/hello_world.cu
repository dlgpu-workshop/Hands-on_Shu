#include<stdio.h>
#include<stdlib.h> 

__global__ void print_from_gpu(void)
{
	printf("Hello World! from thread [%d,%d] from device\n", 
			blockIdx.x, threadIdx.x); 
}

int main(int argc, char *argv[])
{ 
	printf("Hello World from host!\n"); 
	print_from_gpu<<<2, 3>>>();
	cudaDeviceSynchronize();
	return 0;
}

