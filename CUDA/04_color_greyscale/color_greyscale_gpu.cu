#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 15
#define HEIGHT 10

// Three channels corresponding to RGB
struct Pixel {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

	__global__ 
void colorToGreyscaleKernel(unsigned char *d_out, struct Pixel *d_in, int w, int h)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < w && row < h) {
		int offset = row * w + col;
		d_out[offset] = round(0.21 * d_in[offset].r 
				+ 0.71 * d_in[offset].g + 0.07 * d_in[offset].b);
	}
}

void colorToGreyscale(unsigned char *h_out, struct Pixel *h_in, int w, int h)
{
	struct Pixel *d_in;
	unsigned char *d_out;

	int colorImageSize = h * w * sizeof(struct Pixel);
	int greyImageSize = h * w * sizeof(unsigned char);

	cudaMalloc((void **) &d_in, colorImageSize);
	cudaMalloc((void **) &d_out, greyImageSize);

	cudaMemcpy(d_in, h_in, h * w * sizeof(struct Pixel), cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(w / (float) 8), ceil(h / (float) 8), 1);
	dim3 dimBlock(8, 8, 1);
	colorToGreyscaleKernel<<<dimGrid,dimBlock>>>(d_out, d_in, w, h);

	cudaMemcpy(h_out, d_out, h * w * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

int main(int argc, char *argv[])
{
	int colorImageSize = HEIGHT * WIDTH * sizeof(struct Pixel);
	int greyImageSize = HEIGHT * WIDTH * sizeof(unsigned char);

	struct Pixel *h_in = (struct Pixel *) malloc(colorImageSize);
	unsigned char *h_out = (unsigned char *) malloc(greyImageSize);

	printf("Color image:\n");
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			// Randomly generate a color image
			h_in[i * WIDTH + j].r = rand() % 256;
			h_in[i * WIDTH + j].g = rand() % 256;
			h_in[i * WIDTH + j].b = rand() % 256;
			printf("Input pixel (%d, %d) = (%d, %d, %d)\n", i, j, 
					h_in[i * WIDTH + j].r, 
					h_in[i * WIDTH + j].g, 
					h_in[i * WIDTH + j].b);
		}
	}

	colorToGreyscale(h_out, h_in, WIDTH, HEIGHT);

	printf("Greyscale image:\n");
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			printf("Output pixel (%d, %d) = %d\n", 
					i, j, h_out[i * WIDTH + j]);
		}
	}

	free(h_in);
	free(h_out);
	return 0;
}

