#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 15
#define HEIGHT 10
#define BLUR_SIZE 1

// Three channels corresponding to RGB
struct Pixel {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

void imageBlur(struct Pixel *h_out, struct Pixel *h_in, int w, int h)
{
	for(int row = 0; row < h; row++) {
		for(int col = 0; col < w; col++) {
			int sumPixRed = 0;
			int sumPixGreen = 0;
			int sumPixBlue = 0;
			int numPix = 0;

			// Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
			for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
				for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
					int curRow = row + blurRow;
					int curCol = col + blurCol;

					// Verify we have a valid image pixel
					if (curRow >= 0 && curRow <= h - 1 && curCol >= 0 && curCol <= w - 1) {
						// Keep track of number of pixels in the average
						sumPixRed += h_in[curRow * w + curCol].r;
						sumPixGreen += h_in[curRow * w + curCol].g;
						sumPixBlue += h_in[curRow * w + curCol].b;
						numPix++;
					}
				}
			}

			// Write our new pixel value out
			h_out[row * w + col].r = (unsigned char) round(sumPixRed / (float) numPix);
			h_out[row * w + col].g = (unsigned char) round(sumPixGreen / (float) numPix);
			h_out[row * w + col].b = (unsigned char) round(sumPixBlue / (float) numPix);
		}
	}
}

int main(int argc, char *argv[])
{
	int size = HEIGHT * WIDTH * sizeof(struct Pixel);

	struct Pixel *h_in = (struct Pixel *) malloc(size);
	struct Pixel *h_out = (struct Pixel *) malloc(size);

	printf("Clear image:\n");
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

	imageBlur(h_out, h_in, WIDTH, HEIGHT);

	printf("Blur image:\n");
	for(int i = 0; i < HEIGHT; i++) {
		for(int j = 0; j < WIDTH; j++) {
			printf("Output pixel (%d, %d) = (%d, %d, %d)\n", i, j, 
					h_out[i * WIDTH + j].r, 
					h_out[i * WIDTH + j].g, 
					h_out[i * WIDTH + j].b);
		}
	}

	free(h_in);
	free(h_out);
	return 0;
}

