#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 15
#define HEIGHT 10

struct Pixel {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

void colorToGreyscale(unsigned char *h_out, struct Pixel *h_in, int w, int h)
{
	for(int i = 0; i < h; i++) {
		for(int j = 0; j < w; j++) {
			h_out[i * w + j] = round(0.21 * h_in[i * w + j].r 
					+ 0.71 * h_in[i * w + j].g + 0.07 * h_in[i * w + j].b);
		}
	}
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

