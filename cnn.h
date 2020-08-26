#ifndef __CNN_H_
#define __CNN_H_

//====== CNN ======

struct cnn_layer {
	matrix_t *kernel;
	matrix_t *output;

	matrix_t *delta;
	matrix_t *weight_delta;

	void (*activation_func)(double *, unsigned);	//funkcja aktywacji
	unsigned stride;

	struct cnn_layer *next;
	struct cnn_layer *prev;
};

struct cnn_array {
	struct cnn_layer *head;
	struct cnn_layer *tail;

	unsigned in_x, in_y;
};

//---====== DEKLARACJE FUNKCJI DO CNN =====---

int
cnn_count_kernel(	unsigned input_x,	unsigned input_y,
			unsigned krnl_x,	unsigned krnl_y,
			unsigned *out_x,	unsigned *out_y,
			unsigned stride);

int
cnn_crop_input(	const matrix_t* input, const matrix_t* kernel,
			const matrix_t* output, unsigned stride);

struct cnn_array*
cnn_create(unsigned input_x, unsigned input_y);

void cnn_free_layer(struct cnn_layer* layer);

#endif
