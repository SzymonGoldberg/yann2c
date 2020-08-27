#include <stdlib.h>
#include "matrix.h"
#include "deep_network.h"
#include "cnn.h"

//================ FUNKCJE DO CNN ====================================

int
cnn_count_kernel(	unsigned input_x,	unsigned input_y,
			unsigned krnl_x,	unsigned krnl_y,
			unsigned *out_x,	unsigned *out_y,
			unsigned stride)
{
	int x = 1, y = 1, aux = input_x - krnl_x;
	double hlp;

	if(aux < 0) return -1;
	if(aux > 0)
	{
		hlp = aux/stride;
		if(hlp == (double)(int)hlp) x += (int)hlp;
		else return -1; 
	}

	aux = input_y - krnl_y;
	if(aux < 0) return -1;
	if(aux > 0)
	{
		hlp = aux/stride;
		if(hlp == (double)(int)hlp) y += (int)hlp;
		else return -1;
	}

	if(out_x != NULL) *out_x = x;
	if(out_y != NULL) *out_y = y;

	return x*y;
}

matrix_t* 
cnn_crop_input(	const matrix_t* input, const matrix_t* kernel, unsigned stride)
{
	if(input == NULL || kernel == NULL) return NULL;
	//if(output->x != matrix_size(kernel)) return 1; //DEBUG
	if(stride < 1) return NULL;

	int krnl_count = cnn_count_kernel(	input->x, input->y, kernel->x,
						kernel->y, NULL, NULL, stride);

//	if(output->y != (unsigned) krnl_count) return 1; //DEBUG

	matrix_t* output = matrix_alloc(matrix_size(kernel), krnl_count);
	if(output == NULL) return NULL;
	
	int krnl_pos_x = 0, krnl_pos_y = 0, out_x = 0;
	for(int i = 0; i  < krnl_count; ++i)
	{
		for(unsigned x = krnl_pos_x; x < (kernel->x + krnl_pos_x); ++x)
			for(unsigned y = krnl_pos_y; y < (kernel->y + krnl_pos_y); ++y)
				output->matrix[(out_x++) + output->x * i] =
						input->matrix[x + input->x * y];
		krnl_pos_x += stride;

		if(krnl_pos_x + kernel->x > input->x)
		{
			krnl_pos_x = 0;
			krnl_pos_y++;
			out_x = 0;
		}
	}

	return output;
}

struct cnn_array*
cnn_create(unsigned input_x, unsigned input_y)
{
	struct cnn_array* a = (struct cnn_array*)
					calloc(1, sizeof(struct cnn_array*));
	if(a != NULL)
	{
		a->head = NULL; a->head = NULL;
		a->in_x = input_x; a->in_y = input_y;
	}
	return a;
}

int
cnn_add_layer(struct cnn_array* cnn, unsigned krnl_x, unsigned krnl_y, unsigned
		stride, void (*activation_func)(double*, unsigned))
{
	if(cnn == NULL || !stride) return 1;

	struct cnn_layer* layer;
	layer = (struct cnn_layer*)calloc(1, sizeof(struct cnn_layer*));	
	if(layer == NULL) return 1;

	layer->stride = stride;
	layer->activation_func = activation_func;

	unsigned out_x, out_y, krnl_count;

	krnl_count = (cnn->head == NULL) ?
		cnn_count_kernel(cnn->in_x,cnn->in_y,krnl_x,krnl_y,&out_x,&out_y, stride):
		cnn_count_kernel(cnn->tail->output->x, cnn->tail->output->y,
					krnl_x, krnl_y, &out_x, &out_y, stride);

	if(krnl_count < 1) { free(layer); return 1;}

	layer->output	= matrix_alloc(out_x, out_y);
	layer->kernel	= matrix_alloc(krnl_x, krnl_y);
	layer->delta	= matrix_alloc(out_x, out_y);
	layer->weight_delta	= matrix_alloc(out_x, out_y);

	if(	layer->output == NULL	|| layer->kernel == NULL || 
		layer->delta == NULL	|| layer->weight_delta == NULL)
	{
		cnn_free_layer(layer);
		return 1;
	}

	if(cnn->head == NULL)
	{
		cnn->head = layer;
		layer->prev = NULL;
	}
	else
	{
		cnn->tail->next = layer;
		layer->prev = cnn->tail;
	}
	layer->next = NULL;
	cnn->tail = layer;
	return 0;
}

int
cnn_add_fcl(struct cnn_array* cnn, unsigned output_size, void
		(*activation_func)(double*, unsigned))
{
	if(cnn == NULL || !output_size) return 1;
	if(cnn->fcl != NULL) return 1;

	cnn->fcl = nn_create();
	if(cnn->fcl == NULL) return 1;

	unsigned input_size = matrix_size(cnn->fcl->tail->output);

	if(nn_add_layer(cnn->fcl, output_size, input_size, 1, activation_func, 0.0))
	{
		nn_free(cnn->fcl);
		return 1;
	}
	return 0;
}

//NOT DONE YET!!!
int
cnn_predict(struct cnn_array* cnn, const matrix_t* input)
{
	if(cnn == NULL || cnn->tail == NULL || cnn->head == NULL) return 1;
	
//alokacja pamieci na obrobiona macierz wejsciowa
	struct cnn_layer* cnn_ptr = cnn->head;
	do {

		cnn_ptr = cnn_ptr->next;
	} while(cnn_ptr != NULL);
	return 0;
}

void
cnn_free_layer(struct cnn_layer* layer)
{
	if(layer != NULL)
	{
		if(layer->kernel != NULL) matrix_free(layer->kernel);
		if(layer->output != NULL) matrix_free(layer->output);
		if(layer->delta != NULL) matrix_free(layer->delta);
		if(layer->weight_delta != NULL) matrix_free(layer->weight_delta);
		if(layer->crp_in != NULL) matrix_free(layer->crp_in);
		free(layer);
	}
}

void
cnn_free(struct cnn_array* cnn)
{
	if(cnn == NULL) return;
	if(cnn->tail == NULL) { free(cnn); return; }
	
	struct cnn_layer *ptr = cnn->head;
	struct cnn_layer *aux = cnn->head;
	do {
		ptr = aux->next;
		cnn_free_layer(aux);
		aux = ptr;
	} while(ptr != NULL);
	free(cnn);
}
