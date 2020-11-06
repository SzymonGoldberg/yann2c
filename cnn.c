#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "deep_network.h"
#include "cnn.h"

//================ FUNKCJE DO CNN ====================================

int
cnn_count_krnl( unsigned in_x, unsigned in_y, 
				const matrix_t* krnl, 
				unsigned *out_x, unsigned *out_y, unsigned stride)
{
	if(out_x == NULL || out_y == NULL) return -1;
	
	double aux = ((double)(in_x - krnl->x)/stride);
	if(aux < 1) return -1;
	*out_x = aux + 1;
	
	aux = ((double)(in_y - krnl->y)/stride);
	if(aux < 1) return -1;
	*out_y = aux + 1;

	return ((*out_x) * (*out_y));
}

void
copy_matrix_to_row( const matrix_t* cpy,idx_matrix_t* dest, unsigned row, 
					unsigned dest_x,	unsigned dest_y,
					unsigned offset_x,	unsigned offset_y){
	row *= dest->x;

	//filing destination index matrix with -1
	for(unsigned i = 0; i < (dest_x * dest_y); ++i)	dest->m[i + row] = -1.0;

	//writing moved kernels indxs do index matrix
	for(unsigned y = 0; y < dest_y - offset_y; ++y)
	for(unsigned x = 0; x < dest_x - offset_x; ++x)
		dest->m[(x+offset_x + ((y+offset_y) * dest_x)) + row] = (x >= cpy->x || y >= cpy->y) ? (-1.0) : x + y * cpy->x;
}

int 
cnn_conv_create(unsigned in_x, unsigned in_y, const matrix_t* kernel, idx_matrix_t* output,
		unsigned stride)
{
	if(kernel == NULL || stride < 1) return 1;
	if(output->x != (in_x * in_y)) return 1;

	unsigned x_krnl, y_krnl;	
	int krnl_count = cnn_count_krnl(in_x, in_y, kernel, &x_krnl, &y_krnl, stride);
	
	if(output->y != (unsigned) krnl_count) return 1;
	if(krnl_count < 1) return 1;
	
	int counter = 0;
	for(unsigned y = 0; y < y_krnl; ++y)
	for(unsigned x = 0; x < x_krnl; ++x)
		copy_matrix_to_row(kernel, output, counter++, in_x, in_y, x, y);
	
	return 0;
}