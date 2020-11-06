#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "cnn.h"

int main (void)
{
    matrix_t krnl = {.x = 2, .y = 2};
    double m[4] = {1.0, 3.0, 5.0, 2.0};
    krnl.matrix = m;

    int tab[9 * 4] = {0};
    idx_matrix_t a = {.x = 9, .y = 4};
    a.m = tab;

    matrix_t* input = matrix_alloc(9, 1);
    matrix_fill(input, 9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    printf("conv create return = %i\n", cnn_conv_create(3, 3, &krnl, &a, 1));

    for(unsigned y = 0; y < a.y; ++y)
	{
		for(unsigned x = 0; x < a.x; ++x)
		{
			printf("%i\t",a.m[x + y*a.x]);
		}
		printf("\n");
	}

    matrix_t *result = matrix_alloc(1, 4);
    matrix_multiply_indx(krnl, *input, &a, result, 0, 1, 0);
    
    matrix_display(*result);

    free(result);
    return 0;
}