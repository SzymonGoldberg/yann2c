#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "deep_network.h"

//-------------------------------------------------------------------
//PLIK POMOCNICZY - DO USUNIECIA W KONCOWEJ WERSJI !!!!
//-------------------------------------------------------------------

int main (void)
{
	srand(time(NULL));

	struct nn_array *nn = nn_create();

	nn_add_layer(nn, 3, 3, 4, NULL, 0.3);
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
						0.1, 0.4, 0.1);
	nn_add_layer(nn, 3, 3, 4, NULL, 0);
	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					       	0.1, 0.2, 0.0,
						0.0, 1.3, 0.1);
	matrix_t* a = matrix_alloc(3, 4);
	matrix_fill(a, 12,	8.5, 0.65, 1.2,
				9.5, 0.8, 1.3,
				9.9, 0.8, 0.5,
				9.0, 0.9, 1.0);
	matrix_t* b = matrix_alloc(3, 4);
	matrix_fill(b, 12,	0.1, 1.0, 0.1,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.1,
				0.1, 1.0, 0.2);


	nn_predict(nn, a, 1);	//DEBUG
	nn_backpropagation(nn, a, b, 0.01, 1);

	nn_display(nn);

	matrix_free(a);
	matrix_free(b);
	nn_free(nn);

	return 0;
}
