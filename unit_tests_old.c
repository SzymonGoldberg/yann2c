#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include "matrix.h"
#include "deep_network.h"

int main (void)
{


//=== TESTY SIECI GLEBOKICH ===

	printf("TEST 13 (gleboka siec 3x3)\n");

	matrix_t* input = matrix_alloc(3, 4);

	input->matrix[0] = 8.5; input->matrix[1] = 0.65; input->matrix[2] = 1.2;
	input->matrix[3] = 9.5; input->matrix[4] = 0.8;  input->matrix[5] = 1.3;
	input->matrix[6] = 9.9; input->matrix[7] = 0.8;	 input->matrix[8] = 0.5;
	input->matrix[9] = 9.0; input->matrix[10] = 0.9; input->matrix[11] = 1.0;

	struct matrix_array *neural_network = matrix_array_create();


	matrix_array_append(neural_network, 3, 3);

	matrix_fill(neural_network->tail->matrix, 9,	0.1, 0.2, -0.1,
							-0.1, 0.1, 0.9,
       		 					0.1, 0.4, 0.1);

	aux = matrix_array_append_network(neural_network, 3, 0, 0, 0);

	matrix_fill(neural_network->tail->matrix, 9,	0.3, 1.1, -0.3,
							0.1, 0.2, 0.0,
	       		 				0.0, 1.3, 0.1);

	if(aux) printf("---funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	matrix_t* output1 = deep_neural_network(*input, *neural_network);

	double exp_tab[] = {	0.214,	0.145,	0.507,
				0.204,	0.158,	0.53,
				-0.584, 0.018,	-0.462,
       		 		-0.015, 0.116,	0.253};

	err = 0;
	for(int i = 0; i < 12; ++i)
	{
		if( output1->matrix[i] > exp_tab[i] + 0.001 ||
		output1->matrix[i] < exp_tab[i] - 0.001)
		{
			printf("---funkcja deep_neural_network zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_tab[i], output1->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_array_free(neural_network);
	matrix_free(input);
	matrix_free(output1);

//===== TEST OUTER PRODUCT =====

	printf("TEST 16 (iloczyn zewnetrzny)\n");

	a = matrix_alloc(3, 1);
	matrix_fill(a, 3,	0.455, -0.02, 0.865);

	b = matrix_alloc(3, 1);
	matrix_fill(b, 3,	8.5, 0.65, 1.2);

	c = matrix_alloc(3, 3);

	outer_product(*a, *b, c);

	double exp_outer[] =	{3.8675, 0.29575, 0.546,
				-0.17, -0.013, -0.024,
				7.3525, 0.56225, 1.038};

	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( c->matrix[i] > exp_outer[i] + 0.001 ||
			c->matrix[i] < exp_outer[i] - 0.001)
		{
			printf("---funkcja matrix_substraction zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_outer[i], c->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);

	return 0;
}
