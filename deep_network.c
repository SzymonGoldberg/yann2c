#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "deep_network.h"

matrix_t *
neural_network(const matrix_t input, const matrix_t weights)
{
//alokuje pamiec na output
	matrix_t *output = matrix_alloc(weights.y, input.y);
	if(output == NULL) return NULL;

	if(matrix_multiply(input, weights, output, 1) != 0)
	{
		matrix_free(output);
		return NULL;
	}

//zwracam macierz z wyjsciem
	return output;
}

matrix_t *
deep_neural_network(matrix_t input, struct matrix_dl_array network)
{
//sprawdzam dane wejsciowe
	if(network.matrix == NULL) return NULL;
 	if(input.x != network.matrix->x) return NULL;

//wskazniki pomocnicze
	matrix_t* mid_output = &input;
	matrix_t* output;
	int i = 0;
	struct matrix_dl_array * ptr = &network;

	while(ptr != NULL)
	{
        	output = neural_network(*mid_output, (*ptr->matrix));
		if(output == NULL) return NULL;

	//ponizszy if jest tutaj po to by nie wolnic niealokownaej
	//wczesniej pamieci czyli wejscia (matrix_t input). Kazda
	//nastepna macierz bedzie bez problemu walniana
		if(i++) matrix_free(mid_output);
		mid_output = output;
		ptr = ptr->next;
	}
//zwracam macierz z odpowiedziami
	return output;
}
