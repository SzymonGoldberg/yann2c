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


struct matrix_dl_array *
matrix_dll_array_create_elem(unsigned x, unsigned y)
{
//alokacja pamieci na komorke listy z macierza
	struct matrix_dl_array *
        	result = (struct matrix_dl_array *)
			calloc(1, sizeof(struct matrix_dl_array));

	if(result == NULL) return NULL;

//alokacja pamieci strukture macierzy
	(*result).matrix = matrix_alloc(x, y);
	if( (*result).matrix == NULL )
	{
         	free(result);
		return NULL;
	}

//co prawda nie powinno byc z tym problemu bo calloc wszystko zeruje przy
//alokacji ale dla swietego spokoju ustawiam pozostale zmienne w strukturze
//z lista
	result->next = NULL;
	result->prev = NULL;

 	return result;
}

int
matrix_dll_array_append(struct matrix_dl_array * array, unsigned int n,
char random_weight_flag, double weight_min_value, double weight_max_value)
{
//sprawdzam dane wejsciowe
	if(array == NULL) return 1;
	if(n == 0) return 1;

//alokuje pamiec na nowy element listy
	array->next = matrix_dll_array_create_elem(array->matrix->y, n);
	if(array->next == NULL) return 1;

//ustawiam odpowiednio zmienne w nowym elemencie
	array->next->prev = array;

//wypelniam losowymi wartosciami (jesli trzeba)
	if(random_weight_flag)
		matrix_fill_rng(array->matrix, (int)weight_min_value, (int)weight_max_value);

	return 0;
}

void
matrix_dll_array_free_elem(struct matrix_dl_array *array)
{
	if(array == NULL) return;
	matrix_free((*array).matrix);
	free(array);
 	
}


void
matrix_dll_array_free(struct matrix_dl_array *array)
{
 	if(array == NULL) return;
//przeszukuje liste w poszukiwaniu konca
	while(array->prev != NULL)
		array = array->prev;

	struct matrix_dl_array *ptr;
	do {
		ptr = array->next;
		matrix_dll_array_free_elem(array);
		array = ptr;
	} while(ptr != NULL);
}
