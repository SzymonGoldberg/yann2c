#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "deep_network.h"


//======= FUNKCJE OBSLUGUJE STRUKTURY DANYCH SIECI =======

struct nn_array *
nn_create(void)
{
	struct nn_array *a = (struct nn_array *)
				calloc(1, sizeof(struct nn_array));
	if(a == NULL) return NULL;
	a->tail = NULL; a->head = NULL;
	return a;
}

struct nn_layer *
nn_layer_create(unsigned x, unsigned y)
{
	struct nn_layer *a = (struct nn_layer *)
					calloc(1, sizeof(struct nn_layer));
	if(a == NULL) return NULL;

	a->weights = matrix_alloc(x, y);
	if(a->weights == NULL) { free(a); return NULL; }

	a->output = matrix_alloc(y, 1);
	if(a->weights == NULL)
	{
		matrix_free(a->weights);
		free(a);
		return NULL;
	}

	a->prev = NULL; a->next = NULL;
	return a;
}


int
nn_add_layer(struct nn_array *nn, unsigned size, unsigned input,
	int (*activation_func)(matrix_t *, unsigned))
{
//sprawdzam dane wejsciowe
	if(nn == NULL) return 1;
	if(!input || !size) return 1;

//alokuje pamiec
  	struct nn_layer *new_layer;
	if(nn->tail == NULL)
	{
 		new_layer = nn_layer_create(input, size);
		if(new_layer == NULL) return 1;
		nn->head = new_layer;
	}
	else
	{
 		new_layer = nn_layer_create(nn->tail->weights->y, size);
		if(new_layer == NULL) return 1;
		nn->tail->next = new_layer;
		new_layer->prev = nn->tail;
	}
	nn->tail = new_layer;
	nn->tail->activation_func = activation_func;

	return 0;
}


void
nn_free(struct nn_array *nn)
{
	if(nn == NULL) return;
	if(nn->tail == NULL) { free(nn); return; }
	
	struct nn_layer *ptr = nn->head;
	struct nn_layer *aux = nn->head;
	do {
		ptr = aux->next;

		if(aux->weights != NULL) matrix_free(aux->weights);
		if(aux->output != NULL) matrix_free(aux->output);
		free(aux);

		aux = ptr;
	} while(ptr != NULL);

	free(nn);
}


int
nn_predict(struct nn_array *nn, const matrix_t *input)
{
//walidacja danych wejsciowych
 	if(nn == NULL) return 1;
	if(input == NULL) return 1;
	if(input->x != nn->head->weights->x) return 1;

	struct nn_layer *ptr = nn->head;
	do {
 		if(ptr == nn->head) {
			matrix_multiply( *input,
					*(ptr->weights),
					ptr->output, 1);
		}
		else {
                 	matrix_multiply(*(ptr->prev->output),
					*(ptr->weights),
					ptr->output, 1);
		}

	//stosuje funkcje aktywacji na wyjsciu danej warstwy (jesli jakas funkcja jest)
		if(ptr->activation_func != NULL)
			(ptr->activation_func)(ptr->output, 0);

		ptr = ptr->next;
	} while(ptr != NULL);

	return 0;
}


int
nn_backpropagation(struct nn_array *nn, const matrix_t * input,
	const matrix_t* expected_output, double a)
{
//sprawdzanie danych wejsciowych
	if(nn ==NULL || expected_output == NULL) return 1;
	if(nn->tail->output->x != expected_output->x) return 1;

//nn_predict
	if(nn_predict(nn, input)) return 1;

//zmienne pomocniczne
	struct matrix_array * delta_array = matrix_array_create();//tablica delt
	struct nn_layer *nn_ptr = nn->tail;//wskaznik na pojedyncza warstwe neuronow
	int aux = 0;//zmienna pomocnicza z wartosciami jakie zwracaja funkcje w petli

//obliczanie delty dla poszczegolnych warstw
	do {
		if(nn_ptr == nn->tail)
		{
			aux=matrix_array_append_front(delta_array, expected_output->x, 1);
			//last_layer_delta = layer_output - expeced_output
			if(!aux) {
				aux = matrix_substraction(*(nn_ptr->output),
					*expected_output, delta_array->head->matrix);
			}
		}
		else
		{
                 	aux = matrix_array_append_front(delta_array,
				delta_array->head->matrix->y, nn_ptr->output->x);
			//layer_delta = next_layer_delta * next_layer_output
			if(!aux) {
				aux = matrix_multiply(*(delta_array->head->next->matrix),
				*(nn_ptr->next->weights), delta_array->head->matrix, 0);
			}

		}
		if(aux) { matrix_array_free(delta_array); return 1; }
		nn_ptr = nn_ptr->prev;
	} while(nn_ptr != NULL);

	nn_ptr = nn->tail;
	struct matrix_node *delta_ptr = delta_array->tail;//wskaznik na poj. delte

	do {
	//obliczanie delty wag dla poszczegolnych warstw
		if(nn_ptr == nn->head) {
			//layer_weight_delta = input * layer_delta
			aux = outer_product(*input, *(delta_ptr->matrix), delta_ptr->matrix);
		}
		else {
			//layer_weight_delta = prev_layer_output * layer_delta
			aux = outer_product(*(nn_ptr->prev->output),
					*(delta_ptr->matrix),
					delta_ptr->matrix);
		}
		if(aux) { matrix_array_free(delta_array); return 1; }

	//alpha * weight_delta
      		matrix_multiply_by_num(delta_ptr->matrix, a);

	//zmiana wartosci wag o delte wagi * alpha
		matrix_substraction(*nn_ptr->weights, *delta_ptr->matrix,
			nn_ptr->weights);

		nn_ptr = nn_ptr->prev;
		delta_ptr = delta_ptr->prev;
	}
	while(nn_ptr != NULL);

	matrix_array_free(delta_array);
	return 0;
}


void
nn_display(const struct nn_array *nn)
{
	if(nn == NULL) return;
	struct nn_layer *ptr = nn->head;
	while(ptr != NULL)
	{
		puts("WEIGHTS:");
		matrix_display(*(ptr->weights));
		puts("\nOUTPUT:");
		matrix_display(*(ptr->output));
 		if(ptr == nn->head) puts("(HEAD)");
 		if(ptr == nn->tail) puts("(TAIL)");
		if(ptr->next != NULL) puts("|\n|\nV\n");
		ptr = ptr->next;
	}

}
