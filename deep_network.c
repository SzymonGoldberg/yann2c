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
deep_neural_network(matrix_t input, struct matrix_array network)
{
//sprawdzam dane wejsciowe
	if(network.head == NULL) return NULL;
	if(network.tail == NULL) return NULL;
 	if(input.x != network.head->matrix->x) return NULL;

//wskazniki pomocnicze
	matrix_t* mid_output = &input;
	matrix_t* output;
	int i = 0;
	struct matrix_node * ptr = network.head;

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
	if(a->weights == NULL)
	{
		free(a);
		return NULL;
	}

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
nn_add_layer(struct nn_array *nn, unsigned size, unsigned input)
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
		nn->tail = new_layer;
		nn->head = new_layer;
	}
	else
	{
 		new_layer = nn_layer_create(nn->tail->weights->y, size);
		if(new_layer == NULL) return 1;
		nn->tail->next = new_layer;
		new_layer->prev = nn->tail;
		nn->tail = new_layer;

	}

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
 	if(nn == NULL) return 1;
	if(input == NULL) return 1;
	if(input->x != nn->head->weights->x) return 1;

	struct nn_layer *ptr = nn->head;
//zmienna pomocnicza - jest tutaj po to by przemnozyc tylko raz przez wejscie
//pozostale mnozenia wykonywac na poprzednich wyjsciach neuronow
	int i = 0;
	do {
 		if(!i) {
			matrix_multiply( *input,
					*(ptr->weights),
					ptr->output, 1);
			i++;
		}
		else {
                 	matrix_multiply(*(ptr->prev->output),
					*(ptr->weights),
					ptr->output, 1);
		}
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

	struct matrix_array * delta_array = matrix_array_create();
	struct nn_layer *nn_ptr = nn->tail;

//obliczanie delty dla poszczegolnych warstw
	do
	{
		if(nn_ptr == nn->tail)
		{
			if(matrix_array_append_front(delta_array, expected_output->x, 1))
                        	return 1;
			//last_layer_delta = layer_output - expeced_output
			if(matrix_substraction(*(nn_ptr->output), *expected_output,
				delta_array->head->matrix))
			{
					matrix_array_free(delta_array);
					return 1;
			}
		}
		else
		{
                 	if(matrix_array_append_front(delta_array,
				delta_array->head->matrix->y, nn_ptr->output->x))
			{
					matrix_array_free(delta_array);
					return 1;
			}
			//layer_delta = next_layer_delta * next_layer_output
			if(matrix_multiply(*(delta_array->head->next->matrix),
				*(nn_ptr->next->weights), delta_array->head->matrix, 0))
			{
					matrix_array_free(delta_array);
					return 1;
			}

		}
		nn_ptr = nn_ptr->prev;
	} while(nn_ptr != NULL);

	nn_ptr = nn->tail;
	struct matrix_node *delta_ptr = delta_array->tail;

	while(TRUE)
	{
		//obliczanie delty wag dla poszczegolnych warstw
		if(nn_ptr == nn->head)
		{
			//layer_weight_delta = input * layer_delta
			if(matrix_multiply(*input, *(delta_ptr->matrix),
					delta_ptr->matrix,0))
			{
					matrix_array_free(delta_array);
					return 1;
			}
			break;
		}
		else
		{
			//layer_weight_delta = prev_layer_output * layer_delta
			if(matrix_multiply(*(nn_ptr->prev->output),
					*(delta_ptr->matrix),
					delta_ptr->matrix,0))
			{
					matrix_array_free(delta_array);
					return 1;
			}
		}

		//alpha * weight_delta
      		matrix_multiply_by_num(delta_ptr->matrix, a);

		//layer_weights = layer_weights - alpha * weight_delta
		matrix_substraction(*nn_ptr->weights, *delta_ptr->matrix, nn_ptr->weights);


		nn_ptr = nn_ptr->prev;
		delta_ptr = delta_ptr->prev;
	}

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
