#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "deep_network.h"

//======= FUNKCJE OBSLUGUJACE STRUKTURY DANYCH SIECI =======

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
nn_layer_create(unsigned x, unsigned y, unsigned batch_size)
{
	if(!x || !y || !batch_size) return NULL;
	struct nn_layer *a = (struct nn_layer *)
					calloc(1, sizeof(struct nn_layer));
	if(a == NULL) return NULL;

	a->weights = matrix_alloc(x, y);
	if(a->weights == NULL) { free(a); return NULL; }

	a->output = matrix_alloc(y, batch_size);
	if(a->weights == NULL)
	{
		matrix_free(a->weights);
		free(a);
		return NULL;
	}

	a->prev = NULL; a->next = NULL; a->delta = NULL;
	return a;
}


int
nn_add_layer(struct nn_array *nn, unsigned size, unsigned input,
	unsigned batch_size, void (*activation_func)(double *, unsigned))
{
	if(nn == NULL) return 1;

  	struct nn_layer *new_layer;
	if(nn->tail == NULL)
	{
		if(!input) return 1;
 		new_layer = nn_layer_create(input, size, batch_size);
		if(new_layer == NULL) return 1;
		nn->head = new_layer;
	}
	else
	{
 		new_layer = nn_layer_create(nn->tail->weights->y, size, batch_size);
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
		if(aux->delta != NULL) matrix_free(aux->delta);
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
			for(unsigned i = 0; i < (ptr->output->x) * (ptr->output->y); ++i)
				(ptr->activation_func)(&(ptr->output->matrix[i]), 0);

		ptr = ptr->next;
	}
	while(ptr != NULL);

	return 0;
}


//<delta> = <delta> o (funkcja aktywacji)<layer->output>
int
nn_hadamard(struct nn_layer *layer)
{
	if(layer == NULL || layer->delta == NULL) return 1;
	if(layer->activation_func == NULL) return 1;

	double a;
	for(unsigned i = 0; i < (layer->delta->y) * (layer->delta->x); ++i)
	{
		a = layer->output->matrix[i];
		(layer->activation_func)(&a, 1);
		layer->delta->matrix[i] = layer->delta->matrix[i] * a;
	}
	return 0;
}


int
nn_backpropagation(struct nn_array *nn, const matrix_t * input,
	const matrix_t* exp_output, double a)
{
//sprawdzanie danych wejsciowych
	if(nn ==NULL || exp_output == NULL) return 1;
	if(nn->tail->output->x != exp_output->x) return 1;

//nn_predict
	if(nn_predict(nn, input)) return 1;
	
//zmienne pomocniczne
	struct nn_layer *nn_ptr = nn->tail;//wskaznik na pojedyncza warstwe neuronow

//alokacja pamieci na delty jesli nie zostaly wczesniej nie zaalokowane
	if(nn_ptr->delta == NULL)
	{
		do {
			if(nn_ptr == nn->tail)
				nn_ptr->delta = matrix_alloc(exp_output->x, 1);
			else
				nn_ptr->delta = matrix_alloc(nn_ptr->next->output->x,
							nn_ptr->next->delta->y);
			nn_ptr = nn_ptr->prev;
		} while(nn_ptr != NULL);
	}

	nn_ptr = nn->tail;
//obliczanie delty dla poszczegolnych warstw
	do {
		if(nn_ptr == nn->tail)
		{
		//last_layer_delta = layer_output - expeced_output
			matrix_substraction(*(nn_ptr->output),
				*exp_output, nn_ptr->delta);
			matrix_display(*nn_ptr->delta);	//debug
		}
		else
		{
		//layer_delta = next_layer_delta * next_layer_output
			matrix_multiply(*(nn_ptr->next->delta),
			*(nn_ptr->next->weights), nn_ptr->delta, 0);
			matrix_display(*nn_ptr->delta);	//debug
		}

	//layer_delta = layer_delta o activation_func(layer_output)
		if(nn_ptr->activation_func != NULL) nn_hadamard(nn_ptr);

		nn_ptr = nn_ptr->prev;
	} while(nn_ptr != NULL);

	nn_ptr = nn->tail;
	matrix_t *mtrx;	//zmienna pomocnicza do ktorej wpisuje iloczyn zew

//obliczanie delty wag dla poszczegolnych warstw i ew zmiana wartosci wag
	do {
		if(nn_ptr == nn->head) {
			mtrx = matrix_alloc(input->x, nn_ptr->delta->x);
			
		//layer_weight_delta = layer_delta o input 
			outer_product(*(nn_ptr->delta), *input, mtrx);
		}
		else {
			mtrx = matrix_alloc(nn_ptr->prev->output->x, nn_ptr->delta->x);
			
		//layer_weight_delta = layer_delta o prev_layer_output
			outer_product( *(nn_ptr->delta),*(nn_ptr->prev->output), mtrx);
		}
	//weight_delta *= alpha
      		matrix_multiply_by_num(mtrx, a);

	//layer_weights = layer_weights - delta_weight * alpha
		matrix_substraction(*nn_ptr->weights, *mtrx, nn_ptr->weights);

		matrix_free(mtrx);
		nn_ptr = nn_ptr->prev;
	}
	while(nn_ptr != NULL);

	return 0;
}

//---======= DO DEBUGU, WYWALIC W KONCOWEJ WERSJI ======---

void
nn_display(const struct nn_array *nn)
{
	if(nn == NULL) return;
	struct nn_layer *ptr = nn->head;
	while(ptr != NULL)
	{
		if(ptr->delta!= NULL) puts("IS_DELTA");	//DEBUG
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

//---======= FUNKCJE AKTYWACJI =======---

void
ReLU(double *a, unsigned derivative) {
	*a = derivative ? RELU_DERIV(*a) : MAX(*a, 0);
}

void
sigmoid(double *a, unsigned derivative) {
 	*a = derivative ? SIGMOID_DERIV(*a) : SIGMOID(*a);
}

void
xtanh(double *a, unsigned derivative) {
	*a = derivative ? TANH_DERIV(*a) : TANH(*a);
}
