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


//funkcja tworzy pusta strukture sieci
//w przypadku sukcesu zwraca jej adres, w innym przypadku zwraca NULL
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


//funkcja dodaje siec <size> neuronow na koniec struktury <nn> (tail)
//jesli struktura byla pusta to pierwsza macierz ma wymiary <input>, <size>
//w przypadku istniejacych juz wczesniej warstw
//funkcja ma rozmiar wyjscie poprzedniej warstwy, <size>
//zwraca 0 w przypadku porazki, w innym przypadku 1
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


//zwalnia cala pamiec przydzielona na strukture <nn>
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
}

//przelicza odpowiedz sieci <nn> na wejscie <input> w postaci wektora
//czyli macierzy dla ktorej y = 1
//w przypadku sukcesu 0, w innym wypadku !0
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
 		if(i) {
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
	} while(1);
}

//wyswietlanie <nn> w formie macierzy wag i ostatnich odpowiedzi
void
nn_display(const struct nn_array *nn)
{
	if(nn = NULL) return;
	struct nn_layer *ptr = nn->head;
	while(ptr != NULL)
	{
		puts("WEIGHTS:\n");
		matrix_display(*(ptr->weights));
		puts("OUTPUT:\n");
		matrix_display(*(ptr->output));
 		if(ptr = nn->head) puts("(HEAD)\n");
 		if(ptr = nn->tail) puts("(TAIL)\n");
		if(ptr->next != NULL) puts("|\n|\nV\n");
		ptr = ptr->next;
	}

}
