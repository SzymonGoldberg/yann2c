#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include "matrix.h"

//======= FUNKCJE DO MACIERZY (MATRIX_T) =======

matrix_t*
matrix_alloc(unsigned x, unsigned y)
{
	//sprawdzam poprawnosc podanych do funkcji danych
	if( !x || !y) return NULL;

	matrix_t *result = (matrix_t *) calloc(1, sizeof(matrix_t));
	if(result == NULL) return NULL;

	(*result).matrix = (double *) calloc( x * y, sizeof(double));
	if( (*result).matrix == NULL)
	{
		free( result);
		return NULL;
	}
	
	//ustawiam pola w strukturze
	(*result).x = x;
	(*result).y = y;

	//zwracam gotowa strukture
	return result;
}


void
matrix_display(const matrix_t a)
{
	for(unsigned i = 0; i < (a.x * a.y); ++i)
	{
		if(i && !(i % a.x)) printf("\n");
		printf("%.5lf ", a.matrix[i]);
	}
	printf("\n");
}


void
matrix_free(matrix_t *a)
{
	if(a == NULL) return;
	if((*a).matrix != NULL) free((*a).matrix );
	free(a);
}


int
matrix_fill(matrix_t *a, unsigned N, ...)
{
	if(a == NULL) return 1;
	if((int)N > matrix_size(a)) return 1;

	va_list list;
	va_start(list, N);

	//petla wpisujaca argumenty do macierzy
	for(unsigned i = 0; i < N; ++i)
		(*a).matrix[i] = va_arg(list, double);

	va_end(list);
	return 0;
}


int
matrix_multiply(const matrix_t a, const matrix_t b, matrix_t *result,
	char transposed_a, char transposed_b)
{
//walidacja danych
	if(result == NULL) return 1;

	if(!transposed_a && !transposed_b) {
		if(a.x != b.y) return 2;
		if(result->x != b.x || result->y != a.y) return 2;
	}
	if(!transposed_a &&transposed_b) {
		if(a.x != b.x) return 2;
		if(result->x != b.y || result->y != a.y) return 2;
	}
	if(transposed_a && !transposed_b) {
		if(a.y != b.y) return 2;
		if(result->x != b.x || result->y != a.x) return 2;
	}
	if(transposed_a && transposed_b) {
		if(a.y != b.x) return 2;
		if(result->x != b.y || result->y != a.x) return 2;
	}

	double value = 0, a_value = 0, b_value = 0;
	unsigned aux = transposed_a ? a.y : a.x;

//mnozenie macierzy
	for(unsigned y = 0; y < (*result).y; ++y)
	{
		for(unsigned x = 0; x < (*result).x; ++x)
		{
			for(unsigned g = 0; g < aux; ++g)
			{
				b_value = transposed_b ? b.matrix[g + x*b.x]
							:b.matrix[x + g*b.x];
				
				a_value = transposed_a ? a.matrix[y + g*a.x]
							:a.matrix[g + y*a.x];
				
				value += a_value * b_value;
			}
			(*result).matrix[x + y * (*result).x] = value;
			value = 0;
		}
	}
	return 0;
}


int
matrix_hadamard(matrix_t a, matrix_t b, matrix_t *result)
{
//walidacja danych
	if(result == NULL) return 1;
	if(a.x != b.x || a.y != b.y) return 2;
	if(result->x != b.x || result->y != b.y) return 2;

//mnozenie macierzy
	for(int i = 0; i < matrix_size(result); ++i)
		result->matrix[i] = a.matrix[i] * b.matrix[i];
	return 0;
}


void
matrix_fill_rng(matrix_t * a, double min, double max)
{
	double f;
 	for(int i = 0; i < matrix_size(a); ++i)
	{
		f = (double) rand() / RAND_MAX;
		(*a).matrix[i] = min + f * (max - min);
	}
}


int
matrix_substraction(const matrix_t a, const matrix_t b, matrix_t *result)
{
	if(result == NULL) return 1;
//sprawdzam czy wszystkie macierze maja takie same wymiary
	if(a.x != b.x || a.y != b.y) return 1;
	if(a.x != result->x || a.y != result->y) return 1;

//odejmowanie
 	for(int i = 0; i < matrix_size(&a); ++i)
 		(*result).matrix[i] = a.matrix[i] - b.matrix[i];            

	return 0;
}


int
matrix_multiply_by_num(matrix_t *a, const double b)
{
	if(a == NULL) return 1;

	for(unsigned i = 0; i < (a->x) * (a->y); ++i)
		a->matrix[i] = a->matrix[i] * b;

	return 0;
}


int
outer_product(const matrix_t a, const matrix_t b, matrix_t *result)
{
//sprawdzam dane wejsciowe
	if(result == NULL) return 1;
	if(result->x != b.x || result->y != a.x) return 1;

	for(unsigned g = 0; g < a.x; ++g)
		for(unsigned i = 0; i < b.x; ++i)
			result->matrix[i + b.x * g] =  b.matrix[i] * a.matrix[g];

	return 0;
}

int
matrix_compare_max_value_index(const matrix_t* a, const matrix_t* b)
{
	if(a == NULL || b == NULL) return -1;
	if(a->y != b->y || a->x != b->x) return -1;
	int counter = 0, max_a = 0, max_b = 0;
	for(unsigned y = 0; y < a->y; ++y)
	{
		for(unsigned x = 0; x < a->x; ++x)
			if(a->matrix[x + y*a->x] > a->matrix[max_a + y*a->x])
				max_a = x;

		for(unsigned x = 0; x < a->x; ++x)
			if(b->matrix[x + y*a->x] > b->matrix[max_b + y*a->x])
				max_b = x;

		if(max_a != max_b) ++counter;
	}
 	return counter;       
}

int
matrix_size(const matrix_t* a) { return (a == NULL) ? (-1) : ((a->x) * (a->y)); }

int
matrix_resize(matrix_t* a, unsigned new_x, unsigned new_y)
{
	if(a == NULL) return 1;
	if((int)(new_y * new_x) != matrix_size(a)) return 1;
	a->x = new_x; a->y = new_y; return 0;
}

//======= FUNKCJE DO LISTY MACIERZY =======


struct matrix_array *
matrix_array_create(void)
{
	struct matrix_array * a = (struct matrix_array *)
				calloc(1, sizeof(struct matrix_array));
	
	if(a == NULL) return NULL;

	a->tail = NULL; a->head = NULL;
	return a;
}


struct matrix_node *
matrix_node_create(unsigned x, unsigned y)
{
//alokacja pamieci na komorke listy z macierza
	struct matrix_node *
        	result = (struct matrix_node *)
			calloc(1, sizeof(struct matrix_node));

	if(result == NULL) return NULL;

//alokacja pamieci strukture macierzy
	(*result).matrix = matrix_alloc(x, y);
	if( (*result).matrix == NULL ) { free(result); return NULL; }

	result->next = NULL;
	result->prev = NULL;

 	return result;
}


int
matrix_array_append(struct matrix_array * array, unsigned x, unsigned y)
{
//sprawdzam dane wejsciowe
	if(array == NULL) return 1;
	if(!x || !y) return 1;

//alokuje pamiec na nowy element listy
	struct matrix_node *node = matrix_node_create(x, y);
	if(node == NULL) return 1;

//ustawiam odpowiednio zmienne w nowym elemencie
	if(array->head == NULL) array->head = node;
	if(array->tail == NULL) array->tail = node;
	else
	{
		array->tail->next = node;
		node->prev = array->tail;
		array->tail = node;
	}
	return 0;
}

int
matrix_array_append_front(struct matrix_array * array, unsigned x, unsigned y)
{
//sprawdzam dane wejsciowe
	if(array == NULL) return 1;
	if(!x || !y) return 1;

//alokuje pamiec na nowy element listy
	struct matrix_node *node = matrix_node_create(x, y);
	if(node == NULL) return 1;

//ustawiam odpowiednio zmienne w nowym elemencie
	if(array->tail == NULL) array->tail = node;
	if(array->head == NULL) array->head = node;
	else
	{
		array->head->prev = node;
		node->next = array->head;
		array->head = node;
	}
	return 0;
}


void
matrix_array_free(struct matrix_array *array)
{
 	if(array == NULL) return;
	if(array->tail == NULL) {free(array); return;}
	if(array->head == NULL) return;

	struct matrix_node *ptr = array->head;
	struct matrix_node *aux = array->head;
	do {
		ptr = aux->next;

		if(aux->matrix != NULL) matrix_free(aux->matrix);
		free(aux);

		aux = ptr;
	} while(ptr != NULL);
	
	free(array);
}

//======= GLOWNIE DO DEBUGU - USUNAC W KONCOWEJ WERSJI =======

void
matrix_array_display(const struct matrix_array* array)
{
	if(array == NULL) return;
	struct matrix_node *ptr = array->head;
	while(ptr != NULL)
	{
		matrix_display(*ptr->matrix);
		if(ptr == array->head) printf("(HEAD)\n");
		if(ptr == array->tail) printf("(TAIL)\n");
		else printf("|\n|\nV\n");
		ptr = ptr->next;
	}
}
