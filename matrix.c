#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include "matrix.h"

matrix_t*
matrix_alloc(unsigned x, unsigned y)
{
	//sprawdzam poprawnosc podanych do funkcji danych
	if( !x || !y) return NULL;

	//alokuje pamiec
	matrix_t *result = (matrix_t *) calloc(1, sizeof(matrix_t));
	//jesli alokacja sie nie udala zwracam NULL
	if(result == NULL) return NULL;

	//alokacja na macierz
	(*result).matrix = (double *) calloc( x * y, sizeof(double));
	//jesli alokacja sie nie powiedzie
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
	for(unsigned int i = 0; i < (a.x * a.y); ++i)
	{
		if(i && !(i % a.x)) printf("\n");
		printf("%.3lf ", a.matrix[i]);
	}
	printf("\n");
}

void
matrix_free(matrix_t *a) {
	if(a != NULL) {
		if((*a).matrix != NULL) free((*a).matrix );
		free(a);
	}
}


int
matrix_fill(matrix_t *a, unsigned int N, ...)
{
	//sprawdzam poprawnosc danych podanych do funkcji
	if(a == NULL) return 1;
	if(N > ((*a).x * (*a).y)) return 1;

	//lista
	va_list list;
	va_start(list, N);


	//petla wpisujaca argumenty do macierzy
	for(unsigned i = 0; i < N; ++i)
		(*a).matrix[i] = va_arg(list, double);

	//zamykam liste, zwracam zero w przypadku sukcesu
	va_end(list);
	return 0;
}


int
matrix_multiply(const matrix_t a, const matrix_t b, matrix_t *result,
		char transposed)
{
//walidacja danych
	if(result == NULL) return 1;

//sprawdzam czy mozna obie macierze mnozyc
	if(a.x != b.y && !transposed) return 2;
	if(a.x != b.x && transposed) return 2;


//sprawdzam miejsce w macierzy wynikowej
	if(((*result).x != b.y || (*result).y != a.y) && transposed) return 2;
	if(((*result).x != b.x || (*result).y != a.y) && !transposed) return 2;

//mnozenie macierzy
	double value = 0;
	for(unsigned y = 0; y < (*result).y; ++y)
	{
		for(unsigned x = 0; x < (*result).x; ++x)
		{
			for(unsigned g = 0; g < a.x; ++g)
			{
				value += transposed ?
					a.matrix[g + y*a.x] * b.matrix[g + x*b.x]:
					a.matrix[g + y*a.x] * b.matrix[x + g*b.x];
			}
			(*result).matrix[x + y * (*result).x] = value;
			value = 0;
		}
	}
	return 0;
}


void
matrix_fill_rng(matrix_t * a, double min, double max)
{
	double f;
 	for(unsigned i = 0; i < (a->x) * (a->y); ++i)
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
 	for(unsigned i = 0; i < (a.x) * (a.y); ++i)
 		(*result).matrix[i] = a.matrix[i] - b.matrix[i];            

	return 0;
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
		matrix_fill_rng(array->matrix, weight_min_value, weight_max_value);

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

//funkcja liczy iloczyn zewnetrzny z wektora ktory powstaje z <x_ptr> kolumny
//macierzy <a> i z <y_ptr> wiersza macierzy <b> i wpisuje wynik do <result>
int
matrix_outer_product(matrix_t a, matrix_t b, matrix_t *result, unsigned x_ptr,
	unsigned y_ptr)
{
//sprawdzam dane wejsciowe
 	if(result == NULL) return 1;
	if(result->x != b.x || result->y != a.y) return 1;
	if((a.x - 1) < x_ptr || (b.y - 1) < y_ptr) return 1; 

	for(unsigned i = 0; i < a.y; ++i)
	{
		for(unsigned g = 0; g < b.x; ++g)
		{
			result->matrix[g + i * a.x]
				= b.matrix[g + y_ptr * b.x]
					* a.matrix[x_ptr + i * a.x];
		}
	}

	return 0;
}
