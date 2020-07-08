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
matrix_fill_rng(matrix_t * a, int min, int max)
{
 	for(unsigned i = 0; i < (a->x) * (a->y); ++i)
		(*a).matrix[i] = (rand() - min) % max;
}
