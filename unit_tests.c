#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "matrix.h"
#include "deep_network.h"

int main (void)
{
//======= TESTY FUNKCJI ALOKUJACEJ PAMIEC NA MACIERZ =======
	puts("TEST 1 ---matrix_alloc---");
	matrix_t *a = matrix_alloc(3, 2);
	if(a == NULL)
	{
		puts("---Funkcja powinna zwrocic wskaznik na zaalokowana");
		puts("pamiec a zwrocila NULL\n");
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 2 ---matrix_alloc---");
	matrix_t *b = matrix_alloc(2, 3);
	if(b == NULL)
	{
		puts("---Funkcja powinna zwrocic wskaznik na zaalokowana");
		puts("pamiec a zwrocila NULL\n");
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 3 ---matrix_alloc---");
	matrix_t *c = matrix_alloc(2, 2);
	if(c == NULL)
	{
		puts("---Funkcja powinna zwrocic wskaznik na zaalokowana");
		puts("pamiec a zwrocila NULL\n");
		return 1;
	}
	puts("=== OK! ===");

//======= TESTY FUNKCJI DO ZAPELNIANIA MACIERZY =======

	puts("TEST 4 ---matrix_fill---");
	double tab[] =	{1.0, 0.0, 2.0,
		       	-1.0, 3.0, 1.0};

	matrix_fill(a, 6,	tab[0], tab[1], tab[2],
                          	tab[3], tab[4], tab[5]);

	int err = 0;

	for(int i = 0; i < 6; ++i) {
		if( a->matrix[i] != tab[i]) {
			printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",tab[i], a->matrix[i]);
			err++;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 5 ---matrix_fill---");
	double tab1[] =	       {3.0, 1.0,
			      	2.0, 1.0,
       		 		1.0, 0.0};

	matrix_fill(b, 6,	tab1[0], tab1[1],
				tab1[2], tab1[3],
				tab1[4], tab1[5]);

	err = 0;

	for(int i = 0; i < 6; ++i) {
		if( b->matrix[i] != tab1[i]) {
			printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",tab1[i], b->matrix[i]);
			err++;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

//======= TESTY MNOZENIA MACIERZY =======


	puts("TEST 6 ---matrix_multiply---");
	
	int aux = matrix_multiply(*a, *b, c, 0);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_multiply0[] =       {5.0, 1.0,
					4.0, 2.0};

	err = 0;
 	for(int i = 0; i < 4; ++i)
	{
		if(	c->matrix[i] > exp_multiply0[i] + 0.001 ||
	       		c->matrix[i] < exp_multiply0[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_multiply0[i], c->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 6 ---matrix_multiply---(mnozenie przez macierz transponowana)");
	
	aux = matrix_multiply(*c, *b, a, 1);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_multiply1[] =       {16.0,	11.0,
					5.0,	14.0,
					10.0,	4.0};
	err = 0;
 	for(int i = 0; i < 6; ++i)
	{
		if(	a->matrix[i] > exp_multiply1[i] + 0.001 ||
	       		a->matrix[i] < exp_multiply1[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_multiply1[i], a->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 7 ---matrix_multiply_by_num---");

	aux = matrix_multiply_by_num(b, 5);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_multiply2[] =       {15.0,	5.0,
					10.0,	5.0,
					5.0,	0.0};
	err = 0;
 	for(int i = 0; i < 6; ++i)
	{
		if(	b->matrix[i] > exp_multiply2[i] + 0.001 ||
	       		b->matrix[i] < exp_multiply2[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_multiply2[i], b->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	
//=== zwalnianie pamieci przydzielonej na macierzy ===

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);

	return 0;
}
