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
	
	int aux = matrix_multiply(*a, *b, c, 0, 0);
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


	puts("TEST 7 ---matrix_multiply---(mnozenie przez macierz transponowana)");
	
	aux = matrix_multiply(*c, *b, a, 0, 1);
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


	puts("TEST 8 ---matrix_multiply_by_num---");

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

//====== TESTY ODEJMOWANIA MACIERZY =======


	puts("TEST 9 ---matrix_substraction---");

	aux = matrix_substraction(*a, *b, NULL);
	if(!aux) printf("---Funkcja powinna zwrocic 1 a zwrocila %i\n", aux);
	else puts("=== OK! ===");


	puts("TEST 10 ---matrix_substraction---");

	matrix_t *d = matrix_alloc(2, 3);
	matrix_t *e = matrix_alloc(2, 3);

	aux = matrix_substraction(*a, *d, e);
	if(!aux) printf("---Funkcja powinna zwrocic 1 a zwrocila %i\n", aux);
	else puts("=== OK! ===");


	puts("TEST 11 ---matrix_substraction---");

	matrix_fill(d, 6,	3.0,	8.0,
				1.0,	-13.0,
				2.5,	-4.12);

	aux = matrix_substraction(*b, *d, e);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_subs0[] =    {12.0,	-3.0,
				9.0,	18.0,
				2.5,	4.12};
	err = 0;
 	for(int i = 0; i < 6; ++i)
	{
		if(	e->matrix[i] > exp_subs0[i] + 0.001 ||
	       		e->matrix[i] < exp_subs0[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_subs0[i], e->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

//======= TESTY SIECI =======


	puts("TEST 12 ---nn_create---");

	struct nn_array *nn = nn_create();
	if(nn == NULL)
	{
		puts("---Funkcja powinna zwrocic adres a zwrocila NULL");
		return 1;
	}
	else puts("=== OK! ===");


	puts("TEST 13 ---nn_add_layer---");

	aux = nn_add_layer(NULL, 3, 3, 1, NULL, 0.0);
	if(!aux) printf("---Funkcja powinna zwrocic 1 a zwrocila %i\n", aux);
	else puts("=== OK! ===");


	puts("TEST 14 ---nn_add_layer---");

	aux = nn_add_layer(nn, 0, 3, 1, NULL, 0);
	if(!aux) printf("---Funkcja powinna zwrocic 1 a zwrocila %i\n", aux);
	else puts("=== OK! ===");

	puts("TEST 15 ---nn_add_layer---");

	aux = nn_add_layer(nn, 3, 0, 1, NULL, 0);
	if(!aux) printf("---Funkcja powinna zwrocic 1 a zwrocila %i\n", aux);
	else puts("=== OK! ===");


	puts("TEST 16 ---nn_add_layer---");

	aux = nn_add_layer(nn, 3, 3, 1, NULL, 0.0);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);
	else puts("=== OK! ===");


	puts("TEST 17 ---nn_predict---");

	matrix_fill(nn->tail->weights, 9,	0.1, 0.1,-0.3,
						0.1, 0.2, 0.0,
				       		0.0, 1.3, 0.1);

 	matrix_t *input0 = matrix_alloc(3, 1);
	matrix_fill(input0, 3, 8.5, 0.65, 1.2);

	nn_predict(nn, input0, 0);
	double exp_predict0[] = {0.555, 0.98, 0.965};

	err = 0;
 	for(int i = 0; i < 3; ++i)
	{
		if(nn->tail->output->matrix[i] > exp_predict0[i] + 0.001 ||
	       	   nn->tail->output->matrix[i] < exp_predict0[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict0[i], nn->tail->output->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");


	puts("TEST 18 ---nn_predict---");
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
				       		0.1, 0.4, 0.1);
	aux = nn_add_layer(nn, 3, 3, 1, NULL, 0.0);

	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					      	0.1, 0.2, 0.0,
				       		0.0, 1.3, 0.1);
	nn_predict(nn, input0, 0);

	double exp_predict1[] = {0.2135, 0.145, 0.5065};
	err = 0;
 	for(int i = 0; i < 3; ++i)
	{
		if(nn->tail->output->matrix[i] > exp_predict1[i] + 0.001 ||
	       	   nn->tail->output->matrix[i] < exp_predict1[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict1[i], nn->tail->output->matrix[i]);
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
	matrix_free(d);
	matrix_free(e);
	matrix_free(input0);

	puts("TEST 19 ---outer_product---");

	a = matrix_alloc(3, 1);
	matrix_fill(a, 3,	0.455, -0.02, 0.865);

	b = matrix_alloc(3, 1);
	matrix_fill(b, 3,	8.5, 0.65, 1.2);

	c = matrix_alloc(3, 3);

	outer_product(*a, *b, c);

	double exp_outer[] =	{3.8675, 0.29575, 0.546,
				-0.17, -0.013, -0.024,
				7.3525, 0.56225, 1.038};

	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( c->matrix[i] > exp_outer[i] + 0.001 ||
			c->matrix[i] < exp_outer[i] - 0.001)
		{
			printf("-funkcja zle obliczyla pole %i\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_outer[i], c->matrix[i]);
			++err;
		}
	}
	if(err) printf("---funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);
	nn_free(nn);


	puts("TEST 20 ---nn_backpropagation---");

	nn = nn_create();

	nn_add_layer(nn, 1, 1, 1, NULL, 0.0);
	matrix_fill(nn->tail->weights, 1, 0.1);

	nn_add_layer(nn, 1, 1, 1, NULL, 0.0);
	matrix_fill(nn->tail->weights, 1, 0.3);

	a = matrix_alloc(1, 1);
	matrix_fill(a, 1, 8.5);

	nn_predict(nn, a, 0);

	b = matrix_alloc(1, 1);
	matrix_fill(b, 1, 0.1);

	aux = nn_backpropagation(nn, a, b, 0.01, 0, 0);
	
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_weights0[] = {0.2986825, 0.0960475};
	struct nn_layer *nn_ptr = nn->tail;

	err = 0;
	for(int i = 0; nn_ptr != NULL; ++i)
	{
		if( nn_ptr->weights->matrix[0] > exp_weights0[i] + 0.001 ||
			nn_ptr->weights->matrix[0] < exp_weights0[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights0[i],
				nn_ptr->weights->matrix[0]);
			err++;
		}
		nn_ptr = nn_ptr->prev;
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);


	puts("TEST 21 ---matrix_hadamard---");


	a = matrix_alloc(3, 1);
	b = matrix_alloc(2, 1);
	c = matrix_alloc(3, 1);

        aux = matrix_hadamard(*a, *b, c);

	if(!aux) puts("---Funkcja powinna zwrocic 1 a zwrocila 0");
	else	puts("=== OK! ===");
	
	matrix_free(b);


	puts("TEST 22 ---matrix_hadamard---");

	b = matrix_alloc(3, 1);

	matrix_fill(a, 3, 1.0, 2.0, 3.0);
	matrix_fill(b, 3, 5.0, 5.0, 5.0);

	aux = matrix_hadamard(*a, *b, c);
	if(aux) printf("---Funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	double exp_multiply3[] = {5.0, 10.0, 15.0};

	err = 0;
 	for(int i = 0; i < 3; ++i)
	{
		if(	c->matrix[i] > exp_multiply3[i] + 0.001 ||
	       		c->matrix[i] < exp_multiply3[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_multiply3[i], c->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);
	nn_free(nn);

	puts("TEST 23 ---nn_backpropagation---(z funkcja aktywacji)");

	nn = nn_create();

	nn_add_layer(nn, 3, 1, 1, (ReLU), 0.0);
	matrix_fill(nn->tail->weights, 3, 0.1, -0.1, 0.1);

	nn_add_layer(nn, 1, 0, 1, NULL, 0.0);
	matrix_fill(nn->tail->weights, 3, 0.3, 1.1, -0.3);

	a = matrix_alloc(1, 1);
	a->matrix[0] = 8.5;

	b = matrix_alloc(1, 1);
	b->matrix[0] = 0.1;

	nn_backpropagation(nn, a, b, 0.01, 0, 0);

	double exp_weights1[] = {0.30085, 1.1, -0.29915};

	err = 0;
	for(int i = 0; i < 3; ++i)
	{
		if( nn->tail->weights->matrix[i] > exp_weights1[i] + 0.001 ||
			nn->tail->weights->matrix[i] < exp_weights1[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights1[i],
				nn->tail->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	double exp_weights2[] = {0.10255, -0.1, 0.09745};

	err = 0;
	for(int i = 0; i < 3; ++i)
	{
		if( nn->head->weights->matrix[i] > exp_weights2[i] + 0.001 ||
			nn->head->weights->matrix[i] < exp_weights2[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights2[i],
				nn->head->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	nn_free(nn);
	matrix_free(a);
	matrix_free(b);

	puts("TEST 24 ---nn_backpropagation---(z funkcja aktywacji)");

	nn = nn_create();

	nn_add_layer(nn, 3, 3, 1, (ReLU), 0.0);
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
						0.1, 0.4, 0.1);
	nn_add_layer(nn, 3, 3, 1, NULL, 0.0);
	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					       	0.1, 0.2, 0.0,
						0.0, 1.3, 0.1);
	a = matrix_alloc(3, 1);
	matrix_fill(a, 3, 8.5, 0.65, 1.2);

	b = matrix_alloc(3, 1);
	matrix_fill(b, 3, 0.1, 1.0, 0.1);

	nn_backpropagation(nn, a, b, 0.01, 0, 0);
	
	double exp_weights3[] = {0.299024, 1.09967, -0.301396,
				0.107353, 0.202522, 0.0105165,
				-0.0034959, 1.2988, 0.0950001};

	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( nn->tail->weights->matrix[i] > exp_weights3[i] + 0.001 ||
			nn->tail->weights->matrix[i] < exp_weights3[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights3[i],
				nn->tail->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	double exp_weights4[] = {0.104373, 0.200334, -0.0993826,
				-0.140996, 0.0968651, 0.894212,
				0.099439, 0.399957, 0.0999208};

	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( nn->head->weights->matrix[i] > exp_weights4[i] + 0.001 ||
			nn->head->weights->matrix[i] < exp_weights4[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights4[i],
				nn->head->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	nn_free(nn);

	puts("TEST 24 ---nn_predict---(dla batcha, z funkcja aktywacji)");

	nn = nn_create();

	nn_add_layer(nn, 3, 3, 4, (ReLU), 0.0);
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
						0.1, 0.4, 0.1);
	nn_add_layer(nn, 3, 3, 4, NULL, 0.0);
	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					       	0.1, 0.2, 0.0,
						0.0, 1.3, 0.1);
	a = matrix_alloc(3, 4);
	matrix_fill(a, 12,	8.5, 0.65, 1.2,
				9.5, 0.8, 1.3,
				9.9, 0.8, 0.5,
				9.0, 0.9, 1.0);

	double exp_predict2[] = {0.86, 0.295, 1.23,
				0.98, 0.3, 1.4,
				1.1, 0.0, 1.36,
				0.98, 0.0899999, 1.36};
	nn_predict(nn, a, 0);

	err = 0;
 	for(int i = 0; i < 12; ++i)
	{
		if(nn->head->output->matrix[i] > exp_predict2[i] + 0.001 ||
	       	   nn->head->output->matrix[i] < exp_predict2[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict2[i], nn->head->output->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	double exp_predict3[] = {0.2135, 0.145, 0.5056,
				0.204, 0.158, 0.53, 
				-0.078, 0.11, 0.136,
				-0.0150001, 0.116, 0.253};

	err = 0;
 	for(int i = 0; i < 12; ++i)
	{
		if(nn->tail->output->matrix[i] > exp_predict3[i] + 0.001 ||
	       	   nn->tail->output->matrix[i] < exp_predict3[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict3[i], nn->tail->output->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	matrix_free(a);
	nn_free(nn);

	puts("TEST 25 ---matrix_multiply---(obie transponowane)");

	a = matrix_alloc(3, 2);
	b = matrix_alloc(2, 3);
	c = matrix_alloc(3, 3);

	matrix_fill(a, 6,	1.0, 2.0, 3.0,
				4.0, 5.0, 6.0);

	matrix_fill(b, 6,	1.0, 24.0,
				3.0, 6.0,
				8.0, 12.0);

	aux = matrix_multiply(*a, *b, c, 1, 1);
	if(aux)	printf("Powinna zwrocic 0 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	double exp_multiply4[] = {	97, 27, 56,
					122, 36, 76, 
					147, 45, 96};

	
	err = 0;
 	for(int i = 0; i < 9; ++i)
	{
		if(	c->matrix[i] > exp_multiply4[i] + 0.001 ||
	       		c->matrix[i] < exp_multiply4[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
				exp_multiply4[i], c->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}

	puts("=== OK! ===");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);

	puts("TEST 26 ---nn_backpropagation---(batch, funkcja aktywacji");

	nn = nn_create();

	nn_add_layer(nn, 3, 3, 4, (ReLU), 0.0);
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
						0.1, 0.4, 0.1);
	nn_add_layer(nn, 3, 3, 4, NULL, 0.0);
	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					       	0.1, 0.2, 0.0,
						0.0, 1.3, 0.1);
	a = matrix_alloc(3, 4);
	matrix_fill(a, 12,	8.5, 0.65, 1.2,
				9.5, 0.8, 1.3,
				9.9, 0.8, 0.5,
				9.0, 0.9, 1.0);
	b = matrix_alloc(3, 4);
	matrix_fill(b, 12,	0.1, 1.0, 0.1,
				0.0, 1.0, 0.0,
				0.0, 0.0, 0.1,
				0.1, 1.0, 0.2);

	nn_backpropagation(nn, a, b, 0.01, 0, 0);

	double exp_weights5[] = {0.29901, 1.09916, -0.301627,
				0.123058, 0.205844, 0.0328309,
				-0.0096053, 1.29716, 0.0863697};

	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( nn->tail->weights->matrix[i] > exp_weights5[i] + 0.001 ||
			nn->tail->weights->matrix[i] < exp_weights5[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights5[i],
				nn->tail->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	double exp_weights6[] = {0.118847, 0.201724, -0.0977926,
				-0.190674, 0.0930147, 0.886871,
				0.093963, 0.399449, 0.0994944};
	err = 0;
	for(int i = 0; i < 9; ++i)
	{
		if( nn->head->weights->matrix[i] > exp_weights6[i] + 0.001 ||
			nn->head->weights->matrix[i] < exp_weights6[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights6[i],
				nn->head->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	nn_free(nn);

	puts("TEST 27 ---nn_predict---(batch o innym rozmiarze, funkcja aktywacji");

	nn = nn_create();

	nn_add_layer(nn, 3, 3, 4, (ReLU), 0.0);
	matrix_fill(nn->tail->weights, 9,	0.1, 0.2,-0.1,
					       -0.1, 0.1, 0.9,
						0.1, 0.4, 0.1);
	nn_add_layer(nn, 3, 3, 4, NULL, 0.0);
	matrix_fill(nn->tail->weights, 9,	0.3, 1.1,-0.3,
					       	0.1, 0.2, 0.0,
						0.0, 1.3, 0.1);
	a = matrix_alloc(3, 1);
	matrix_fill(a, 3,	8.5, 0.65, 1.2);

	double exp_predict4[] = {0.86, 0.295, 1.23,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0};
	nn_predict(nn, a, 0);

	err = 0;
 	for(int i = 0; i < 12; ++i)
	{
		if(nn->head->output->matrix[i] > exp_predict4[i] + 0.001 ||
	       	   nn->head->output->matrix[i] < exp_predict4[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict4[i], nn->head->output->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	double exp_predict5[] = {0.2135, 0.145, 0.5056,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0};

	err = 0;
 	for(int i = 0; i < 12; ++i)
	{
		if(nn->tail->output->matrix[i] > exp_predict5[i] + 0.001 ||
	       	   nn->tail->output->matrix[i] < exp_predict5[i] - 0.001)
		{
                	printf("-Funkcja zle wypelnila %i komorke macierzy\n", i);
			printf("--powinno byc %lf a jest %lf\n",
			exp_predict5[i], nn->tail->output->matrix[i]);
			++err;
		}
	}
	if(err) {
		printf("---Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	puts("=== OK! ===");

	matrix_free(a);
	nn_free(nn);

	puts("TEST 28 ---matrix_size---");

	aux = matrix_size(NULL);
	if(aux != -1)	printf("Powinna zwrocic -1 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	puts("TEST 29 ---matrix_size---");

	a = matrix_alloc(3, 4);
	aux = matrix_size(a);
	if(aux != 12)	printf("Powinna zwrocic 12 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	matrix_free(a);

	puts("TEST 30 ---nn_write---");

	nn = nn_create();
              //	network		size	input	batch	act func	dropout
	nn_add_layer(	nn,		3, 	4, 	1, 	NULL,		0.0);
	nn_add_layer(	nn,		3,	0,	1,	NULL,		0.0);

	double exp_weights7[] = { 	12.0,	2.0,	3.0,
					32.0,	5.0,	8.0,
					85.0,	3.5,	6.7,
					91.0,	9.0,	0.0};
	
	double exp_weights8[] = { 	2.3,	2.0,	3.0,
					2.0,	5.0,	8.0,
					1.0,	9.0,	0.0};
	for(int i = 0; i < 12; ++i)
		nn->head->weights->matrix[i] = exp_weights7[i];

	for(int i = 0; i < 9; ++i)
		nn->tail->weights->matrix[i] = exp_weights8[i];

	aux = nn_write(nn, "test_nn_file.bin");
	if(aux)	printf("Powinna zwrocic 0 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	nn_free(nn);

	puts("TEST 31 ---nn_read---");

	struct nn_array* dn = nn_create();

              //	network		size	input	batch	act func	dropout
	nn_add_layer(	dn,		3, 	4, 	1, 	NULL,		0.0);
	nn_add_layer(	dn,		3,	0,	0,	NULL,		0.0);

	aux = nn_read(dn, "test_nn_file.bin");
	if(aux)	printf("Powinna zwrocic 0 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	err = 0;
	for(int i = 0; i < 12; ++i)
	{
		if(	dn->head->weights->matrix[i] > exp_weights7[i] + 0.001 ||
			dn->head->weights->matrix[i] < exp_weights7[i] - 0.001)
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",exp_weights7[i],
				dn->head->weights->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i wag\n", err);
	else	printf("=== OK! ===\n");

	nn_free(dn);

	puts("TEST 32 ---matrix_resize---");

	a = matrix_alloc(2, 3);
	aux = matrix_resize(a, 3, 3);
	if(aux != 1)	printf("Powinna zwrocic 1 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	puts("TEST 33 ---matrix_resize---");

	double tab2[] = {1, 5, 0.3,
			 3, 4, 5.4};

	for(int i = 0; i < 6; ++i)
		a->matrix[i] = tab2[i];

	aux = matrix_resize(a, 3, 2);
	if(aux)		printf("Powinna zwrocic 1 a zwrocila %i\n", aux);
	else		puts("=== OK! ===");

	err = 0;
	for(int i = 0; i < 6; ++i)
	{
		if(a->matrix[i] != tab2[i])
		{
			printf("--powinno byc");
			printf("%lf a jest %lf\n",tab2[i], a->matrix[i]);
			err++;
		}
	}
	if(err) printf("---funkcja zle obliczyla %i komorek\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);

	return 0;
}
