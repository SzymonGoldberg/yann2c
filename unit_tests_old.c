#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include "matrix.h"
#include "deep_network.h"

int main (void)
{
//===== random seed =====

 	srand(time(NULL));

//===== TESTY DOT. ALOKACJI =====
	printf("TEST 1 (alokacja)\n");
	matrix_t *a = matrix_alloc(3, 2);
	if(a != NULL) { printf("=== OK ===\n"); }
	else
	{
		printf("Funkcja powinna zwrocic wskaznik na zaalokowana pamiec a zwrocila NULL\n");
		return 1;
	}

	printf("TEST 2 (alokacja)\n");
	matrix_t *b = matrix_alloc(2, 3);
	if(b != NULL) { printf("=== OK ===\n"); }
	else
	{
		printf("Funkcja powinna zwrocic wskaznik na zaalokowana pamiec a zwrocila NULL\n");
		return 1;
	}

	printf("TEST 3 (alokacja)\n");
	matrix_t *c = matrix_alloc(2, 2);
	if(c != NULL) { printf("=== OK ===\n"); }
	else
	{
		printf("Funkcja powinna zwrocic wskaznik na zaalokowana pamiec a zwrocila NULL\n");
		return 1;
	}


//===== TEST DOT. ZAPELNIANIA MACIERZY =====

	printf("TEST 4 (zapelnianie)\n");
	double tab[6] = {1.0, 0.0, 2.0,
			-1.0, 3.0, 1.0};

        matrix_fill(a, 6,	tab[0], tab[1], tab[2],
				tab[3], tab[4], tab[5]);

	int err = 0;

	for(int i = 0; i < 6; ++i)
		if( (*a).matrix[i] != tab[i]) ++err;

	if(err)
	{
		printf("Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	printf("=== OK! ===\n");
	
	printf("TEST 5 (zapelnianie)\n");
	double tab2[6] =       {3.0, 1.0,
				2.0, 1.0,
				1.0, 0.0};

        matrix_fill(b, 6,	tab2[0], tab2[1],
				tab2[2], tab2[3],
				tab2[4], tab2[5]);

	err = 0;
	for(int i = 0; i < 6; ++i)
		if( (*b).matrix[i] != tab2[i]) ++err;

	if(err)
	{
		printf("Funkcja zapelnila nieprawidlowo %i komorek macierzy\n", err);
		return 1;
	}
	printf("=== OK! ===\n");
	


//===== TESTY DOT. MNOZENIA =====

	printf("TEST 6 (mnozenie)\n");

	int aux = matrix_multiply(*a, *b, c, 0);

        if(aux) printf("--- funkcja powinna zwrocic 0 a zwrocila %i ---\n", aux); 
	
	if((*c).matrix[0] != 5 || (*c).matrix[1] != 1 || (*c).matrix[2] != 4 ||
		(*c).matrix[3] != 2)
	{
        	printf("--- blad w mnozeniu - macierz c powinna miec nastepujace wartosci:");
		printf("5 1\n4 2\na ma:\n");
		matrix_display(*c);
		return 1;
	}
	printf("=== OK! ===\n");

	printf("TEST 7 (mnozenie macierzy transponowanej)\n");

	aux = matrix_multiply(*c, *b, a, 1);

        if(aux) printf("--- funkcja powinna zwrocic 0 a zwrocila %i ---\n", aux); 
	
	if((int)(*a).matrix[0] != 16 || (int)(*a).matrix[1] != 11 ||
	(int)(*a).matrix[2] != 5 || (int)(*a).matrix[3] != 14 ||
	(int)(*a).matrix[4] != 10 || (int)(*a).matrix[5] != 4)
	{
        	printf("--- blad w mnozeniu - macierz c powinna miec nastepujace wartosci:");
		printf("\n16 11 5\n14 10 4\na ma:\n");
		matrix_display(*a);
		return 1;
	}
	printf("=== OK! ===\n");

//===== TESTY DOT. OBLICZANIA ODP SIECI =====

	printf("TEST 8 (siec neuronowa - 1 warstwa, 4 serie)\n");

	matrix_t* output = neural_network(*c, *b);
	if(output == NULL)
	{
		printf("Funkcja powinna zwrocic wskaznik na strukture a zwrocila NULL\n");
		return 1;
	}

	if((int)(*output).matrix[0] != 16 || (int)(*output).matrix[1] != 11 ||
	(int)(*output).matrix[2] != 5 || (int)(*output).matrix[3] != 14 ||
	(int)(*output).matrix[4] != 10 || (int)(*output).matrix[5] != 4)
	{
        	printf("--- blad w mnozeniu - macierz c powinna miec nastepujace wartosci:");
		printf("\n16 11 5\n14 10 4\na ma:\n");
		matrix_display(*output);
		return 1;
	}
	printf("=== OK! ===\n");


//===== TESTY DOT. ZWALNIANIA PAMIECI =====
	
	printf("TEST 9 (zwalnianie)\n");

 	matrix_free(output);
	printf("=== OK! ===\n");
	
	printf("TEST 10 (zwalnianie)\n");

 	matrix_free(a);
	printf("=== OK! ===\n");

	printf("TEST 11 (zwalnianie)\n");

 	matrix_free(b);
	printf("=== OK! ===\n");
	
	printf("TEST 12 (zwalnianie)\n");

 	matrix_free(c);
	printf("=== OK! ===\n");


//=== TESTY SIECI GLEBOKICH ===

	printf("TEST 13 (gleboka siec 3x3)\n");

	matrix_t* input = matrix_alloc(3, 4);

	input->matrix[0] = 8.5; input->matrix[1] = 0.65; input->matrix[2] = 1.2;
	input->matrix[3] = 9.5; input->matrix[4] = 0.8;  input->matrix[5] = 1.3;
	input->matrix[6] = 9.9; input->matrix[7] = 0.8;	 input->matrix[8] = 0.5;
	input->matrix[9] = 9.0; input->matrix[10] = 0.9; input->matrix[11] = 1.0;

	struct matrix_array *neural_network = matrix_array_create();


	matrix_array_append(neural_network, 3, 3);

	matrix_fill(neural_network->tail->matrix, 9,	0.1, 0.2, -0.1,
							-0.1, 0.1, 0.9,
       		 					0.1, 0.4, 0.1);

	aux = matrix_array_append_network(neural_network, 3, 0, 0, 0);

	matrix_fill(neural_network->tail->matrix, 9,	0.3, 1.1, -0.3,
							0.1, 0.2, 0.0,
	       		 				0.0, 1.3, 0.1);

	if(aux) printf("---funkcja powinna zwrocic 0 a zwrocila %i\n", aux);

	matrix_t* output1 = deep_neural_network(*input, *neural_network);

	double exp_tab[] = {	0.214,	0.145,	0.507,
				0.204,	0.158,	0.53,
				-0.584, 0.018,	-0.462,
       		 		-0.015, 0.116,	0.253};

	err = 0;
	for(int i = 0; i < 12; ++i)
	{
		if( output1->matrix[i] > exp_tab[i] + 0.001 ||
		output1->matrix[i] < exp_tab[i] - 0.001)
		{
			printf("---funkcja deep_neural_network zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_tab[i], output1->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_array_free(neural_network);
	matrix_free(input);
	matrix_free(output1);

//===== TESTY DOT. ODEJMOWANIA MACIERZY =====

	printf("TEST 14 (odejmowanie macierzy)\n");

	a = matrix_alloc(3, 2);
	matrix_fill(a, 6,	9.0, 1.0, -4.0,
				3.0, -5.0, 0.0);

	b = matrix_alloc(3, 2);
	matrix_fill(b, 6,	6.0, -4.0, 5.0,
				7.0, -2.0, 3.0);

	c = matrix_alloc(3, 2);
	aux = matrix_substraction(*a, *b, c);

	if(aux) printf("-funkcja zwrocila %i a powinna 0\n", aux);

	double exp_subs_tab[] = {3, 5, -9,
				-4, -3, -3};

	err = 0;
	for(int i = 0; i < 6; ++i)
	{
		if( c->matrix[i] > exp_subs_tab[i] + 0.001 ||
			c->matrix[i] < exp_subs_tab[i] - 0.001)
		{
			printf("---funkcja matrix_substraction zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_subs_tab[i], c->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);

//===== TESTY DOT. MNOZENIA MACIERZY PRZEZ LICZBE =====

	printf("TEST 15 (mnozenie macierzy przez liczbe)\n");

	a = matrix_alloc(2, 2);
	matrix_fill(a, 4,	1.0, 3.0,
				8.0, 6.0);

	b = matrix_alloc(2, 2);
        aux = matrix_multiply_by_num(*a, (double) 5, b);

	if(aux) printf("-funkcja zwrocila %i a powinna 0\n", aux);

	double exp_mult_by_num[] = { 5.0, 15.0, 40.0, 30.0 };

	err = 0;
	for(int i = 0; i < 4; ++i)
	{
		if( b->matrix[i] > exp_mult_by_num[i] + 0.001 ||
			b->matrix[i] < exp_mult_by_num[i] - 0.001)
		{
			printf("---funkcja matrix_substraction zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_mult_by_num[i], b->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);


//===== TEST OUTER PRODUCT =====

	printf("TEST 16 (iloczyn zewnetrzny)\n");

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
			printf("---funkcja matrix_substraction zle obliczyla pole %i\n------powinno byc %lf a jest %lf\n", i, exp_outer[i], c->matrix[i]);
			++err;
		}
	}
	if(err) printf("-funkcja zle wypelnila %i komorek macierzy\n", err);
	else	printf("=== OK! ===\n");

	matrix_free(a);
	matrix_free(b);
	matrix_free(c);

	return 0;
}
