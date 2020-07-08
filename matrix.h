#ifndef _MATRIX_H_
#define _MATRIX_H_

#define TRUE 1
#define FALSE 0

//typ opisujacy pojedyncza, prosta macierz
typedef struct
{
	double *matrix; //tablica macierzy w formie tablicy jednowymiarowej
	unsigned int x; //liczba kolumn
	unsigned int y; //liczba wersow
} matrix_t;

//funkcja alokujaca pamiec na macierz o wymiarach x i y, zwraca adres na nowa
//macierz w przypadku sukcesu, NULL w przypadku porazki
matrix_t* matrix_alloc(unsigned x, unsigned y);

//funkcja do wyswietlania macierzy
void matrix_display(const matrix_t a);

//zwalnianie pamieci przydzielonej na strukture
void matrix_free(matrix_t *a);

//funkcja ktora podane N argumentow (double) wpisuje po kolei do macierzy
int matrix_fill(matrix_t *a, unsigned int N, ...);

//mnozenie macierzy a przez b i wpisywanie wyniku do macierzy result
//jesli transposed jest ustawione na 1 macierz b jest macierza transponowana
//w przypadku sukcesu 0, nieprawidlowe dane 1, niemozliwosc przemnozenia 2
int matrix_multiply(const matrix_t a, const matrix_t b, matrix_t *result, char transposed);

//wypelnianie macierzy randomowymi wartosciami mieszczacymi sie w przedziale
//min < x < max
void
matrix_fill_rng(matrix_t * a, int min, int max);

#endif
