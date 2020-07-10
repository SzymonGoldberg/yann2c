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

//dwustronna lista macierzy
struct matrix_node {
 	matrix_t *matrix;
	struct matrix_node *next;
	struct matrix_node *prev;
};

struct matrix_array {
 	struct matrix_node *head;
	struct matrix_node *tail;
};

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
int matrix_multiply(const matrix_t a, const matrix_t b,
			matrix_t *result, char transposed);

//wypelnianie macierzy randomowymi wartosciami mieszczacymi sie w przedziale
//min < x < max
void
matrix_fill_rng(matrix_t * a, double min, double max);

//odejmowanie dwoch macierzy, wynik jest wpisywany do <result>, w przypadku
//sukcesu zwraca 0, w przeciwnym wypadku 1
int
matrix_substraction(const matrix_t a, const matrix_t b, matrix_t *result);

//mnozy macierz <a> przez wartosc liczbowa <b> i zapisuje wynik w macierzy
//<result>, w przypadku sukcesu zwraca 0, w innym - 1
int
matrix_multiply_by_num(const matrix_t a, const double b, matrix_t *result);


//zwykly append listy, dodaje nowy element z macierza o wielkosci <x,y>
//w przypadku sukcesu 0, w przypadku porazki 1
int
matrix_array_append(struct matrix_array * array, unsigned x, unsigned y);

//apend zrobiony pod sieci neuronowe
//funkcja dodaje do listy macierzy <array>, nastepna macierz reprezentujaca
//<N> neuronow, jesli flaga <random_weight_flag> jest ustawiona funkcja wypelnia
//nowa warstwe randomowymi wartosciami od <weight_min_value> do <weight_max_value>
//odpowiednik <add_layer> w pdfie PSI
int
matrix_array_append_network(struct matrix_array * array, unsigned int n,
char random_weight_flag, double weight_min_value, double weight_max_value);

//zwalnia pamiec przydzielona na pojedynczy element listy macierzy
void
matrix_node_free(struct matrix_node *array);

//tworzy pojedynczy element listy z macierzami, macierz ma wymiary x,y
//zwraca adres na nowa liste w przypadku sukcesu, w przypadku porazki nulla
struct matrix_node *
matrix_node_create(unsigned x, unsigned y);

//zwalnia pamiec przydzielona na kazdy pojedyczny element listy oraz na sama liste
void
matrix_array_free(struct matrix_array *array);

//tworzy pusta strukture matrix_array
struct matrix_array *
matrix_array_create(void);

void
matrix_array_display(const struct matrix_array* array);

#endif
