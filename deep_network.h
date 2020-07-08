#ifndef _DEEP_NETWORK_H_
#define _DEEP_NETWORK_H_

//dwustronna lista macierzy
struct matrix_dl_array {
 	matrix_t *matrix;
	struct matrix_dl_array *next;
	struct matrix_dl_array *prev;
};

//funkcja liczy macierz wyjsciowa mnozac wejscie przez transponowana macierz wag
//alokuje pamiec na macierz wyjsciowa i zwraca na nia wskaznik w przypadku sukcesu
//w innym przypadku zwraca NULL
matrix_t * neural_network(const matrix_t weights, const matrix_t input);

//zwraca odpowiedz sieci neuronowej - w przypadku porazki NULL,
//dane do niej podajemy w formie listy dwukierunkowej z macierzami (network)
//wejscie podajemy jako pojedyncza macierz
//TODO
//>optymalizacja
matrix_t *
deep_neural_network(matrix_t input, struct matrix_dl_array network);

//funkcja dodaje do listy macierzy <array>, nastepna macierz reprezentujaca
//<N> neuronow, jesli flaga <random_weight_flag> jest ustawiona funkcja wypelnia
//nowa warstwe randomowymi wartosciami od <weight_min_value> do <weight_max_value>
//odpowiednik <add_layer> w pdfie PSI
int
matrix_dll_array_append(struct matrix_dl_array * array, unsigned int n,
char random_weight_flag, double weight_min_value, double weight_max_value);

//zwalnia pamiec przydzielona na pojedynczy element listy macierzy
void
matrix_dll_array_free_elem(struct matrix_dl_array *array);

//tworzy pojedynczy element listy z macierzami, macierz ma wymiary x,y
//zwraca adres na nowa liste w przypadku sukcesu, w przypadku porazki nulla
struct matrix_dl_array *
matrix_dll_array_create_elem(unsigned x, unsigned y);

//szuka poczatku listy (prev = nul) i zwalnia kazdy element do samego konca
//(next = nul)
void
matrix_dll_array_free(struct matrix_dl_array *array);

#endif
