#ifndef _DEEP_NETWORK_H_
#define _DEEP_NETWORK_H_

//struktury wskazujace na siec neuronowa (nEURAL nETWORK)
//wieghts to macierz wag, output to po prostu ostatnie wyjscie z danej
//warstwy neuronow
struct nn_layer {
 	matrix_t *weights;
 	matrix_t *output;
	struct nn_layer *next;
	struct nn_layer *prev;
};

struct nn_array {
 	struct nn_layer *head;
	struct nn_layer *tail;
};

//funkcja liczy macierz wyjsciowa mnozac wejscie przez transponowana macierz wag
//alokuje pamiec na macierz wyjsciowa i zwraca na nia wskaznik w przypadku sukcesu
//w innym przypadku zwraca NULL
matrix_t * neural_network(const matrix_t weights, const matrix_t input);

//zwraca odpowiedz sieci neuronowej - w przypadku porazki NULL,
//dane do niej podajemy w formie listy dwukierunkowej z macierzami (network)
//wejscie podajemy jako pojedyncza macierz
matrix_t *
deep_neural_network(matrix_t input, struct matrix_array network);

//funkcja tworzy pusta strukture sieci
//w przypadku sukcesu zwraca jej adres, w innym przypadku zwraca NULL
struct nn_array * nn_create(void);

struct nn_layer * nn_layer_create(unsigned x, unsigned y);

//funkcja dodaje siec <size> neuronow na koniec struktury <nn> (tail)
//jesli struktura byla pusta to pierwsza macierz ma wymiary <input>, <size>
//w przypadku istniejacych juz wczesniej warstw
//funkcja ma rozmiar wyjscie poprzedniej warstwy, <size>
//zwraca 0 w przypadku porazki, w innym przypadku 1
int nn_add_layer(struct nn_array *nn, unsigned size, unsigned input);

//zwalnia cala pamiec przydzielona na strukture <nn>
void nn_free(struct nn_array *nn);

//przelicza odpowiedz sieci <nn> na wejscie <input> w postaci wektora
//czyli macierzy dla ktorej y = 1
//w przypadku sukcesu 0, w innym wypadku !0
int nn_predict(struct nn_array *nn, const matrix_t *input);

//wyswietlanie <nn> w formie macierzy wag i ostatnich odpowiedzi
void nn_display(const struct nn_array *nn);

//na podstawie <expected_output> modyfikuje wagi poszczegolnych warstw
//neuronow w sieci <nn> <input> - macierz wejsciowa,
//w przypadku sukcesu 0, w innym wypadku 1
int nn_backpropagation(struct nn_array *nn, const matrix_t * input,
	const matrix_t* expected_output, double a);

#endif
