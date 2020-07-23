#ifndef _DEEP_NETWORK_H_
#define _DEEP_NETWORK_H_

//struktury wskazujace na siec neuronowa (nEURAL nETWORK)
//wieghts to macierz wag, output to po prostu ostatnie wyjscie z danej
//warstwy neuronow
struct nn_layer {
 	matrix_t *weights;
 	matrix_t *output;
	int (*activation_func)(matrix_t *, unsigned);
	struct nn_layer *next;
	struct nn_layer *prev;
};

struct nn_array {
 	struct nn_layer *head;
	struct nn_layer *tail;
};

//funkcja tworzy pusta strukture sieci
//w przypadku sukcesu zwraca jej adres, w innym przypadku zwraca NULL
struct nn_array * nn_create(void);

struct nn_layer * nn_layer_create(unsigned x, unsigned y);

//funkcja dodaje siec <size> neuronow na koniec struktury <nn> (tail)
//i przypisuje jej pewna funkcje aktywacji <activation_func>
//jesli struktura byla pusta to pierwsza macierz ma wymiary - (<input>, <size>)
//w przypadku istniejacych juz wczesniej warstw - (wyjscie poprzedniej warstwy, <size>)
//zwraca 0 w przypadku porazki, w innym przypadku 1
int
nn_add_layer(struct nn_array *nn, unsigned size, unsigned input,
	int (*activation_func)(matrix_t *, unsigned));

//zwalnia cala pamiec przydzielona na strukture <nn>
void nn_free(struct nn_array *nn);

//przelicza odpowiedz sieci <nn> na wejscie <input> w postaci wektora
//czyli macierzy dla ktorej y = 1. jezeli dana warstwa ma funkcje aktywacji to
//funkcja ta bierze to pod uwage. w przypadku sukcesu 0, w innym wypadku !0
int nn_predict(struct nn_array *nn, const matrix_t *input);

//wyswietlanie <nn> w formie macierzy wag i ostatnich odpowiedzi
void nn_display(const struct nn_array *nn);

//na podstawie <expected_output> modyfikuje wagi poszczegolnych warstw
//neuronow w sieci <nn>; <input> - macierz wejsciowa, <a> - wspolczynnik uczenia alpha
//w przypadku sukcesu 0, w innym wypadku 1
int nn_backpropagation(struct nn_array *nn, const matrix_t * input,
	const matrix_t* expected_output, double a);

#endif
