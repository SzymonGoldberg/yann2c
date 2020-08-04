#ifndef _DEEP_NETWORK_H_
#define _DEEP_NETWORK_H_

//---======= MAKRA DO FUNKCJI AKTYWACJI ======---

#define MAX(a, b)	((a) < (b) ? (b) : (a))
#define RELU_DERIV(a)	((a) <= (0.0) ? (0.0) : (1.0))

#define SIGMOID(x) 	((1.0)/((1.0) + exp((double) -x)))
#define SIGMOID_DERIV(x)((x) * ((1.0) - (double)(x)))

#define TANH(x)	       	((exp((double)(x)) - exp((double)(-x)))/	\
			(exp((double)(x)) + exp((double)(-x)))) 
#define TANH_DERIV(x)	((1.0) - ((x) * (x)))


//struktury wskazujace na siec neuronowa (nEURAL nETWORK)
//wieghts to macierz wag, output to po prostu ostatnie wyjscie z danej
//warstwy neuronow
struct nn_layer {
 	matrix_t *weights;
 	matrix_t *output;

 	matrix_t *delta;
	matrix_t *weight_delta;

	void (*activation_func)(double *, unsigned);

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

struct nn_layer * nn_layer_create(unsigned x, unsigned y, unsigned batch_size);

//funkcja dodaje siec <size> neuronow na koniec struktury <nn> (tail)
//i przypisuje jej pewna funkcje aktywacji <activation_func>
//jesli struktura byla pusta to pierwsza macierz ma wymiary - (<input>, <size>)
//w przypadku istniejacych juz wczesniej warstw - (wyjscie poprzedniej warstwy, <size>)
//<batch_size> okresla ilosc danych w serii
//zwraca 0 w przypadku porazki, w innym przypadku 1
int nn_add_layer(struct nn_array *nn, unsigned size, unsigned input,
	unsigned batch_size, void (*activation_func)(double *, unsigned));

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

//---====== DEKLARACJE FUNKCJI AKTYWACJI ======---

//funkcja poddaje <a> funkcji aktywacji, flaga <derivative> wskazuje czy
//ma zwrocic wynik funkcji czy pochodnej
void ReLU(double *a, unsigned derivative);
void sigmoid(double *a, unsigned derivative);
void xtanh(double *a, unsigned derivative);

#endif
