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
//wieghts to macierz wag, output to wyjscie z danej warstwy neuronow
struct nn_layer {
 	matrix_t *weights;	//wagi polaczen miedzy neuronami
 	matrix_t *output;	//wartosci danej warstwy neuronow

 	matrix_t *delta;
	matrix_t *weight_delta;

	matrix_t *dropout_mask;
	double dropout_rate;	//procentowa ilosc wlaczonych neuronow w dropout_mask

	void (*activation_func)(double *, unsigned);	//funkcja aktywacji

	struct nn_layer *next;
	struct nn_layer *prev;
};

struct nn_array {
 	struct nn_layer *head;
	struct nn_layer *tail;
};

typedef struct {
	matrix_t *kernels;
	matrix_t *output;
	matrix_t *cropped_input;

	void (*activation_func)(double *, unsigned);	//funkcja aktywacji

	unsigned in_x, in_y, stride;
} cnn_layer;

//funkcja tworzy pusta strukture sieci
//w przypadku sukcesu zwraca jej adres, w innym przypadku zwraca NULL
struct nn_array * nn_create(void);

struct nn_layer * nn_layer_create(unsigned x, unsigned y, unsigned batch_size);

//funkcja dodaje siec <size> neuronow na koniec struktury <nn> (tail)
//i przypisuje jej pewna funkcje aktywacji <activation_func>
//jesli struktura byla pusta to pierwsza macierz ma wymiary - (<input>, <size>)
//w przypadku istniejacych juz wczesniej warstw - (wyjscie poprzedniej warstwy, <size>)
//<batch_size> okresla ilosc danych w serii, jest brane pod uwage tylko przy
//pierwszej warstwie, w kolejnych wartosc ta jest pobierana z pierwszej
//<dropout> to procentowa ilosc neuronow do wylaczenia w przypadku uzywania
//tej metody - aby jej nie uzywac wystarczy wpisac 0.0
//zwraca 0 w przypadku porazki, w innym przypadku 1
int nn_add_layer(struct nn_array *nn, unsigned size, unsigned input,
unsigned batch_size, void (*activation_func)(double *, unsigned), double dropout);

//zwalnia cala pamiec przydzielona na strukture <nn>
void nn_free(struct nn_array *nn);

//przelicza odpowiedz sieci <nn> na wejscie <input> w postaci macierzy.
//Jezeli dana warstwa ma funkcje aktywacji to funkcja ta bierze to pod uwage.
//flaga <dropout> okresla czy uzywac metode dropout
//w przypadku sukcesu 0, w innym wypadku !0
int nn_predict(struct nn_array *nn, const matrix_t *input, char dropout);

//wyswietlanie <nn> w formie macierzy wag i ostatnich odpowiedzi
void nn_display(const struct nn_array *nn);

//na podstawie <expected_output> modyfikuje wagi poszczegolnych warstw
//neuronow w sieci <nn>; <input> - macierz wejsciowa, <a> - wspolczynnik uczenia alpha
//<dropout> - flaga wskazuje czy korzystac z metody uczenia dropout
//w przypadku sukcesu 0, w innym wypadku 1
//jesli obawiasz sie ze Twoje obliczenia moga potrwac dlugo mozesz ustawic
//flage <verbose> na 1, wtedy program bedzie wraz z dzialaniem wyswietlal
//komunikaty tekstowe
int nn_backpropagation(struct nn_array *nn, const matrix_t * input,
	const matrix_t* exp_output, double a, char dropout, char verbose);

//TODO
//>OPIS
int nn_dropout_reroll(struct nn_array *nn);

//TODO
//>OPIS
int nn_softmax(struct nn_array * nn);

int nn_read(struct nn_array *nn, const char* filename);

int nn_write(const struct nn_array *nn, const char* filename);

void nn_fill_rng(struct nn_array *nn, double min, double max);

//---====== DEKLARACJE FUNKCJI DO CNN =====---

int cnn_count_kernel(unsigned input_x, unsigned input_y,
	unsigned krnl_x, unsigned krnl_y, unsigned stride);

cnn_layer * cnn_create(	unsigned input_x, unsigned input_y,
		void (*activation_func)(double *, unsigned), unsigned stride);

int cnn_add_kernel(cnn_layer *cnn, const matrix_t *krnl);
void cnn_free(cnn_layer* cnn);
//---====== DEKLARACJE FUNKCJI AKTYWACJI ======---

//funkcja poddaje <a> funkcji aktywacji, flaga <derivative> wskazuje czy
//ma zwrocic wynik funkcji czy pochodnej
void ReLU(double *a, unsigned derivative);
void sigmoid(double *a, unsigned derivative);
void xtanh(double *a, unsigned derivative);

#endif
