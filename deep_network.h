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
//TODO
//>optymalizacja
matrix_t *
deep_neural_network(matrix_t input, struct matrix_array network);

#endif
