#ifndef _DEEP_NETWORK_H_
#define _DEEP_NETWORK_H_

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
