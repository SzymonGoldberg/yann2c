#ifndef __CNN_H_
#define __CNN_H_

//====== CNN ======

//sources:
//--->https://arxiv.org/pdf/1603.07285.pdf
//--->https://www.machinecurve.com/index.php/2019/09/29/understanding-transposed-convolutions/

//---====== DEKLARACJE FUNKCJI DO CNN =====---

//count kernels function
//how it work: it calculate how many kernels <krnl> we get by moving them over
//matrix which size is x = <in_x>, y = <in_y>.
//Example formula: kernel width counter = (kernel width - input width)/stride + 1
//Return value
//amount of all kernels possible positions. In case of non valid data -1.

int
cnn_count_krnl( unsigned in_x, 			//input width
				unsigned in_y, 			//input height
				const matrix_t* krnl, 	//kernel matrix
				unsigned *out_x,		//returns kernel diffrent positions in width
				unsigned *out_y,		//returns kernel diffrent positions in height
				unsigned stride);		//stride

//convolution mask creator
//how it work: it takes index matrix <output> (which is pre-allocated and non validated)
//and fill any row with indexes of kernels in different positions.
//Return value
//0 - success 1 - something go wrong

int 
cnn_conv_create(unsigned in_x,			//input width
				unsigned in_y,			//input height
				const matrix_t* kernel,	//kernel matrix
				idx_matrix_t* output,	//index matrix filled with kernel indexes in valid positions
				unsigned stride);		//stride
#endif
