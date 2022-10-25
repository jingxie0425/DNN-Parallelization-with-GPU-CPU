#ifndef _FUNCS_H_
#define _FUNCS_H_

#include "dense.h"
#include "cnn.h"

__global__  void relu(unsigned int oD,float* outputs);
__global__ void reluDense(float* out);
__global__ void softmax(float* out, unsigned int nodeNum,float* sumExp);
#endif //_FUNCS_H_
