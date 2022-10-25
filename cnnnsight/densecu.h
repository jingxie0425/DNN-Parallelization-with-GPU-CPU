#ifndef _DENSECU_H_
#define _DENSECU_H_

#include "cnn.h"
#include "dense.h"

__global__ void forward(float* denseouts, float* denseweights, float* preOut,unsigned int preOD,unsigned int fN, unsigned int nodeNum);
__global__ void forwardD(float* predenseouts,float* currdensewights,float* currdenseouts, unsigned int preNodeNum, unsigned int nodeNum);
__global__ void addBias(float* denseouts,float* densebias,unsigned int nodeNum);

#endif //_CNN_H_
