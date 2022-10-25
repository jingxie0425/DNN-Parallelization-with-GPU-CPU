#ifndef _CNNCU_H_
#define _CNNCU_H_

#include "cnn.h"
__global__ void addBias(unsigned int oD,unsigned int fN,float* biasAry,float* outputs);
__global__ void inferencePicLev(float* iPic, float* oPic, float* fPic, unsigned int inD, unsigned int fD, unsigned int oD);
__global__ void inferencePicPickHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int inD);
__global__ void inferencePicPickNonHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int picN,unsigned int inD);

#endif //_CNNCU_H_
