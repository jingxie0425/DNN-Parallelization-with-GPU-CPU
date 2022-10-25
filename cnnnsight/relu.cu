/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "cnn.h"
#include "cnncu.h"
#include "funcs.h"
#include "dense.h"

//call:
//grid = {oN}
//block = {oD,oD}
__global__
void relu(unsigned int oD,float* outputs){
	unsigned int oN = blockIdx.x;
	unsigned int oDX = threadIdx.x;
	unsigned int oDY = threadIdx.y;

	if(*(outputs+(oN)*oD*oD+oDY*oD+oDX) < 0){
		*(outputs+(oN)*oD*oD+oDY*oD+oDX) = 0;
	}

	__syncthreads();
}

__global__
void reluDense(float* out){
	unsigned int n = blockIdx.x;

	if(*(out+n) < 0){
		*(out+n) = 0;
	}

	__syncthreads();
}

