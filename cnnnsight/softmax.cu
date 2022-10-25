/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dense.h"
#include "funcs.h"

/*
 * call:
 * grids: {inputNum}
 * blocks: {nodeNum}
 */
__global__
void softmax(float* out, unsigned int nodeNum,float* sumExp){
	unsigned int inputNodeIndex = blockIdx.x;
	unsigned int nodeIndex = threadIdx.x;

	float expEach = exp(*(out+inputNodeIndex*nodeNum+nodeIndex));
	atomicAdd(sumExp+inputNodeIndex,expEach);
	__syncthreads();
	*(out+inputNodeIndex*nodeNum+nodeIndex) = expEach / *(sumExp+inputNodeIndex);
	__syncthreads();
}
