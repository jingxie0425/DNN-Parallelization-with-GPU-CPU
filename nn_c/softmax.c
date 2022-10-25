
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "softmax.h"

//#define TIME

/*
 * call:
 * grids: {inputNum,nodeNum}
 * blocks: {1}
 */
void softmax(unsigned int inputNum,float* out, unsigned int nodeNum,float* sumExp){
	float expEach = 0;
	#pragma omp parallel for collapse(2)
	for(int inputNodeIndex=0; inputNodeIndex<inputNum;inputNodeIndex++ ){
		for(int nodeIndex=0; nodeIndex<nodeNum;nodeIndex++){
			expEach = exp(*(out+inputNodeIndex*nodeNum+nodeIndex));
			*(sumExp+inputNodeIndex)+=expEach;
		}
	}

	#pragma omp parallel for collapse(2)
	for(int inputNodeIndex=0; inputNodeIndex<inputNum;inputNodeIndex++ ){
		for(int nodeIndex=0; nodeIndex<nodeNum;nodeIndex++){
			expEach = exp(*(out+inputNodeIndex*nodeNum+nodeIndex));
			*(out+inputNodeIndex*nodeNum+nodeIndex) = expEach / *(sumExp+inputNodeIndex);
		}
	}
}

SOFTMAXHANDLER softmaxNew(unsigned int inN){
	SOFTMAXHANDLER softmaxhandler = (SOFTMAXHANDLER)malloc(sizeof(SOFTMAXSTATE));
	softmaxhandler->sumExp = (float*)malloc(inN*sizeof(float));
	return softmaxhandler;
}

void softmaxRun(SOFTMAXHANDLER softmaxhandler,float* outs,unsigned int inN, unsigned int nodeNum){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	softmax(inN,outs, nodeNum,softmaxhandler->sumExp);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("softmaxRun takes %f ms\n", time_spent*1000);
#endif
}

void softmaxFree(SOFTMAXHANDLER softmaxhandler){
	free(softmaxhandler->sumExp);
	free(softmaxhandler);
}
