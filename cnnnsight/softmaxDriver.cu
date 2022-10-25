/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include "funcs.h"
#include "softmax.h"

#define TIME

SOFTMAXHANDLER softmaxNew(unsigned int inN){
	SOFTMAXHANDLER softmaxhandler = (SOFTMAXHANDLER)malloc(sizeof(SOFTMAXSTATE));
	cudaMalloc(&(softmaxhandler->sumExp),inN*sizeof(float));
	return softmaxhandler;
}

void softmaxRun(SOFTMAXHANDLER softmaxhandler,float* outs,unsigned int inN, unsigned int nodeNum){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif	
	//grids: {inputNum,nodeNum}
	//blocks: {1}
	softmax<<<inN,nodeNum>>>(outs, nodeNum,softmaxhandler->sumExp);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("softmaxRun takes %.4f ms\n", elapsed);
#endif
}

void softmaxFree(SOFTMAXHANDLER softmaxhandler){
	cudaFree(softmaxhandler->sumExp);
	free(softmaxhandler);
}
