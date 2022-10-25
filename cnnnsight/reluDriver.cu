/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <iostream> 
#include "funcs.h"
#include "relu.h"

#define TIME

void reluCnn( float* outputs,unsigned int fn, unsigned int inN,unsigned int oD){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif	
	relu<<<{inN*fn},{oD,oD}>>>(oD,outputs);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("reluCnn takes %.4f ms\n", elapsed);
#endif
}


void reluDen(float* outputs,unsigned int inN,unsigned int nodeNum){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif	
	reluDense<<<{inN*nodeNum},1>>>(outputs);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("reluDen takes %.4f ms\n", elapsed);
#endif
}
