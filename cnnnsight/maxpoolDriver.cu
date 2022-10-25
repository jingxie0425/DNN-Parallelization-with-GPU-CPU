/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <iostream>
#include "cnn.h"
#include "cnncu.h"
#include "funcs.h"
#include "maxpooling.h"
#include "maxpool.h"

#define TIME

MAXPOOLHANDLER maxpoolNew(unsigned int steps, unsigned int oN, unsigned int oD){
	unsigned int newOD = oD/steps;
	MAXPOOLHANDLER maxpoolhandler = (MAXPOOLHANDLER)malloc(sizeof(MAXPOOLSTATE));
	cudaMalloc(&(maxpoolhandler->maxpoolOuts),oN*newOD*newOD*sizeof(float));//malloc new space for maxpooling out

	return maxpoolhandler;
}

void maxpool(float* preOut,MAXPOOLHANDLER maxpoolhandler,unsigned int steps, unsigned int oN, unsigned int oD){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	unsigned int newOD = oD/steps;
	maxpooling<<<oN,{newOD,newOD}>>>(preOut,oD,steps,maxpoolhandler->maxpoolOuts,newOD);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("maxpool takes %.4f ms\n", elapsed);
#endif
}

void maxpoolFree(MAXPOOLHANDLER maxpoolhandler){
	cudaFree(maxpoolhandler->maxpoolOuts);
	free(maxpoolhandler);
}