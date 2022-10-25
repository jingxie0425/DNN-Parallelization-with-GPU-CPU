/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include "cnn.h"
#include "cnncu.h"
#include "funcs.h"

//#define DEBUG
#define TIME

//not used yet
void cnnRun(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	inferencePicPickHead<<<{inN,fn},1>>>(inputs,filterWeights,outputs,oD,fd,fn,inD);
	//add bias to outputs
	addBias<<<{inN,fn},{oD,oD}>>>(oD,fn,bias,outputs);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cnnRun takes %.4f ms\n", elapsed);
#endif
}

void cnnRunNonHead(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	inferencePicPickNonHead<<<{inN,picNum,fn},1>>>(inputs, filterWeights,outputs,oD,fd,fn,picNum,inD);
	//add bias to outputs
	addBias<<<{inN,fn},{oD,oD}>>>(oD,fn,bias,outputs);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cnnRunNonHead takes %.4f ms\n", elapsed);
#endif
}

void CnnFree(CNNHANDLER cnnhandler){
	cudaFree(cnnhandler->filterWeights);
	cudaFree(cnnhandler->bias);
	cudaFree(cnnhandler->inputs);
	cudaFree(cnnhandler->outputs);

	free(cnnhandler);
}
