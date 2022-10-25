/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include "dense.h"
#include "densecu.h"
#include "funcs.h"

#define TIME

void denserun(DENSEHANDLER densehandler,unsigned int inN,unsigned int nodeNum,float* preOut,unsigned int oD, unsigned int fN){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	//grid:{inN,oD*oD*fN,nodeN} here oD is new oD after max pooling
	//block:{1}
	forward<<<{inN,oD*oD,fN},nodeNum>>>(densehandler->outs, densehandler->Weights,preOut,oD,fN,nodeNum);
	//grid: {inN,nodeN}
	//block: {1}
	addBias<<<{inN,nodeNum},1>>>(densehandler->outs,densehandler->bias,nodeNum);
	//apply relu nonlinearlity to outputs
	//reluDense<<<{inN*nodeNum},1>>>(densehandler->outs);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("denserun takes %.4f ms\n", elapsed);
#endif
}


void denserunD(DENSEHANDLER densehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	//grid:{inN,preNodeNum,nodeN}
	//block:{1}
	forwardD<<<{inN,preNodeNum,nodeNum},1>>>(predenseouts,densehandler->Weights,densehandler->outs, preNodeNum, nodeNum);
	//grid: {inN,nodeN}
	//block: {1}
	addBias<<<{inN,nodeNum},1>>>(densehandler->outs,densehandler->bias,nodeNum);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("denserunD takes %.4f ms\n", elapsed);
#endif
}

void headDenserunD(HEADDENSEHANDLER headdensehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts){
#ifdef TIME
	float elapsed=0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	//grid:{inN,preNodeNum,nodeN}
	//block:{1}
	forwardD<<<{inN,preNodeNum,nodeNum},1>>>(predenseouts,headdensehandler->Weights,headdensehandler->outs, preNodeNum, nodeNum);
	//grid: {inN,nodeN}
	//block: {1}
	addBias<<<{inN,nodeNum},1>>>(headdensehandler->outs,headdensehandler->bias,nodeNum);
#ifdef TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("headDenserunD takes %.4f ms\n", elapsed);
#endif
}

void denseFree(DENSEHANDLER densehandler){
	cudaFree(densehandler->Weights);
	cudaFree(densehandler->bias);
	cudaFree(densehandler->outs);
	free(densehandler);
}

void headDenseFree(HEADDENSEHANDLER headdensehandler){
	cudaFree(headdensehandler->Weights);
	cudaFree(headdensehandler->bias);
	cudaFree(headdensehandler->outs);
	cudaFree(headdensehandler->inputs);

	free(headdensehandler);
}