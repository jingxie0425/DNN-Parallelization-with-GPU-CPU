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

//#define DEBUG

CNNHANDLER CnnNew(unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN, unsigned int oD, unsigned int oN,unsigned int picNum){
	//for bias
	CNNHANDLER cnnhandler = (CNNHANDLER)malloc(sizeof(CNNSTATE));
	cudaMalloc(&(cnnhandler->bias),fN*sizeof(float));
	//for filterweights
	cudaMalloc(&(cnnhandler->filterWeights),fN*picNum*fD*fD*sizeof(float));
	//for inputs
	cudaMalloc(&(cnnhandler->inputs),inN*picNum*inD*inD*sizeof(float));
	//for outputs
	cudaMalloc(&(cnnhandler->outputs),oN*oD*oD*sizeof(float));

	return cnnhandler;
}



void CnnLoad(CNNHANDLER cnnhandler,unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights, float* cpuinputs){
	//for bias
	cudaMemcpy(cnnhandler->bias, cpubias,fN*sizeof(float),cudaMemcpyHostToDevice);
	//for filterweights
	cudaMemcpy(cnnhandler->filterWeights, cpufilterWeights,fN*picNum*fD*fD*sizeof(float),cudaMemcpyHostToDevice);
	//for inputs
	cudaMemcpy(cnnhandler->inputs, cpuinputs,inN*picNum*inD*inD*sizeof(float),cudaMemcpyHostToDevice);
}

void CnnLoadnonHead(CNNHANDLER cnnhandler, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights){
	//for bias
	cudaMemcpy(cnnhandler->bias, cpubias,fN*sizeof(float),cudaMemcpyHostToDevice);
	//for filterweights
	cudaMemcpy(cnnhandler->filterWeights, cpufilterWeights,fN*picNum*fD*fD*sizeof(float),cudaMemcpyHostToDevice);
	//for inputs
	//cudaMemcpy(cnnhandler->inputs, cpuinputs,inN*inD*inD*sizeof(float),cudaMemcpyHostToDevice);
}

//call:
//grid = {inN,fN}
//block = {oD,oD}
__global__
void addBias(unsigned int oD,unsigned int fN,float* biasAry,float* outputs){
	unsigned int filterIdx = blockIdx.y;
	unsigned int oDX = threadIdx.x;
	unsigned int oDY = threadIdx.y;
	unsigned int inputIdx = blockIdx.x;

	float bias = *(biasAry+filterIdx);
	*(outputs+(inputIdx*fN+filterIdx)*oD*oD+oDY*oD+oDX) += bias;
	__syncthreads();
}

/*call:
 * grid:{inN,fN}
 * block:{1}
 * */
__global__
void inferencePicPickHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int inD){
	unsigned int filterIdx = blockIdx.y;
	unsigned int inputIdx = blockIdx.x;

	float* iPic;
	float* oPic;
	float* fPic;

	iPic = inputs+inputIdx*inD*inD;
	fPic = filterWeights+filterIdx*fD*fD;
	oPic = outputs+(inputIdx*fN+filterIdx)*oD*oD;

	inferencePicLev<<<{oD,oD},{fD,fD}>>>(iPic, oPic, fPic, inD, fD, oD);
}

/*call:
 * inN: input sample amount
 * picN: pic amount of each sample
 * fN: filter amount, each filter has picN filterPic for pics in each sample
 * grid:{inN,picN,fN}
 * block:{1}
 * */
__global__
void inferencePicPickNonHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int picN,unsigned int inD){
	unsigned int inIdx = blockIdx.x;
	unsigned int picIdx = blockIdx.y;
	unsigned int fIdx = blockIdx.z;

	float* iPic;
	float* oPic;
	float* fPic;

	//Insize = picN*inPicSize 	 	inPicSize = inD*inD
	iPic = inputs + inIdx*picN*inD*inD + picIdx*inD*inD;
	//filterPicSize = fD*fD
	fPic = filterWeights + picIdx*fN*fD*fD + fIdx*fD*fD;
	//outsize = oD*oD
	oPic = outputs + oD*oD*(inIdx*fN+fIdx);

	inferencePicLev<<<{oD,oD},{fD,fD}>>>(iPic, oPic, fPic, inD, fD, oD);
}

/*
 * call: grid{oD,oD}
 * 		 block{fD,fD}
 * */
__global__
void inferencePicLev(float* iPic, float* oPic, float* fPic, unsigned int inD, unsigned int fD, unsigned int oD){
	unsigned int tIDx = threadIdx.x;
	unsigned int tIDy = threadIdx.y;
	unsigned int bIDx = blockIdx.x;
	unsigned int bIDy = blockIdx.y;
														//[0,1,2,3]
	float tmpMulti = *(iPic+(bIDy+tIDy)*inD+bIDx+tIDx) * *(fPic+tIDy*fD+tIDx);

	atomicAdd(oPic+bIDy*oD+bIDx,tmpMulti);
	__syncthreads();
}





