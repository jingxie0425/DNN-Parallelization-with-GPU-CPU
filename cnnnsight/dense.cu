/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "dense.h"
#include "densecu.h"
#include "cnn.h"

DENSEHANDLER denseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum){
	//first allocate host side
	DENSEHANDLER densehandler;
	densehandler = (DENSEHANDLER)malloc(sizeof(DENSESTATE));//malloc cpu handler

	cudaMalloc(&(densehandler->Weights),preNodeNum*nodeNum*sizeof(float));
	cudaMalloc(&(densehandler->bias),nodeNum*sizeof(float));
	cudaMalloc(&(densehandler->outs),inputN*nodeNum*sizeof(float));
	return densehandler;
}

HEADDENSEHANDLER headDenseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum){
	//first allocate host side
	HEADDENSEHANDLER headdensehandler;
	headdensehandler = (HEADDENSEHANDLER)malloc(sizeof(HEADDENSESTATE));//malloc cpu handler

	cudaMalloc(&(headdensehandler->Weights),preNodeNum*nodeNum*sizeof(float));
	cudaMalloc(&(headdensehandler->bias),nodeNum*sizeof(float));
	cudaMalloc(&(headdensehandler->outs),inputN*nodeNum*sizeof(float));
	cudaMalloc(&(headdensehandler->inputs),inputN*preNodeNum*sizeof(float));

	return headdensehandler;
}

void denseLoad(DENSEHANDLER densehandler,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b){
	cudaMemcpy(densehandler->Weights,w,preNodeNum*nodeNum*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(densehandler->bias,b,nodeNum*sizeof(float),cudaMemcpyHostToDevice);
}

void headDenseLoad(HEADDENSEHANDLER headdensehandler,unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b,float* in){
	cudaMemcpy(headdensehandler->Weights,w,preNodeNum*nodeNum*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(headdensehandler->bias,b,nodeNum*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(headdensehandler->inputs,in,inputN*preNodeNum*sizeof(float),cudaMemcpyHostToDevice);
}

/**
 * This function is for dense layer at edge of convolutional and dense (it includes flatten inside)
 * call
 * grid:{inN,oD*oD,fN} here oD is new oD after max pooling
 * block:{nodeN} 1024 maximum node number
 */
__global__
void forward(float* denseouts, float* denseweights, float* preOut,unsigned int preOD,unsigned int fN, unsigned int nodeNum){
	unsigned int inputIndex = blockIdx.x;
	unsigned int columnIndex = blockIdx.y;
	unsigned int rowIndex = blockIdx.z;
	unsigned int nodeIndex = threadIdx.x;
	float tmpMulti = *((preOut + inputIndex*preOD*preOD*fN) + (rowIndex*preOD*preOD + columnIndex)) * *(denseweights + (rowIndex + columnIndex*fN)*nodeNum + nodeIndex);
	atomicAdd(denseouts + inputIndex*nodeNum + nodeIndex,tmpMulti);
	__syncthreads();
}

/**
 * This function is for dense layer and dense layer
 * call
 * grid:{inN,preNodeNum,nodeN}
 * block:{1}
 */
__global__
void forwardD(float* predenseouts,float* currdensewights,float* currdenseouts, unsigned int preNodeNum, unsigned int nodeNum){
	unsigned int inputIndex = blockIdx.x;
	unsigned int preNodeIndex = blockIdx.y;
	unsigned int nodeIndex = blockIdx.z;
	float tmpMulti = *((predenseouts + inputIndex*preNodeNum) + preNodeIndex) * *(currdensewights + preNodeIndex*nodeNum + nodeIndex);
	atomicAdd(currdenseouts + inputIndex*nodeNum + nodeIndex,tmpMulti);
	__syncthreads();
}

/**
 * call
 * grid: {inN,nodeN}
 * block: {}
 */
__global__
void addBias(float* denseouts,float* densebias,unsigned int nodeNum){
	unsigned int inputIndex = blockIdx.x;
	unsigned int nodeIndex = blockIdx.y;
	*(denseouts + inputIndex*nodeNum + nodeIndex) = *(denseouts + inputIndex*nodeNum + nodeIndex) + *(densebias + nodeIndex);
	__syncthreads();
}
