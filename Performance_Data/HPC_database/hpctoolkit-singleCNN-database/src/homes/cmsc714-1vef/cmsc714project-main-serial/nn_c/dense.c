/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dense.h"

//#define TIME

DENSEHANDLER denseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum){
	//first allocate host side
	DENSEHANDLER densehandler;
	densehandler = (DENSEHANDLER)malloc(sizeof(DENSESTATE));//malloc cpu handler

	densehandler->Weights = (float*)malloc(preNodeNum*nodeNum*sizeof(float));
	densehandler->bias = (float*)malloc(nodeNum*sizeof(float));
	densehandler->outs = (float*)malloc(inputN*nodeNum*sizeof(float));
	return densehandler;
}

HEADDENSEHANDLER headDenseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum){
	//first allocate host side
	HEADDENSEHANDLER headdensehandler;
	headdensehandler = (HEADDENSEHANDLER)malloc(sizeof(HEADDENSESTATE));//malloc cpu handler

	headdensehandler->Weights = (float*)malloc(preNodeNum*nodeNum*sizeof(float));
	headdensehandler->bias = (float*)malloc(nodeNum*sizeof(float));
	headdensehandler->outs = (float*)malloc(inputN*nodeNum*sizeof(float));
	headdensehandler->inputs = (float*)malloc(inputN*preNodeNum*sizeof(float));

	return headdensehandler;
}

void denseLoad(DENSEHANDLER densehandler,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b){
	memcpy(densehandler->Weights,w,preNodeNum*nodeNum*sizeof(float));
	memcpy(densehandler->bias,b,nodeNum*sizeof(float));
}

void headDenseLoad(HEADDENSEHANDLER headdensehandler,unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b,float* in){
	memcpy(headdensehandler->Weights,w,preNodeNum*nodeNum*sizeof(float));
	memcpy(headdensehandler->bias,b,nodeNum*sizeof(float));
	memcpy(headdensehandler->inputs,in,inputN*preNodeNum*sizeof(float));
}

/**
 * This function is for dense layer at edge of convolutional and dense (it includes flatten inside)
 * call
 * grid:{inN,oD*oD,fN} here oD is new oD after max pooling
 * block:{nodeN} 1024 maximum node number
 */
void forward(unsigned int inN,float* denseouts, float* denseweights, float* preOut,unsigned int preOD,unsigned int fN, unsigned int nodeNum){
	float tmpMulti = 0;

	for(int inputIndex=0; inputIndex<inN;inputIndex++){
		for(int columnIndex=0; columnIndex<preOD*preOD;columnIndex++){
			for(int rowIndex=0;rowIndex<fN;rowIndex++){
				for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
					tmpMulti = *((preOut + inputIndex*preOD*preOD*fN) + (rowIndex*preOD*preOD + columnIndex)) * *(denseweights + (rowIndex + columnIndex*fN)*nodeNum + nodeIndex);
					*(denseouts + inputIndex*nodeNum + nodeIndex)+=tmpMulti;
					//printf("\n%f\n",tmpMulti);
				}
			}
		}
	}

}

/**
 * This function is for dense layer and dense layer
 * call
 * grid:{inN,preNodeNum,nodeN}
 * block:{1}
 */
void forwardD(unsigned int inN,float* predenseouts,float* currdensewights,float* currdenseouts, unsigned int preNodeNum, unsigned int nodeNum){
	float tmpMulti = 0;

	for(int inputIndex=0; inputIndex<inN;inputIndex++){
		for(int preNodeIndex=0;preNodeIndex<preNodeNum;preNodeIndex++){
			for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
				tmpMulti = *((predenseouts + inputIndex*preNodeNum) + preNodeIndex) * *(currdensewights + preNodeIndex*nodeNum + nodeIndex);
				*(currdenseouts + inputIndex*nodeNum + nodeIndex)+=tmpMulti;
				//printf("\n%f\n",tmpMulti);
			}
		}
	}

}

/**
 * call
 * grid: {inN,nodeN}
 * block: {}
 */
void addBias(unsigned int inN, float* denseouts,float* densebias,unsigned int nodeNum){
	for(int inputIndex=0;inputIndex<inN;inputIndex++){
		for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
			*(denseouts + inputIndex*nodeNum + nodeIndex) = *(denseouts + inputIndex*nodeNum + nodeIndex) + *(densebias + nodeIndex);
		}
	}
}

void denserun(DENSEHANDLER densehandler,unsigned int inN,unsigned int nodeNum,float* preOut,unsigned int oD, unsigned int fN){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	forward(inN,densehandler->outs, densehandler->Weights,preOut,oD,fN,nodeNum);
	addBias(inN,densehandler->outs,densehandler->bias,nodeNum);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("denserun takes %f ms\n", time_spent*1000);
#endif
}

void denserunD(DENSEHANDLER densehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	forwardD(inN,predenseouts,densehandler->Weights,densehandler->outs, preNodeNum, nodeNum);
	addBias(inN,densehandler->outs,densehandler->bias,nodeNum);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("denserunD takes %f ms\n", time_spent*1000);
#endif
}

void headDenserunD(HEADDENSEHANDLER headdensehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	forwardD(inN,predenseouts,headdensehandler->Weights,headdensehandler->outs, preNodeNum, nodeNum);
	addBias(inN,headdensehandler->outs,headdensehandler->bias,nodeNum);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("headDenserunD takes %f ms\n", time_spent*1000);
#endif	
}

void denseFree(DENSEHANDLER densehandler){
	free(densehandler->Weights);
	free(densehandler->bias);
	free(densehandler->outs);
	free(densehandler);
}

void headDenseFree(HEADDENSEHANDLER headdensehandler){
	free(headdensehandler->Weights);
	free(headdensehandler->bias);
	free(headdensehandler->outs);
	free(headdensehandler->inputs);

	free(headdensehandler);
}