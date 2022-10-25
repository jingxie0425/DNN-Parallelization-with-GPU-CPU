/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cnn.h"

//#define DEBUG
//#define TIME

CNNHANDLER CnnNew(unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN, unsigned int oD, unsigned int oN,unsigned int picNum){
	//for bias
	CNNHANDLER cnnhandler = (CNNHANDLER)malloc(sizeof(CNNSTATE));
	cnnhandler->bias = (float*)malloc(fN*sizeof(float));
	//for filterweights
	cnnhandler->filterWeights = (float*)malloc(fN*picNum*fD*fD*sizeof(float));
	//for inputs
	cnnhandler->inputs = (float*)malloc(inN*picNum*inD*inD*sizeof(float));
	//for outputs
	cnnhandler->outputs = (float*)malloc(oN*oD*oD*sizeof(float));

	return cnnhandler;
}



void CnnLoad(CNNHANDLER cnnhandler,unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights, float* cpuinputs){
	//for bias
	memcpy(cnnhandler->bias, cpubias,fN*sizeof(float));
	//for filterweights
	memcpy(cnnhandler->filterWeights, cpufilterWeights,fN*picNum*fD*fD*sizeof(float));
	//for inputs
	memcpy(cnnhandler->inputs, cpuinputs,inN*picNum*inD*inD*sizeof(float));
}

void CnnLoadnonHead(CNNHANDLER cnnhandler, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights){
	//for bias
	memcpy(cnnhandler->bias, cpubias,fN*sizeof(float));
	//for filterweights
	memcpy(cnnhandler->filterWeights, cpufilterWeights,fN*picNum*fD*fD*sizeof(float));//for inputs
	//cudaMemcpy(cnnhandler->inputs, cpuinputs,inN*inD*inD*sizeof(float),cudaMemcpyHostToDevice);
}


//call:
//grid = {inN,fN}
//block = {oD,oD}
void addBias(unsigned int inN,unsigned int oD,unsigned int fN,float* biasAry,float* outputs){
	float bias;
	for(int filterNum=0;filterNum<fN;filterNum++){
		bias = *(biasAry+filterNum);
		for(int inputNum=0; inputNum<inN;inputNum++){
			for(int oDX=0; oDX < oD; oDX++){
				for(int oDY=0; oDY<oD;oDY++){
					*(outputs+(inputNum*fN+filterNum)*oD*oD+oDY*oD+oDX) += bias;
				}
			}
		}		
	}
}

//call:
//grids dim: {inN}
//blocks dim: {fN}
//grids dim: {oD,oD,inN}
//blocks dim: {fD,fD,fN}
void inferencePicLev(float* iPic, float* oPic, float* fPic, unsigned int inD, unsigned int fD, unsigned int oD){
	float tmpMulti;

	for(int bIDx=0; bIDx < oD; bIDx++){
		for(int bIDy=0; bIDy<oD;bIDy++){
			for(int tIDx=0; tIDx < fD; tIDx++){
				for(int tIDy=0; tIDy<fD;tIDy++){
					tmpMulti = *(iPic+(bIDy+tIDy)*inD+bIDx+tIDx) * *(fPic+tIDy*fD+tIDx);
					*(oPic+bIDy*oD+bIDx)+=tmpMulti;
				}
			}
		}
	}

}


void inferencePicPickNonHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int picN,unsigned int inN,unsigned int inD){
	float* iPic;
	float* oPic;
	float* fPic;
	for(int inIdx = 0; inIdx<inN; inIdx++){
		for(int picIdx = 0;picIdx<picN;picIdx++){
			for(int fIdx = 0;fIdx<fN;fIdx++){
					//Insize = picN*inPicSize 	 	inPicSize = inD*inD
					iPic = inputs + inIdx*picN*inD*inD + picIdx*inD*inD;
					//filterPicSize = fD*fD
					fPic = filterWeights + picIdx*fN*fD*fD + fIdx*fD*fD;
					//outsize = oD*oD
					oPic = outputs + oD*oD*(inIdx*fN+fIdx);
					inferencePicLev(iPic,oPic,fPic,inD,fD,oD);
			}
		}
	}

}

void inferencePicPickHead(float* inputs, float* filterWeights,float* outputs, unsigned int oD,unsigned int fD,unsigned int fN,unsigned int inN,unsigned int inD){
	float* iPic;
	float* oPic;
	float* fPic;
	for(int inIdx = 0; inIdx<inN; inIdx++){
		for(int fIdx = 0;fIdx<fN;fIdx++){
			iPic = inputs+inIdx*inD*inD;
			fPic = filterWeights+fIdx*fD*fD;
			oPic = outputs+(inIdx*fN+fIdx)*oD*oD;
			inferencePicLev(iPic,oPic,fPic,inD,fD,oD);
		}
	}
}

void cnnRun(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	inferencePicPickHead(inputs,filterWeights,outputs,oD,fd,fn,inN,inD);
	//add bias to outputs
	addBias(inN,oD,fn,bias,outputs);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("cnnRun takes %f ms\n", time_spent*1000);
#endif
}

void cnnRunNonHead(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum){
	
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif

	inferencePicPickNonHead(inputs,filterWeights,outputs,oD,fd,fn,picNum,inN,inD);
	//add bias to outputs
	addBias(inN,oD,fn,bias,outputs);

#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("cnnRunNonHead takes %f ms\n", time_spent*1000);
#endif

}

void CnnFree(CNNHANDLER cnnhandler){
	free(cnnhandler->filterWeights);
	free(cnnhandler->bias);
	free(cnnhandler->inputs);
	free(cnnhandler->outputs);

	free(cnnhandler);
}