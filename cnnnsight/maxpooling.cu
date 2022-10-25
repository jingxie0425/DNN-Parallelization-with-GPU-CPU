#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "cnn.h"
/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "cnncu.h"
#include "maxpooling.h"
//call:
//grid = {oN}
//block = {oD/MAXPOOLINGSIZE,oD/MAXPOOLINGSIZE}
__global__
void maxpooling(float* outputs,unsigned int oD,unsigned int steps, float* out, unsigned int newOD){
	unsigned int outNum = blockIdx.x;
	unsigned int oDX = threadIdx.x;
	unsigned int oDY = threadIdx.y;
	//unsigned int fN = cnnhandler->filterNum;
	float *opic = outputs+(outNum)*oD*oD;
	float *nopic = out+(outNum)*newOD*newOD;
	float max = 0;
	//choose the max within step range
	for(int i = 0; i < steps; i++){
		for(int j = 0; j<steps;j++){
			if((steps*oDY+i)<oD && (steps*oDX+j)<oD){
				//zero padding
				if(*(opic+(steps*oDY+i)*oD+steps*oDX+j) > max)
									max = *(opic+(steps*oDY+i)*oD+steps*oDX+j);
			}

		}
	}
	//store max into the root position
	*(nopic+oDY*newOD+oDX) = max;
	__syncthreads();
}

//call:
//grid = {oN}
//block = {oD/MAXPOOLINGSIZE,oD/MAXPOOLINGSIZE}
__global__
void agragate(float* outputs,unsigned int oD,unsigned int steps){
	unsigned int outNum = blockIdx.x;
	unsigned int oDX = threadIdx.x;
	unsigned int oDY = threadIdx.y;
	float *opic = outputs+(outNum)*oD*oD;
	float max = *(opic+(steps*oDY)*oD+steps*oDX);
	//wait everyone get max
	__syncthreads();
	//resign max
	*(opic+(steps*oDY-oDY)*oD+steps*oDX-oDX) = max;
}

