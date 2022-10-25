/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "maxpool.h"

//#define TIME

//call:
//grid = {oN}
//block = {oD/MAXPOOLINGSIZE,oD/MAXPOOLINGSIZE}
void maxpooling(unsigned int oN,float* outputs,unsigned int oD,unsigned int steps, float* out, unsigned int newOD){
	float* opic;
	float* nopic;
	float max = 0;

	for(int outNum=0; outNum< oN;outNum++){
		nopic = out+(outNum)*newOD*newOD;
		opic = outputs+(outNum)*oD*oD;
		for(int oDX=0;oDX < newOD;oDX++){
			for(int oDY=0;oDY<newOD;oDY++){
				max = 0;
				for(int i = 0; i < steps; i++){
					for(int j = 0; j<steps;j++){
						if((steps*oDY+i)<oD && (steps*oDX+j)<oD){
							//zero padding
							if(*(opic+(steps*oDY+i)*oD+steps*oDX+j) > max)
								max = *(opic+(steps*oDY+i)*oD+steps*oDX+j);
								
						}
					}
				}
				*(nopic+oDY*newOD+oDX) = max;
			}
		}
	}

}

MAXPOOLHANDLER maxpoolNew(unsigned int steps, unsigned int oN, unsigned int oD){
	unsigned int newOD = oD/steps;
	MAXPOOLHANDLER maxpoolhandler = (MAXPOOLHANDLER)malloc(sizeof(MAXPOOLSTATE));
	maxpoolhandler->maxpoolOuts = (float*)malloc(oN*newOD*newOD*sizeof(float));
	return maxpoolhandler;
}

void maxpool(float* preOut,MAXPOOLHANDLER maxpoolhandler,unsigned int steps, unsigned int oN, unsigned int oD){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	unsigned int newOD = oD/steps;
	maxpooling(oN,preOut,oD,steps,maxpoolhandler->maxpoolOuts,newOD);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("maxpool takes %f ms\n", time_spent*1000);
#endif
}

void maxpoolFree(MAXPOOLHANDLER maxpoolhandler){
	free(maxpoolhandler->maxpoolOuts);
	free(maxpoolhandler);
}
