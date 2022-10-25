
#include <stdio.h>
#include <time.h>
#include "relu.h"

//#define TIME

void relu(unsigned int oNin,unsigned int oD,float* outputs){
	#pragma omp parallel for collapse(3)
	for(int oN=0; oN<oNin; oN++){
		for(int oDX=0; oDX<oD; oDX++){
			for(int oDY=0; oDY<oD;oDY++){
				if(*(outputs+(oN)*oD*oD+oDY*oD+oDX) < 0){
					*(outputs+(oN)*oD*oD+oDY*oD+oDX) = 0;
				}
			}
		}
	}
}


void reluDense(unsigned int num,float* out){
	for(int n=0;n<num;n++){
		if(*(out+n) < 0){
			*(out+n) = 0;
		}
	}
}

void reluCnn( float* outputs,unsigned int fn, unsigned int inN,unsigned int oD){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	relu(inN*fn,oD,outputs);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("reluCnn takes %f ms\n", time_spent*1000);
#endif
}


void reluDen(float* outputs,unsigned int inN,unsigned int nodeNum){
#ifdef TIME
	double time_spent = 0.0;
	clock_t begin = clock();
#endif
	reluDense(inN*nodeNum,outputs);
#ifdef TIME
	clock_t end = clock();
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
	printf("reluDen takes %f ms\n", time_spent*1000);
#endif
}