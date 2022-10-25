#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

typedef struct{
	float* sumExp;
}SOFTMAXSTATE;

typedef SOFTMAXSTATE* SOFTMAXHANDLER;


SOFTMAXHANDLER softmaxNew(unsigned int inN);

void softmaxRun(SOFTMAXHANDLER softmaxhandler,float* outs,unsigned int inN, unsigned int nodeNum);

void softmaxFree(SOFTMAXHANDLER softmaxhandler);

void softmax(unsigned int inputNum,float* out, unsigned int nodeNum,float* sumExp);

#endif
