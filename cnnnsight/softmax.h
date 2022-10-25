#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

typedef struct{
	float* sumExp;
}SOFTMAXSTATE;

typedef SOFTMAXSTATE* SOFTMAXHANDLER;


SOFTMAXHANDLER softmaxNew(unsigned int inN);

void softmaxRun(SOFTMAXHANDLER softmaxhandler,float* outs,unsigned int inN, unsigned int nodeNum);

void softmaxFree(SOFTMAXHANDLER softmaxhandler);

#endif
