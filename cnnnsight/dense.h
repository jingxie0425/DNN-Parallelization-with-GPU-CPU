#ifndef _DENSE_H_
#define _DENSE_H_

typedef struct{
	float* Weights;
	float* bias;
	float* outs;
}DENSESTATE;

typedef DENSESTATE* DENSEHANDLER;

typedef struct{
	float* Weights;
	float* bias;
	float* outs;
	float* inputs;
}HEADDENSESTATE;

typedef HEADDENSESTATE* HEADDENSEHANDLER;


DENSEHANDLER denseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum);
HEADDENSEHANDLER headDenseNew(unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum);

void denseLoad(DENSEHANDLER densehandler,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b);
void headDenseLoad(HEADDENSEHANDLER headdensehandler,unsigned int inputN,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b,float* in);

void denserun(DENSEHANDLER densehandler,unsigned int inN,unsigned int nodeNum,float* preOut,unsigned int oD, unsigned int fN);

void denserunD(DENSEHANDLER densehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts);
void headDenserunD(HEADDENSEHANDLER headdensehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts);

void denseFree(DENSEHANDLER densehandler);
void headDenseFree(HEADDENSEHANDLER headdensehandler);

#endif //_CNN_H_
