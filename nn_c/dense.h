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
void headDenseLoad(HEADDENSEHANDLER headdensehandler,unsigned int preNodeNum, unsigned int nodeNum, float* w, float* b,float* in);

void forward(unsigned int inN,float* denseouts, float* denseweights, float* preOut,unsigned int preOD,unsigned int fN, unsigned int nodeNum);
void forwardD(unsigned int inN,float* predenseouts,float* currdensewights,float* currdenseouts, unsigned int preNodeNum, unsigned int nodeNum);
void addBias(unsigned int inN, float* denseouts,float* densebias,unsigned int nodeNum);
void denserun(DENSEHANDLER densehandler,unsigned int inN,unsigned int nodeNum,float* preOut,unsigned int oD, unsigned int fN);
void denserunD(DENSEHANDLER densehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts);
void headDenserunD(HEADDENSEHANDLER headdensehandler,unsigned int inN,unsigned int preNodeNum, unsigned int nodeNum,float* predenseouts);

void denseFree(DENSEHANDLER densehandler);
void headDenseFree(HEADDENSEHANDLER headdensehandler);


#endif //_CNN_H_