#ifndef _CNN_H_
#define _CNN_H_

typedef struct{
	float* filterWeights; //three D arrary store filter weights
	float* inputs;	//3D ary store inputs
	float* outputs; //3D ary store outputs
	float* bias;
}CNNSTATE;
typedef CNNSTATE* CNNHANDLER;

//void cnn(int inD, int inN, int fD, int fN);
CNNHANDLER CnnNew(unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN, unsigned int oD, unsigned int oN,unsigned int picNum);
void CnnLoad(CNNHANDLER cnnhandler,unsigned int inD, unsigned int inN, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights, float* cpuinputs);
void CnnLoadnonHead(CNNHANDLER cnnhandler, unsigned int fD, unsigned int fN,unsigned int picNum, float* cpubias, float* cpufilterWeights);

void cnnRun(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum);
void cnnRunNonHead(float* inputs, float* filterWeights,float* outputs,float* bias,unsigned int fn, unsigned int fd, unsigned int inN,unsigned int inD,unsigned int oD,unsigned int picNum);

void CnnFree(CNNHANDLER cnnhandler);

#endif //_CNN_H_
