#ifndef _MAXPOOL_H_
#define _MAXPOOL_H_

typedef struct{
	float* maxpoolOuts;
}MAXPOOLSTATE;
typedef MAXPOOLSTATE* MAXPOOLHANDLER;

MAXPOOLHANDLER maxpoolNew(unsigned int steps, unsigned int oN, unsigned int oD);
void maxpool(float* preOut,MAXPOOLHANDLER maxpoolhandler,unsigned int steps, unsigned int oN, unsigned int oD);
void maxpoolFree(MAXPOOLHANDLER maxpoolhandler);

void maxpooling(float* outputs,unsigned int oD,unsigned int steps, float* out, unsigned int newOD);
void agragate(float* outputs,unsigned int oD,unsigned int steps); //not in use now, might be a better method, no need new cudaMemcpy


#endif