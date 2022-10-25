#ifndef _MAXPOOL_H_
#define _MAXPOOL_H_

typedef struct{
	float* maxpoolOuts;
}MAXPOOLSTATE;
typedef MAXPOOLSTATE* MAXPOOLHANDLER;

MAXPOOLHANDLER maxpoolNew(unsigned int steps, unsigned int oN, unsigned int oD);
void maxpool(float* preOut,MAXPOOLHANDLER maxpoolhandler,unsigned int steps, unsigned int oN, unsigned int oD);
void maxpoolFree(MAXPOOLHANDLER maxpoolhandler);

#endif
