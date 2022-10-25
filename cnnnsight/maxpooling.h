#ifndef _MAXPOOLING_H_
#define _MAXPOOLING_H_

__global__ void maxpooling(float* outputs,unsigned int oD,unsigned int steps, float* out, unsigned int newOD);
__global__ void agragate(float* outputs,unsigned int oD,unsigned int steps); //not in use now, might be a better method, no need new cudaMemcpy

#endif //_FUNCS_H_
