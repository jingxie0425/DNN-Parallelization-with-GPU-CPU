/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "writeOut.h"

void writeTrans(float* cpudata,float* gpudata, unsigned int d1,unsigned int d2){
	cudaMemcpy(cpudata,gpudata,d1*d2*sizeof(float),cudaMemcpyDeviceToHost);
}

