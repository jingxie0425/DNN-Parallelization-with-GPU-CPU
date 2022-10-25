/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "dataTrans.h"
#include <iostream>

using namespace std;

DataTrans::DataTrans(READHANDLER cpuHandler){
    cpuH = cpuHandler;
}

void DataTrans::Trans(){
    //data need to be put in GPU structure (more than two nested-level) when created
}


void DataTrans::TransBack(float* cpu_d3, float* gpuout,unsigned int oD,unsigned int oN){
	cudaMemcpy(cpu_d3,gpuout,oN*oD*oD*sizeof(float),cudaMemcpyDeviceToHost);

}

void DataTrans::TransBackDense(DENSEHANDLER densehandler, float* realOut,unsigned int inN,unsigned int nodeN){
	cudaMemcpy(realOut,densehandler->outs,nodeN*inN*sizeof(float),cudaMemcpyDeviceToHost);
}

