/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */

#include "cnn.h"
#include "dataTrans.h"
#include "readIn.h"
#include <iostream>
#include "maxpool.h"
#include "dense.h"
#include "relu.h"
#include "softmax.h"
#include "read1D.h"
#include "read2D.h"
#include "writeOut.h"

#include<iostream>
#include <cstring>

#define INPUTDIM2	8
#define INPUTNUM2	140
#define FILTERNUM2	16

#define OUTDIM2 7

#define INPUTDIM 17
#define INPUTNUM 140//78
#define FILTERDIM 2
#define FILTERNUM 32//32   max as 4, due to thread max num per block
#define NODENUM 32
#define NODENUM2 2
#define OUTDIM  16
#define OUTNUM  4480

using namespace std;
void testOuts3D(float* realOutput,float* expectOutput,unsigned int oN,unsigned int oD);
void testOuts2D(float* realOutput,float* expectOutput, unsigned int oN,unsigned int oD);

void fifowritedemo(void* storage,void *data) {
    memcpy(storage, data, sizeof(float*));
}
void fiforeaddemo(void* storage,void *data) {
    memcpy(data, storage, sizeof(float*));
}



int main(){
    unsigned int oN = OUTNUM;
	unsigned int oD = OUTDIM; //no zero padding, step always = 1
    //load in data
    ReadFiles rf(INPUTDIM,INPUTNUM,FILTERDIM,FILTERNUM);
    //rf.readWeights();
    Read1D* rcnnweights;
    rcnnweights = new Read1D("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/cnn1Weight.csv", FILTERNUM*FILTERDIM*FILTERDIM);
    rcnnweights->readExecute();
    Read1D rcnn2weights("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/cnn2Weight.csv", FILTERNUM2*32*FILTERDIM*FILTERDIM);//2is pic amount
    rcnn2weights.readExecute();

    Read1D rcnnbias2("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/cnn2Bias.csv",FILTERNUM2);
    rcnnbias2.readExecute();

    Read1D rinputs("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/testInputX.csv",INPUTNUM*INPUTDIM*INPUTDIM);
    rinputs.readExecute();
    //rf.readInputs();
    Read1D rcnnbias("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/cnn1Bias.csv",FILTERNUM);
    rcnnbias.readExecute();
    //rf.readBias();
    Read2D rcnnexpectedout("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/conv1out/conv1out",INPUTNUM,FILTERNUM,(oD)*(oD));
    rcnnexpectedout.readExecute();
    //rf.readExpectOuts();
    Read2D rmpexpectedout("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/maxPoolout/maxpoolout",INPUTNUM,FILTERNUM,(oD/2)*(oD/2));
    rmpexpectedout.readExecute();

    Read2D rcnn2expectedout("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/conv2out/conv2out",INPUTNUM2,FILTERNUM2,OUTDIM2*OUTDIM2);
    rcnn2expectedout.readExecute();



    //rf.readExpectMpOuts();
    Read1D rdense1weights("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/dense1Weight.csv",FILTERNUM*OUTDIM2*OUTDIM2*NODENUM);
    rdense1weights.readExecute();
    //rf.readDense1Weights(FILTERNUM*64,NODENUM);
    Read1D rdense1bias("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/dense1Bias.csv",NODENUM);
    rdense1bias.readExecute();
    //rf.readDense1Bias(NODENUM);
    Read1D rdense1expectedouts("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/outdense/dense1out.csv",INPUTNUM*NODENUM);
    rdense1expectedouts.readExecute();
    //rf.readDense1ExpectedOut(NODENUM);
    Read1D rdense2weights("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/dense2Weight.csv",NODENUM*NODENUM2);
    rdense2weights.readExecute();
    //rf.readDense2Weights(NODENUM,NODENUM2);
    Read1D rdense2bias("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/dense2Bias.csv",NODENUM2);
    rdense2bias.readExecute();
    //rf.readDense2Bias(NODENUM2);
    Read1D rdense2expectedouts("/home/xiaomwu/Desktop/cuda-workspace/extractParas_2conv2/outdense/dense2out.csv",INPUTNUM*NODENUM2);
    rdense2expectedouts.readExecute();
    //rf.readDense2ExpectedOut(NODENUM2);

    READHANDLER cpuhandler = rf.getHandler();
    //float* expectedMP = rf.getmpout();
    float* expectedMP = rmpexpectedout.getData();
    
    cout << "cnn start\n";
    //construct CNN layer
    CNNHANDLER cnn1handler;
    CNNHANDLER cnn2handler;
    unsigned int picNum1 = 1;
    unsigned int picNum2 = 32;

    cnn1handler = CnnNew(INPUTDIM,INPUTNUM,FILTERDIM,FILTERNUM,oD,oN,picNum1);
    cnn2handler = CnnNew(INPUTDIM2,INPUTNUM2,FILTERDIM,FILTERNUM2,OUTDIM2,INPUTNUM2*FILTERNUM2,picNum2);

    CnnLoad(cnn1handler,INPUTDIM,INPUTNUM,FILTERDIM,FILTERNUM,picNum1,rcnnbias.getData(),rcnnweights->getData(),rinputs.getData());

    CnnLoadnonHead(cnn2handler,FILTERDIM,FILTERNUM2,picNum2,rcnnbias2.getData(),rcnn2weights.getData());

    cout << "cnn layer execute start\n";
    float* test;
    printf("test : %d",test);
    printf("cnn outputs : %d",cnn1handler->outputs);
    void* storage;
    storage = malloc(sizeof(float*));
    //memcpy(storage, &cnn1handler->outputs,sizeof(float*));
    fifowritedemo(storage,&cnn1handler->outputs);
    //memcpy(&test, storage,sizeof(float*));
    fiforeaddemo(storage,&test);
    printf("test : %d",test);
    //execute test
    cnnRunNonHead(cnn1handler->inputs,cnn1handler->filterWeights,test,cnn1handler->bias,FILTERNUM,FILTERDIM,INPUTNUM,INPUTDIM,oD,picNum1);
    reluCnn(cnn1handler->outputs,FILTERNUM,INPUTNUM,oD);
    cout << "cnn layer execute end\n";

    MAXPOOLHANDLER maxpoolhandler = maxpoolNew(2,oN,oD);
    maxpool(cnn1handler->outputs,maxpoolhandler, 2,oN,oD);
    cout << "maxpool layer execute end\n";

    oD = oD/2; //for testing maxpool layer
    float* expectOutput = expectedMP;//cpuhandler->outputs;
    //get real output
    float* realOutput= (float*)malloc(oN*oD*oD*sizeof(float));
    DataTrans dt(cpuhandler);
    dt.TransBack(realOutput,maxpoolhandler->maxpoolOuts, oD,INPUTNUM*FILTERNUM);
    cout << "real output got\n";

    //test maxpooling layer outputs
    testOuts3D(realOutput,expectOutput,oN,oD);


    cnnRunNonHead(maxpoolhandler->maxpoolOuts,cnn2handler->filterWeights,cnn2handler->outputs,cnn2handler->bias,FILTERNUM2,FILTERDIM,INPUTNUM2,INPUTDIM2,OUTDIM2,picNum2);
    reluCnn(cnn2handler->outputs,FILTERNUM2,INPUTNUM2,OUTDIM2);

    float* expectOutputcnn2 = rcnn2expectedout.getData();//cpuhandler->outputs;
    //get real output
    float* realOutputcnn2= (float*)malloc(FILTERNUM2*INPUTNUM2*OUTDIM2*OUTDIM2*sizeof(float));
    dt.TransBack(realOutputcnn2,cnn2handler->outputs, OUTDIM2,INPUTNUM2*FILTERNUM2);
    cout << "real output got\n";

    //test maxpooling layer outputs
    testOuts3D(realOutputcnn2,expectOutputcnn2,INPUTNUM2*FILTERNUM2,OUTDIM2);



    oD = OUTDIM2;

    cout << "start dense1 layer\n";
    //float* w = rf.getDense1W();
    float* w = rdense1weights.getData();
    //float* b = rf.getDense1B();
    float* b = rdense1bias.getData();
    //float* expectedDenseOut = rf.getDense1Out();
    float* expectedDenseOut = rdense1expectedouts.getData();
    float* realDenseOut= (float*)malloc(INPUTNUM*NODENUM*sizeof(float));
    DENSEHANDLER densehandler = denseNew(INPUTNUM,FILTERNUM2*oD*oD, NODENUM);
    denseLoad(densehandler,FILTERNUM2*oD*oD, NODENUM,w,b);
    denserun(densehandler,INPUTNUM, NODENUM,cnn2handler->outputs,oD,FILTERNUM2);
    reluDen(densehandler->outs,INPUTNUM,NODENUM);
    cout << "end dense1 layer\n";
    //trans back GPU out
    dt.TransBackDense(densehandler, realDenseOut,INPUTNUM,NODENUM);
    cout << "dense out trans back success\n";
    //test with expected
    testOuts2D(realDenseOut,expectedDenseOut, INPUTNUM,NODENUM);
    //dense layer 2 with softmax
    cout << "start dense2 layer\n";
    //float* w2 = rf.getDense2W();
    float* w2 = rdense2weights.getData();
    //float* b2 = rf.getDense2B();
    float* b2 = rdense2bias.getData();
    //float* expectedDense2Out = rf.getDense2Out();
    float* expectedDense2Out = rdense2expectedouts.getData();
    float* realDense2Out= (float*)malloc(INPUTNUM*NODENUM2*sizeof(float));
    DENSEHANDLER densehandler2 = denseNew(INPUTNUM,NODENUM, NODENUM2);
    denseLoad(densehandler2,NODENUM, NODENUM2,w2,b2);
    denserunD(densehandler2,INPUTNUM,NODENUM, NODENUM2,densehandler->outs);
    SOFTMAXHANDLER softmaxhandler = softmaxNew(INPUTNUM);
    softmaxRun(softmaxhandler,densehandler2->outs,INPUTNUM, NODENUM2);

    WriteOut* wout;
    wout = new WriteOut("/home/xiaomwu/Desktop/cuda-workspace/result.csv", INPUTNUM,NODENUM2);
    wout->writeExecute(densehandler2->outs);
    wout->freeWriteOut();
    delete wout;

    cout << "end dense2 layer\n";
    dt.TransBackDense(densehandler2, realDense2Out,INPUTNUM,NODENUM2);
    cout << "dense2 out trans back success\n";

    testOuts2D(realDense2Out,expectedDense2Out, INPUTNUM,NODENUM2);

    return 0;
}

//test output with 3D structure
void testOuts3D(float* realOutput,float* expectOutput, unsigned int oN,unsigned int oD){
    float real;
    float expect;
    float diff;
    bool pass = true;
    for(int i = 0; i < oN; i++){
        for(int j = 0; j < oD; j++){
            for(int k = 0; k < oD; k++){
            	real = *(realOutput+i*oD*oD+j*oD+k);
            	expect = *(expectOutput+i*oD*oD+j*oD+k);
            	diff = real-expect;
            	if(diff< 0.00001 && diff>-0.00001){
            		//cout<<"pass at "<<i<<"th outpic "<<j<<"th row "<<k<<"th column real "<<real<<" expect "<<expect<<"\n";

            	}else{
            		pass = false;
            		//cout<<"fail at "<<i<<"th outpic "<<j<<"th row "<<k<<"th column real "<<real<<" expect "<<expect<<"\n";
            	}

            }
        }
    }
    if(pass)
        cout<<"all passed\n";
    else
    	cout<<"failed\n";
}

//test output with 2D structure
void testOuts2D(float* realOutput,float* expectOutput, unsigned int oN,unsigned int oD){
    float real;
    float expect;
    float diff;
    bool pass = true;
    for(int i = 0; i < oN; i++){

		for(int k = 0; k < oD; k++){
			real = *(realOutput+i*oD+k);
			expect = *(expectOutput+i*oD+k);
			diff = real-expect;
			if(diff< 0.001 && diff>-0.001){
				//cout<<"pass at "<<i<<"th row "<<k<<"th column real "<<real<<" expect "<<expect<<"\n";

			}else{
				pass = false;
				cout<<"fail at "<<i<<"th row "<<k<<"th column real "<<real<<" expect "<<expect<<"\n";
			}
		}

    }
    if(pass)
    	cout<<"all passed\n";
    else
    	cout<<"failed\n";
}

    
