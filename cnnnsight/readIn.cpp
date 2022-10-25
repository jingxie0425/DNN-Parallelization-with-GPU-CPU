/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <ios>
#include <sstream>


#include "readIn.h"

using namespace std;

//#define DEBUG


ReadFiles::ReadFiles(int inD, int inN, int fD, int fN){
    int oN = inN*fN;
    int oD = inD-fD+1; //no zero padding, step always = 1

    maxpoolout = (float*)malloc(oN*(oD/2)*(oD/2)*sizeof(float));
    dense1W = (float*)malloc(fN*(oD/2)*(oD/2)*16*sizeof(float));//16 is nodeNum of dense 1 layer
    dense1B = (float*)malloc(16*sizeof(float));
    dense1Out = (float*)malloc(inN*16*sizeof(float));

    dense2W = (float*)malloc(16*2*sizeof(float));//16 is nodeNum of dense 1 layer
    dense2B = (float*)malloc(2*sizeof(float));
    dense2Out = (float*)malloc(inN*2*sizeof(float));

    cnnhandler = (READHANDLER)malloc(sizeof(READSTATE));
    cnnhandler->outputNum = oN;
    cnnhandler->filterDim = fD;
    cnnhandler->filterNum = fN;
    cnnhandler->inputDim = inD;
    cnnhandler->inputNum = inN;
    cnnhandler->outputDim = oD;
    cnnhandler->bias = (float*)malloc(fN*sizeof(float));

    cnnhandler->filterWeights = (float*)malloc(fN*fD*fD*sizeof(float));

    cnnhandler->inputs = (float*)malloc(inN*inD*inD*sizeof(float));

    cnnhandler->outputs = (float*)malloc(oN*oD*oD*sizeof(float));
    
}

float* ReadFiles::getDense1W(){
	return dense1W;
}

float* ReadFiles::getDense1B(){
	return dense1B;
}

float* ReadFiles::getDense1Out(){
	return dense1Out;
}

float* ReadFiles::getDense2W(){
	return dense2W;
}

float* ReadFiles::getDense2B(){
	return dense2B;
}

float* ReadFiles::getDense2Out(){
	return dense2Out;
}

READHANDLER ReadFiles::getHandler(){
    return cnnhandler;
}

float* ReadFiles::getmpout(){
	return maxpoolout;
}
//1Done
void ReadFiles::readWeights() {
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/cnnWeight.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        //vector to hold 4 float for one filter's weight
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<cnnhandler->filterNum*cnnhandler->filterDim*cnnhandler->filterDim){
            	*(cnnhandler->filterWeights+j) = x;
            }
            j++;
        }
    }
}
//1Done
void ReadFiles::readBias(){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/cnnBias.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int i = 0;
    for(std::string each : *vec){
        stringstream geek(each);
        float x;
        geek >> x;
        if(i < cnnhandler->filterNum)
        	*(cnnhandler->bias+i) =x;
        i++;
    }
} 
//1
void ReadFiles::readInputs(){
    std::fstream fin;
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/testInputX.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream ss(each);
        //vector to hold 4 float for one filter's weight
        //while(ss.good() and count < cnnhandler->inputDim*cnnhandler->inputDim){
        while(ss.good()){
        	getline(ss,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j < cnnhandler->inputNum*cnnhandler->inputDim*cnnhandler->inputDim)
            	*(cnnhandler->inputs+j) = x;
            j++;
        }
    }

} 

//2Done
void ReadFiles::readExpectOuts(){
    std::fstream fin;
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    int cc = 0; //count to limit only inputDim lines will be read
    // Open an existing file
    for(int i=0; i< cnnhandler->inputNum;i++){
        cc = 0;
        fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/conv1out/conv1out"+to_string(i)+".csv", std::ios::in);
        while(fin.good() and cc < cnnhandler->filterNum){
            getline(fin, line);
            vec->push_back(line);
            cc++;
        }
        fin.close();
    }
    int j = 0;
    int count = 0;
    for(std::string each : *vec){
        count = 0;
        std::stringstream ss(each);
        //vector to hold 4 float for one filter's weight
        while(ss.good() and count < cnnhandler->outputDim*cnnhandler->outputDim){
            getline(ss,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<cnnhandler->outputNum*cnnhandler->outputDim*cnnhandler->outputDim)
            	*(cnnhandler->outputs+j) = x;
            count++;
            j++;
        }
    }
}
//2DOne
void ReadFiles::readExpectMpOuts(){
    std::fstream fin;
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    int cc = 0; //count to limit only inputDim lines will be read
    // Open an existing file
    for(int i=0; i< cnnhandler->inputNum;i++){
        cc = 0;
        fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/maxPoolout/maxpoolout"+to_string(i)+".csv", std::ios::in);
        while(fin.good() and cc < cnnhandler->filterNum){
            getline(fin, line);
            vec->push_back(line);
            //cout<< line << '\n';
            cc++;
        }
        fin.close();
    }
    int j = 0;
    int count = 0;
    int od = cnnhandler->outputDim;
    for(std::string each : *vec){
        count = 0;
        std::stringstream ss(each);
        //vector to hold 4 float for one filter's weight
        std::vector<float> *out2DVec = new std::vector<float>();
        while(ss.good() and count < (od/2)*(od/2)){
            getline(ss,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            out2DVec->push_back(x);
            if(j<cnnhandler->outputNum*(od/2)*(od/2))
            	*(maxpoolout+j) = x;
            count++;
            j++;
        }
    }
} 
//1DONE
float* ReadFiles::readDense1Weights(int inNum, int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/dense1Weight.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<inNum*nodeNum){
            	*(dense1W+j) = x;
            }
            j++;
        }
    }
	return dense1W;
}
//1DONE
float* ReadFiles::readDense1Bias(int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/dense1Bias.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<nodeNum){
            	*(dense1B+j) = x;
            }
            j++;
        }
    }
	return dense1B;
}
//1DONE
float* ReadFiles::readDense1ExpectedOut(int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/outdense/dense1out.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<cnnhandler->inputNum*nodeNum){
            	*(dense1Out+j) = x;
            }
            j++;
        }
    }
	return dense1Out;
}
//1DONE
float* ReadFiles::readDense2Weights(int inNum, int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/dense2Weight.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<inNum*nodeNum){
            	*(dense2W+j) = x;
            }
            j++;
        }
    }
	return dense2W;
}
//1DONE
float* ReadFiles::readDense2Bias(int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/dense2Bias.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<nodeNum){
            	*(dense2B+j) = x;
            }
            j++;
        }
    }
	return dense2B;
}
//1DONE
float* ReadFiles::readDense2ExpectedOut(int nodeNum){
    std::fstream fin;
    // Open an existing file
    fin.open("/home/xiaomwu/Desktop/cuda-workspace/extractParas/outdense/dense2out.csv", std::ios::in);
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    while(fin.good()){
        getline(fin, line);
        vec->push_back(line);
    }
    int j = 0;
    for(std::string each : *vec){
        std::stringstream stream(each);
        while(stream.good()){
            getline(stream,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            if(j<cnnhandler->inputNum*nodeNum){
            	*(dense2Out+j) = x;
            }
            j++;
        }
    }
	return dense2Out;
}




