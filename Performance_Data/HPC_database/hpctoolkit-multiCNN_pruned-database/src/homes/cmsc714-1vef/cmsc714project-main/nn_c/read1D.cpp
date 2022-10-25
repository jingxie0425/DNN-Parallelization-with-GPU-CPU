/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "read1D.h"

using namespace std;

Read1D::Read1D(const char* file, unsigned int size){
	fin1D.open(file, std::ios::in);
	size1D = size;
	data1D = (float*)malloc(size*sizeof(float));

}

void Read1D::readExecute(){
	std::string line, temp, part;
	std::vector<std::string> *vec = new std::vector<std::string>();
	while(fin1D.good()){
		getline(fin1D, line);
		vec->push_back(line);
		//cout<< line << '\n';
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
			if(j<size1D){//size = fn*fd*fd
				*(data1D+j) = x;
			}
			j++;
		}
	}
}

float* Read1D::getData(){
	return data1D;
}

//free inside malloced space and close file.
//need to delete object after calling this funciton to
//free space allocated by the object
void Read1D::freeRead1D(){
	fin1D.close();
	free(data1D);
}
