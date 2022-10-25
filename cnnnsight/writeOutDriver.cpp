/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "writeOut.h"

WriteOut::WriteOut(const char* file, unsigned int d1,unsigned int d2){
	D1 = d1;
	D2 = d2;
	cpudata = (float*)malloc(D1*D2*sizeof(float));
	fout.open(file,std::ios::out);
}

void WriteOut::outDirectResult(){
	for(int i = 0 ; i < D1;i++){
		for(int j = 0; j <D2;j++){
			fout << *(cpudata+i*D2 +j);
			fout << " ";
		}
		fout << "\n";
	}
}

void WriteOut::outConvertResult(){
	for(int i = 0 ; i < D1;i++){
		if(*(cpudata+i*D2 +0) > *(cpudata+i*D2 +1))
			fout << "0\n";
		else
			fout << "1\n";
	}
}

void WriteOut::writeExecute(float* gpudataptr){
	writeTrans(cpudata,gpudataptr,D1,D2);

	//outDirectResult();
	//or
	outConvertResult();
}


void WriteOut::freeWriteOut(){
	fout.close();
	free(cpudata);
}
