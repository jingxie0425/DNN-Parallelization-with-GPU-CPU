
#include "writeOut.h"

WriteOut::WriteOut(const char* file, unsigned int d1,unsigned int d2){
	D1 = d1;
	D2 = d2;
	fout.open(file,std::ios::out);
}

void WriteOut::outDirectResult(float* gpudataptr){
	for(int i = 0 ; i < D1;i++){
		for(int j = 0; j <D2;j++){
			fout << *(gpudataptr+i*D2 +j);
			fout << " ";
		}
		fout << "\n";
	}
}

void WriteOut::outConvertResult(float* gpudataptr){
	for(int i = 0 ; i < D1;i++){
		if(*(gpudataptr+i*D2 +0) > *(gpudataptr+i*D2 +1))
			fout << "0\n";
		else
			fout << "1\n";
	}
}

void WriteOut::writeExecute(float* gpudataptr){
	//outDirectResult(gpudataptr);
	//or
	outConvertResult(gpudataptr);
}

void WriteOut::freeWriteOut(){
	fout.close();
}
