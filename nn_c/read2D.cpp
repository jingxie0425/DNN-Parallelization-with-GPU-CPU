
#include "read2D.h"

using namespace std;

Read2D::Read2D(const char* folderIn, unsigned int PicNumIn1, unsigned int PicNumIn2, unsigned int PicSizeIn){
	folder = folderIn;
	PicNum1 = PicNumIn1;
	PicNum2 = PicNumIn2;
	PicSize = PicSizeIn;
	data2D = (float*)malloc(PicNumIn1*PicNumIn2*PicSizeIn*sizeof(float));

}

void Read2D::readExecute(){
    std::fstream fin;
    std::string line, temp, part;
    std::vector<std::string> *vec = new std::vector<std::string>();
    int cc = 0; //count to limit only inputDim lines will be read
    // Open an existing file
    for(int i=0; i< PicNum1;i++){
        cc = 0;
        fin.open(folder+to_string(i)+".csv", std::ios::in);
        while(fin.good() and cc < PicNum2){
            getline(fin, line);
            vec->push_back(line);
            //cout<< line << '\n';
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
        std::vector<float> *out2DVec = new std::vector<float>();
        while(ss.good() and count < PicSize){
            getline(ss,part,' ');
            stringstream geek(part);
            float x;
            geek >> x;
            out2DVec->push_back(x);
            if(j<PicNum1*PicNum2*PicSize)
            	*(data2D+j) = x;
            count++;
            j++;
        }
    }
}

float* Read2D::getData(){
	return data2D;
}

//free inside malloced space and close file.
//need to delete object after calling this funciton to
//free space allocated by the object
void Read2D::freeRead2D(){
	free(data2D);
}
