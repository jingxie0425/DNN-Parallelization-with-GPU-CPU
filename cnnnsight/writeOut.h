#ifndef _WRITEOUT_H_
#define _WRITEOUT_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <ios>
#include <sstream>

void writeTrans(float* cpudata,float* gpudata, unsigned int d1,unsigned int d2);

class WriteOut
{
private:
	std::fstream fout;
    float* cpudata;
    unsigned int D1;
    unsigned int D2;

public:
    explicit WriteOut(const char* file, unsigned int d1,unsigned int d2);
    inline virtual ~WriteOut() {}
    virtual void writeExecute(float* gpudataptr);
    virtual void freeWriteOut();
    virtual void outConvertResult();
    virtual void outDirectResult();
};

#endif  //_READFILES_
