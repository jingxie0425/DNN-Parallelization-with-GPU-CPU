#ifndef _READ2D_H_
#define _READ2D_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <ios>
#include <sstream>

class Read2D
{
private:
	const char* folder;
    float* data2D;
    unsigned int PicNum1;
    unsigned int PicNum2;
    unsigned int PicSize;

public:
    //PicNumIn1 is inputnum; PicNumIn2 is filternum; PicSizeIn is dim*dim
    explicit Read2D(const char* folderIn, unsigned int PicNumIn1, unsigned int PicNumIn2, unsigned int PicSizeIn);
    inline virtual ~Read2D() {}
    virtual void readExecute();
    virtual float* getData();
    virtual void freeRead2D();
};

#endif  //_READFILES_
