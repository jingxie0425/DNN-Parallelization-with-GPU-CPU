#ifndef _READ1D_H_
#define _READ1D_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <ios>
#include <sstream>

class Read1D
{
private:
	std::fstream fin1D;
    float* data1D;
    unsigned int size1D;

public:
    explicit Read1D(const char* file, unsigned int size);
    inline virtual ~Read1D() {}
    virtual void readExecute();
    virtual float* getData();
    virtual void freeRead1D();
};

#endif  //_READFILES_
