#ifndef _DATATRANS_H_
#define _DATATRANS_H_

#include "cnn.h"
#include "dense.h"
#include "readIn.h"
using namespace std;

class DataTrans{
protected:
    READHANDLER cpuH;

public:
    explicit DataTrans(READHANDLER cpuH);
    inline virtual ~DataTrans(){}
    virtual void Trans();
    virtual void TransBack(float* cpu_d3, float* gpuout,unsigned int oD,unsigned int oN);
    virtual void TransBackDense(DENSEHANDLER densehandler, float* realOut,unsigned int inN,unsigned int nodeN);

};

#endif  //_DATATRANS_H_
