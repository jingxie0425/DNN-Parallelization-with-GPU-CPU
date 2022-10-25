#ifndef _READFILES_
#define _READFILES_

typedef struct{
        unsigned int inputDim;
        unsigned int inputNum;
        unsigned int filterNum;
        unsigned int filterDim;
        unsigned int outputDim;
        unsigned int outputNum;

        float* filterWeights; //three D arrary store filter weights
        float* inputs;  //3D ary store inputs
        float* outputs; //3D ary store outputs
        float* bias;
}READSTATE;
typedef READSTATE* READHANDLER;

class ReadFiles
{ 
private:

    READHANDLER cnnhandler;
    float* maxpoolout;
    float* dense1Out;
    float* dense1W;
    float* dense1B;
    float* dense2Out;
    float* dense2W;
    float* dense2B;

public:
    explicit ReadFiles(int inD, int inN, int fD, int fN);
    inline virtual ~ReadFiles() {}
    virtual void readWeights();
    virtual void readBias();
    virtual void readInputs();
    virtual void readExpectOuts();
    virtual void readExpectMpOuts();
    virtual READHANDLER getHandler();
    virtual float* getmpout();
    virtual float* getDense1W();
    virtual float* getDense1B();
    virtual float* getDense1Out();
    virtual float* readDense1Weights(int inNum, int nodeNum);
    virtual float* readDense1Bias(int nodeNum);
    virtual float* readDense1ExpectedOut(int nodeNum);

    virtual float* getDense2W();
    virtual float* getDense2B();
    virtual float* getDense2Out();
    virtual float* readDense2Weights(int inNum, int nodeNum);
    virtual float* readDense2Bias(int nodeNum);
    virtual float* readDense2ExpectedOut(int nodeNum);

};

#endif  //_READFILES_
