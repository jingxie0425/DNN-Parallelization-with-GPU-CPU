/*
 * @author: xiaomin wu
 * @date: 1/16/2020
 * */
#include "readIn.h"

void testReadWeights(){
    ReadFiles *reader = new ReadFiles(17, 78, 2, 32);
    reader->readWeights();
}

void testReadInputs(){
    ReadFiles *reader = new ReadFiles(17, 78, 2, 32);
    reader->readInputs();
}

void testReadOutputs(){
    ReadFiles *reader = new ReadFiles(17, 78, 2, 32);
    reader->readExpectOuts();
}

void testReadBias(){
    ReadFiles *reader = new ReadFiles(17, 78, 2, 32);
    reader->readBias();
}


/*
int main(){

    //testReadBias();
    //testReadWeights();
    //testReadInputs();
	testReadOutputs();
    return 0;
}
*/
