
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "lide_c_util.h"

#include "lide_c_simpleCnn_graph.h"


int main(int argc, char **argv) {
    const char* WConvFile = "../../../extractParas/cnnWeight.csv";
    const char* BConvFile = "../../../extractParas/cnnBias.csv";
    const char* InputsFile = "../../../extractParas/testInputX.csv";
    const char* WFlattenDenseFile = "../../../extractParas/dense1Weight.csv";
    const char* BFlattenDenseFile = "../../../extractParas/dense1Bias.csv";
    const char* WDenseFile = "../../../extractParas/dense2Weight.csv";
    const char* BDenseFile = "../../../extractParas/dense2Bias.csv";
    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_simpleCnn_graph_new(WConvFile,BConvFile,InputsFile,WFlattenDenseFile,BFlattenDenseFile,WDenseFile,BDenseFile,OutFile);
    //execute graph
    graph->scheduler(graph);

    return 0;
}