
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "lide_c_util.h"

#include "lide_c_multiMLP_graph.h"


int main(int argc, char **argv) {
    
    const char* InputsFile = "../../../extractParas_mod1/testInputX.csv";
    const char* WDense1 = "../../../extractParas_mod1/dense1Weight.csv";
    const char* BDense1 = "../../../extractParas_mod1/dense1Bias.csv";
    const char* WDense2 = "../../../extractParas_mod1/dense2Weight.csv";
    const char* BDense2 = "../../../extractParas_mod1/dense2Bias.csv";
    const char* WDense3 = "../../../extractParas_mod1/dense3Weight.csv";
    const char* BDense3 = "../../../extractParas_mod1/dense3Bias.csv";
    const char* WDense4 = "../../../extractParas_mod1/dense4Weight.csv";
    const char* BDense4 = "../../../extractParas_mod1/dense4Bias.csv";
            
    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_multiMLP_graph_new(InputsFile, WDense1, BDense1, WDense2, 
        BDense2, WDense3, BDense3, WDense4, BDense4, OutFile );  
    //execute graph
    graph->scheduler(graph);

    return 0;
}