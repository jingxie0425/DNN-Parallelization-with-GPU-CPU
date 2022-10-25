
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "lide_c_util.h"

#include "lide_c_singleMLP_graph.h"


int main(int argc, char **argv) {
    
    const char* InputsFile = "../../../extractParas_mod4/testInputX.csv";
    const char* WDense1 = "../../../extractParas_mod4/dense1Weight.csv";
    const char* BDense1 = "../../../extractParas_mod4/dense1Bias.csv";
    const char* WDense2 = "../../../extractParas_mod4/dense2Weight.csv";
    const char* BDense2 = "../../../extractParas_mod4/dense2Bias.csv";
     
    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_singleMLP_graph_new(InputsFile, WDense1, BDense1, WDense2, 
        BDense2, OutFile);  
    //execute graph
    graph->scheduler(graph);

    return 0;
}