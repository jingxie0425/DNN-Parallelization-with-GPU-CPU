
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "lide_c_util.h"

#include "lide_c_multiCNN_graph.h"


int main(int argc, char **argv) {
    const char* WC1 = "../../../extractParas_mod2_pruned/cnn1Weight.csv";
    const char* BC1 = "../../../extractParas_mod2_pruned/cnn1Bias.csv";
    const char* WC2 = "../../../extractParas_mod2_pruned/cnn2Weight.csv";
    const char* BC2 = "../../../extractParas_mod2_pruned/cnn2Bias.csv";
    const char* InputsFile = "../../../extractParas_mod2_pruned/testInputX.csv";

    const char* WFD = "../../../extractParas_mod2_pruned/dense1Weight.csv";
    const char* BFD = "../../../extractParas_mod2_pruned/dense1Bias.csv";
    const char* WD2 = "../../../extractParas_mod2_pruned/dense2Weight.csv";
    const char* BD2 = "../../../extractParas_mod2_pruned/dense2Bias.csv";
    const char* WD3 = "../../../extractParas_mod2_pruned/dense3Weight.csv";
    const char* BD3 = "../../../extractParas_mod2_pruned/dense3Bias.csv";
    const char* WD4 = "../../../extractParas_mod2_pruned/dense4Weight.csv";
    const char* BD4 = "../../../extractParas_mod2_pruned/dense4Bias.csv";
    

    const char* OutFile = "../result.csv";

    lide_c_graph_context_type *graph = NULL;

    /* Create graph*/                               
    graph = (lide_c_graph_context_type*)lide_c_multiCNN_graph_new(WC1,BC1, 
    InputsFile,WC2,BC2,WFD,BFD,WD2,BD2,WD3,BD3,WD4,BD4,OutFile);
    //execute graph
    graph->scheduler(graph);

    return 0;
}