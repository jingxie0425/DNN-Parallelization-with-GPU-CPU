#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "lide_c_util.h"
#include "lide_c_cnnsingle_graph.h"
int main(int argc, char **argv) {
    const char* InputsFile = "../../../../graphGen/extractedParas/testInputXCNN.csv";
    const char* OutFile = "../result.csv";
    const char* Wconv1="../../../../graphGen/extractedParas/conv1Weight.csv";
    const char* Bconv1 = "../../../../graphGen/extractedParas/conv1Bias.csv";
    const char* Wconv2="../../../../graphGen/extractedParas/conv2Weight.csv";
    const char* Bconv2 = "../../../../graphGen/extractedParas/conv2Bias.csv";
    const char* Wdense1="../../../../graphGen/extractedParas/dense1Weight.csv";
    const char* Bdense1 = "../../../../graphGen/extractedParas/dense1Bias.csv";
    const char* Wdense2="../../../../graphGen/extractedParas/dense2Weight.csv";
    const char* Bdense2 = "../../../../graphGen/extractedParas/dense2Bias.csv";
    lide_c_graph_context_type *graph = NULL;
    graph = (lide_c_graph_context_type*)lide_c_cnnsingle_graph_new(InputsFile,
    Wconv1, Bconv1,
    Wconv2, Bconv2,
    Wdense1, Bdense1,
    Wdense2, Bdense2,
    OutFile);
    graph->scheduler(graph);
    return 0;
}
