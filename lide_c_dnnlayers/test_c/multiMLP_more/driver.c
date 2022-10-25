#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "lide_c_util.h"
#include "lide_c_mlpmulti_graph.h"
int main(int argc, char **argv) {
    const char* InputsFile = "../../../../graphGen/extractedParas/testInputXDEN.csv";
    const char* OutFile = "../result.csv";
    const char* Wfst="../../../../graphGen/extractedParas/fstWeight.csv";
    const char* Bfst = "../../../../graphGen/extractedParas/fstBias.csv";
    const char* Wsnd="../../../../graphGen/extractedParas/sndWeight.csv";
    const char* Bsnd = "../../../../graphGen/extractedParas/sndBias.csv";
    const char* Wtrd="../../../../graphGen/extractedParas/trdWeight.csv";
    const char* Btrd = "../../../../graphGen/extractedParas/trdBias.csv";
    const char* Wfth="../../../../graphGen/extractedParas/fthWeight.csv";
    const char* Bfth = "../../../../graphGen/extractedParas/fthBias.csv";
    lide_c_graph_context_type *graph = NULL;
    graph = (lide_c_graph_context_type*)lide_c_mlpmulti_graph_new(InputsFile,
    Wfst, Bfst,
    Wsnd, Bsnd,
    Wtrd, Btrd,
    Wfth, Bfth,
    OutFile);
    graph->scheduler(graph);
    return 0;
}
